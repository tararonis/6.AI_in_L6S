from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping
from typing_extensions import Protocol

from pandas import DataFrame

from sberpm.visual._graph import Graph
from sberpm.visual._types import NodeType


class MiningDataHolder(Protocol):
    @property
    def id_column(self) -> str:
        ...

    @property
    def activity_column(self) -> str:
        ...

    @property
    def data(self) -> DataFrame:
        ...

    @property
    def grouped_data(self) -> DataFrame:
        """
        The data grouped by id and, as expected, activity
        May have more columns grouped
        """


class AbstractMiner:
    """
    Abstract class for miners.
    Contains fields and methods used by all miners.

    Parameters
    ----------
    data_holder : sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    Attributes
    ----------
    data_holder : sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    graph: sberpm.visual._graph.Graph
        Mined graph of the process.
    """

    def __init__(self, data_holder: MiningDataHolder):
        self._data_holder = data_holder.copy() #TODO WRONG COPY!!!!!
        self.graph = None

    # TODO refactor speed
    def _get_first_last_activities(self):
        """
        Returns activities that event traces start and end with.

        Returns
        -------
        first_activities : list of str
            Names of the activities that event traces start with.

        last_activities : list of str
            Names of the activities that event traces end with.
        """
        activity_column = self._data_holder.activity_column

        if self._data_holder.grouped_data is not None and activity_column in self._data_holder.grouped_data:
            first_activities = set()
            last_activities = set()

            for chain in self._data_holder.grouped_data[activity_column].values:
                first_activities.add(chain[0])
                last_activities.add(chain[-1])

            return sorted(first_activities), sorted(last_activities)
        else:
            id_column = self._data_holder.id_column
            df = self._data_holder.data[[id_column, activity_column]]

            mask_first = df[id_column] != df[id_column].shift(1)
            mask_last = df[id_column] != df[id_column].shift(-1)

            return df[activity_column][mask_first].unique(), df[activity_column][mask_last].unique()

    @staticmethod
    def _get_causal_parallel_pairs(activity_follows_dict: Mapping[str, set[str]]) -> tuple[set[tuple], set[tuple]]:
        """
        Returns two list of pairs of activities that have causal and parallel relation.

        "activity_1" and "activity_2" have causal relation ("activity_1 -> activity_2") if
            "activity_1 > activity_2" and not "activity_2 > activity_1"
        "activity_1" and "activity_2" have parallel relation ("activity_1 || activity_2") if
            "activity_1 > activity_2" and "activity_2 > activity_1"
        "activity_1 > activity_2" means that "activity_1" is directly followed by "activity_2" in at least
            one event trace in the event log.

        Returns
        -------
        causal_pairs : set of tuples (str, str)
            Pairs (activity_1, activity_2) that have causal relation: "activity_1 -> activity_2"

        parallel_pairs : set of tuples (str, str)
            Pairs (activity_1, activity_2) that have parallel relation: "activity_1 || activity_2"
        """

        def get_targets_of_reversible_transitions(source_activity: str) -> Iterable[str]:
            return {
                target_activity
                for target_activity in activity_follows_dict[source_activity]
                if activity_follows_dict.get(target_activity)
                and source_activity in activity_follows_dict.get(target_activity)
            }

        causal_pairs = set()
        parallel_pairs = set()

        for source_activity in activity_follows_dict:
            reversible_targets = get_targets_of_reversible_transitions(source_activity)

            parallel_pairs.update(
                {(source_activity, reversible_target) for reversible_target in reversible_targets}
            )
            causal_pairs.update(
                {
                    (source_activity, irreversible_target)
                    for irreversible_target in activity_follows_dict[source_activity].difference(
                        reversible_targets
                    )
                }
            )

        return causal_pairs, parallel_pairs

    def __construct_transitions_mapping(self, activity_transition_pairs: zip[tuple]):
        transitions_mapping = defaultdict(set)

        for activity_pair in activity_transition_pairs:
            source_act, target_act = activity_pair
            transitions_mapping[source_act].add(target_act)

        return transitions_mapping

    def __assign_activity_transitions_per_process(
        self, data: DataFrame, id_column: str, activity_name: str
    ) -> DataFrame:
        where_same_process = data[id_column] == data[id_column].shift(-1)

        return data[where_same_process].assign(
            **dict(
                current_activities=data[activity_name],
                following_activities=data[activity_name].shift(-1),
            )
        )

    # TODO refactor speed
    def _get_following_activity_mapping(self) -> Mapping[str, set[str]]:
        """
        Returns defaultdict of mapped unique transitions for each 'transition source activity':
            activity_n -> {activity_k, activity_l, ...}, where "activity_n" is source mapped to bound targets
            from transitions in the event log.

        Returns
        -------
        activity_follows_dict : Mapping[str, set[str]]
            Unique pairs of activities that present in the event log.
        """
        id_column, activity_column = self._data_holder.id_column, self._data_holder.activity_column
        id_activity_data_slice = self._data_holder.data[[id_column, activity_column]]

        df_current_following = self.__assign_activity_transitions_per_process(
            id_activity_data_slice, id_column, activity_column
        )

        return self.__construct_transitions_mapping(
            zip(df_current_following["current_activities"], df_current_following["following_activities"]),
        )

    @staticmethod
    def _map_encode_labels_to_activities(unique_activities: Iterable[str]) -> Mapping[int, str]:
        """
        Add encoding labels for the activities.

        # TODO rm or mv desc to alpha_miner
        Numbers corresponding to the activities will be used as rows/columns
        in a matrix of edges between the activities.

        Returns
        -------
        Mapping[int, str]
            Mapping of numeric labels to corresponding unique activities
        """
        return dict(enumerate(unique_activities))

        # ----------------------------------------------------------------------------------
        # -----------------------------   Graph methods   ----------------------------------
        # ----------------------------------------------------------------------------------

    # TODO bound to graph
    @staticmethod
    def _add_event_edges(
        graph: Graph, event_activities: Iterable[str], event_type: NodeType, reversed=False
    ) -> None:
        AbstractMiner._create_act_nodes(graph, activities=[event_type], add_labels=False, event_type=event_type)

        for activity in event_activities:
            if reversed:
                graph.add_edge(activity, event_type)
            else:
                graph.add_edge(event_type, activity)

    # TODO bound to graph
    @staticmethod
    def _create_act_nodes(
        graph: Graph, activities: Iterable[str], add_labels=True, event_type=NodeType.TASK
    ) -> None:
        """
        Creates nodes for given activities.

        Parameters
        ----------
        graph: sberpm.visual._graph.Graph
            Graph.

        activities: list of str
            List of the activities.
        """
        for activity_name in activities:
            label = add_labels and activity_name or ""
            graph.add_node(node_id=activity_name, label=label, node_type=event_type)

    # TODO bound to graph
    @staticmethod
    def _create_start_end_events_and_edges(
        graph: Graph, first_activities: Iterable[str], last_activities: Iterable[str]
    ) -> None:
        """
        Creates nodes for activities and two artificial nodes: "start" node and "end" node.
        Creates edges between artificial nodes and first/last transitions (nodes that represent activities).

        Parameters
        ----------
        graph: sberpm.visual._graph.Graph
            Graph.

        first_activities: list of str
            The starting activities in one or more event traces.

        last_activities: list of str
            The last activities in one or more event traces.
        """
        AbstractMiner._add_event_edges(graph, first_activities, NodeType.START_EVENT)
        AbstractMiner._add_event_edges(graph, last_activities, NodeType.END_EVENT, reversed=True)
