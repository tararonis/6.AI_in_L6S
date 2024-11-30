from __future__ import annotations

from typing import Iterable, Mapping
from collections import defaultdict

from numpy import isin

from sberpm._holder import DataHolder
from sberpm.miners._alpha_miner import AlphaMiner
from sberpm.visual._graph import Graph, create_petri_net


class AlphaPlusMiner(AlphaMiner):
    """
    Realization of an Alpha+ Miner algorithm.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names of its necessary columns.

    n_jobs: int, default=1
        Maximum number of processes created if possible.
        If n_jobs > 0 - max number of processes = n_jobs;
        if n_jobs < 0 - max number of processes = mp.cpu_count() + 1 + n_jobs;
        in other cases: max number of processes = 1.

    Examples
    --------
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import AlphaPlusMiner
    >>>
    >>> # Create data_holder
    >>> df = DataFrame({
    ...     'id_column': [1, 1, 2],
    ...     'activity_column':['st1', 'st2', 'st1'],
    ...     'dt_column':[123456, 123457, 123458]})
    >>> data_holder = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> miner = AlphaPlusMiner(data_holder)
    >>> miner.apply()

    Notes
    -----
    This algorithm can not handle loops of lengths 1 (and 2).

    References
    ----------
    A.K.A. de Medeiros, B.F. van Dongen, W.M.P. van der Aalst, and A.J.M.M. Weijters. Process Mining:
    Extending the α-algorithm to Mine Short Loops
    W.M.P. van der Aalst, A.J.M.M. Weijters, and L. Maruster. Workflow Mining: Discovering Process Models
    from Event Logs. IEEE Transactions on Knowledge and Data Engineering (TKDE), Accepted for publication, 2003
    """

    def apply(self):
        """
        Starts the calculation of the graph using the miner.
        """
        activity_follows_dict = super()._get_following_activity_mapping()
        loop_1_activities = self._get_loop_1_activities(activity_follows_dict)
        follows_pairs_without_l1a = self._get_following_activity_mapping_without(
            activity_follows_dict, without_activities=loop_1_activities
        )
        unique_activities = self._data_holder.get_unique_activities()
        unique_activities_without_l1a = unique_activities[~isin(unique_activities, loop_1_activities)]

        # places = super().find_places(super()._get_following_activity_mapping(), unique_activities_without_l1a)
        places = super().find_places(follows_pairs_without_l1a, unique_activities_without_l1a)

        # Create and fill the graph
        graph = create_petri_net()
        super()._create_act_nodes(graph, unique_activities)  # all transitions (including l1a)
        super()._create_start_end_events_and_edges(
            graph, *self._get_first_last_activities_without(loop_1_activities)
        )
        super().create_places_and_edges(graph, places)

        # Add edges from/to loop_1_activities to the graph
        incoming_l1a, outgoing_l1a = self._get_incoming_outgoing_for_l1_activities(
            activity_follows_dict, loop_1_activities
        )
        self.add_l1a_edges(graph, incoming_l1a, outgoing_l1a, places)
        self.graph = graph

    def _get_first_last_activities_without(self, activities_to_remove):
        """
        Returns the first end last activities of the event traces in the event log
        that does not contain given activities.

        Parameters
        ----------
        activities_to_remove: set of str
            Activities that needed to be removed from the event log.

        Returns
        -------
        first_activities : list of str
            Names of the activities that event traces in the shortened event log start with.

        last_activities : list of str
            Names of the activities that event traces in the shortened event log end with.

        """
        id_column = self._data_holder.id_column
        activity_column = self._data_holder.activity_column
        mask = ~self._data_holder.data[activity_column].isin(activities_to_remove)
        df = self._data_holder.data[[id_column, activity_column]][mask]
        mask_first = df[id_column] != df[id_column].shift(1)
        mask_last = df[id_column] != df[id_column].shift(-1)
        return df[activity_column][mask_first].unique(), df[activity_column][mask_last].unique()

    @staticmethod
    def _get_incoming_outgoing_for_l1_activities(
        activity_follows_dict: Mapping[str, set[str]], loop_1_activities: set[str]
    ) -> tuple[Mapping[str, Iterable[str]], Mapping[str, Iterable[str]]]:
        """
        Returns incoming_l1a and outgoing_l1a non-loop-one activities for each loop-one activity.

        Parameters
        ----------
        activity_follows_dict: Mapping[str, set[str]]
            'key' activities that are followed by 'value' activities in the event log

        loop_1_activities: set[str]
            Activities that create loops of length one

        Returns
        -------
        incoming_l1a: Mapping[str, set[str]]
            Key: loop-one activity. Value: set of incoming_l1a non-loop-one activities.

        outgoing_l1a: Mapping[str: set[str]]
            Key: loop-one activity. Value: set of outgoing_l1a non-loop-one activities.
        """
        incoming_l1a = defaultdict(set)
        outgoing_l1a = defaultdict(set)

        for source_activity, activity_targets in activity_follows_dict.items():
            if source_activity in loop_1_activities:
                outgoing_l1a[source_activity].update(activity_targets.difference(loop_1_activities))
            else:
                incoming_l1a[source_activity].update(activity_targets.intersection(loop_1_activities))

        return incoming_l1a, outgoing_l1a

    @staticmethod
    def _get_loop_1_activities(activity_follows_dict: Mapping[str, set[str]]) -> Iterable[str]:
        """
        Returns activities that create one-length loops.

        Parameters
        ----------
        activity_follows_dict: Mapping[str, set[str]]
            'key' activities that are followed by 'value' activities in the event log.

        Returns
        -------
        loop_1_activities: set[str]
            Set of activities that create one-length loops.
        """
        return {
            source_act
            for source_act in activity_follows_dict
            if any(source_act == activity for activity in activity_follows_dict[source_act])
        }

    def _get_following_activity_mapping_without(
        self, transitions_activity_mapping: Mapping[str, set[str]], without_activities: Iterable[str]
    ) -> Mapping[str, set[str]]:
        """
        Returns follows pairs from the event log that does not contain given activities.

        Parameters
        ----------
        without_activities: Iterable[str]
            Activities that needed to be removed from the event log

        Returns
        -------
        Mapping[str, set[str]]
            Unique pairs of activities that present in the event log that does not contain given activities
        """
        return defaultdict(
            set,
            {
                source_activity: activity_targets.difference(without_activities)
                for source_activity, activity_targets in transitions_activity_mapping.items()
                if source_activity not in without_activities
            },
        )

    # TODO refactor
    @staticmethod
    def add_l1a_edges(graph, incoming_l1a, outgoing_l1a, places):
        """
        Adds edges between loop-one transitions and places to the graph if possible.

        Parameters
        ----------
        graph: Graph
            Graph.

        incoming_l1a: dict of {str: set of str}
            Key: loop-one activity. Value: set of incoming non-loop-one activities.

        outgoing_l1a: dict of {str: set of str}
            Key: loop-one activity. Value: set of outgoing non-loop-one activities.

        places: list of tuple of two sets of str
            List of places. A place is represented by two sets of activities (incoming and outgoing).
        """
        for l1act in incoming_l1a.keys():
            if l1act in outgoing_l1a:
                inc_activities = incoming_l1a[l1act]
                out_activities = outgoing_l1a[l1act]
                inc_without_out = inc_activities - out_activities
                out_without_inc = out_activities - inc_activities
                possible_place_id = ",".join(inc_without_out) + " -> " + ",".join(out_without_inc)
                place_id = None
                if possible_place_id in graph.nodes:
                    place_id = possible_place_id
                else:
                    for place in places:
                        if inc_without_out.issubset(place[0]) and out_without_inc.issubset(place[1]):
                            place_id = ",".join(place[0]) + " -> " + ",".join(place[1])
                            break
                if place_id is not None:
                    graph.add_edge(l1act, place_id)
                    graph.add_edge(place_id, l1act)


def alpha_plus_miner(data_holder: DataHolder, n_jobs: int = 1) -> Graph:
    """
    Realization of an Alpha+ Miner algorithm.

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names of its necessary columns.

    n_jobs: int, default=1
        Maximum number of processes created if possible.
        If n_jobs > 0 - max number of processes = n_jobs;
        if n_jobs < 0 - max number of processes = mp.cpu_count() + 1 + n_jobs;
        in other cases: max number of processes = 1.

    Returns
    -------
    graph : Graph

    Notes
    -----
    This algorithm can not handle loops of lengths 1 (and 2).

    References
    ----------
    A.K.A. de Medeiros, B.F. van Dongen, W.M.P. van der Aalst, and A.J.M.M. Weijters. Process Mining:
    Extending the α-algorithm to Mine Short Loops
    W.M.P. van der Aalst, A.J.M.M. Weijters, and L. Maruster. Workflow Mining: Discovering Process Models
    from Event Logs. IEEE Transactions on Knowledge and Data Engineering (TKDE), Accepted for publication, 2003
    """
    miner = AlphaPlusMiner(data_holder, n_jobs)
    miner.apply()

    return miner.graph
