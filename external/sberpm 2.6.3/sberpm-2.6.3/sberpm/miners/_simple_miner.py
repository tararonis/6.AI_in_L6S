from __future__ import annotations

from typing import Mapping


from sberpm._holder import DataHolder
from sberpm.miners._abstract_miner import AbstractMiner
from sberpm.visual._graph import Graph, create_dfg


class SimpleMiner(AbstractMiner):
    """
    Realization of a simple miner algorithm that creates all edges that exist
    according to the event log (no filtration is performed).

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names of its necessary columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import SimpleMiner
    >>>
    >>> # Create data_holder
    >>> df = pd.DataFrame({
    ...     'id_column': [1, 1, 2],
    ...     'activity_column':['st1', 'st2', 'st1'],
    ...     'dt_column':[123456, 123457, 123458]})
    >>> data_holder = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> miner = SimpleMiner(data_holder)
    >>> miner.apply()
    """

    def apply(self):
        """
        Starts the calculation of the graph using the miner.
        """
        unique_activities = self._data_holder.get_unique_activities()
        activity_follows_dict = super()._get_following_activity_mapping()

        graph = create_dfg()
        super()._create_act_nodes(graph, unique_activities)
        super()._create_start_end_events_and_edges(graph, *super()._get_first_last_activities())
        self.create_edges(graph, activity_follows_dict)
        self.graph = graph

    @staticmethod
    def create_edges(graph, activity_transitions_dict: Mapping[str, set[str]]) -> None:
        """
        Adds edges between transitions to the graph.

        Parameters
        ----------
        graph: Graph
            Graph.

        activity_tranisitions_dict: Mapping[str, set[str]]:
            Mapping of activities that have causal relation
        """
        for source_activity, activity_targets in activity_transitions_dict.items():
            _ = [graph.add_edge(source_activity, target) for target in activity_targets]


def simple_miner(data_holder: DataHolder) -> Graph:
    """
    Realization of a simple miner algorithm that creates all edges that exist
    according to the event log (no filtration is performed).

    Parameters
    ----------
    data_holder : DataHolder
        Object that contains the event log and the names of its necessary columns.

    Returns
    -------
    graph : Graph

    """
    miner = SimpleMiner(data_holder)
    miner.apply()
    return miner.graph
