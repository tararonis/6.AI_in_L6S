from sberpm._holder import DataHolder
from sberpm.miners._abstract_miner import AbstractMiner
from sberpm.miners.mining_utils._node_utils import ProcessTreeNode
from sberpm.miners._simple_miner import SimpleMiner
from sberpm.visual._types import NodeType  # FIXME partially initialized import


class InductiveMiner(AbstractMiner):
    """
    Realization of an inductive miner algorithm.

    Parameters
    ----------
    data_holder: sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    parallel_activity: bool, default=True
        If True and a normal cut during an iteration is not found, try to find a node
        so that the graph without it has a cut. If such node is found,
        (EXCLUSIVE_CHOICE between the node and hidden activity) group
        becomes parallel to the graph without the node.


    Attributes
    ----------
    graph: ProcessTreeNode
        Graph in the form of a process tree. The leaves are the process activities,
        the other nodes are the operators.

    Examples
    --------
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import InductiveMiner
    >>>
    >>> # Create data_holder
    >>> df = pd.DataFrame({
    ...     'id_column': [1, 1, 2],
    ...     'activity_column':['st1', 'st2', 'st1'],
    ...     'dt_column':[123456, 123457, 123458]})
    >>> data_holder = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> miner = InductiveMiner(data_holder)
    >>> miner.apply()


    References
    ----------
    S.J.J. Leemans, D. Fahland, and W.M.P. van der Aalst. Discovering Block-Structured
    Process Models From Event Logs - A Constructive Approach. 2013
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.396.197&rep=rep1&type=pdf
    """

    def __init__(self, data_holder: DataHolder, parallel_activity: bool = True):
        super().__init__(data_holder)
        self.parallel_activity = parallel_activity

    def apply(self):
        """
        Starts the calculation of the graph using the miner.
        """
        miner = SimpleMiner(self._data_holder)
        miner.apply()
        graph = miner.graph

        # Edit graph (remove start_event and end_event)
        start_nodes = {edge.target_node.id for edge in graph.nodes[NodeType.START_EVENT].output_edges}
        end_nodes = {edge.source_node.id for edge in graph.nodes[NodeType.END_EVENT].input_edges}

        graph.remove_node_by_id(NodeType.START_EVENT)
        graph.remove_node_by_id(NodeType.END_EVENT)

        root_node = ProcessTreeNode(self.parallel_activity, graph, start_nodes, end_nodes)
        root_node.unite_operators()
        self.graph = root_node


def inductive_miner(data_holder: DataHolder, parallel_activity: bool = True) -> ProcessTreeNode:
    """
    Realization of an inductive miner algorithm.

    Parameters
    ----------
    data_holder: sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    parallel_activity: bool, default=True
        If True and a normal cut during an iteration is not found, try to find a node
        so that the graph without it has a cut. If such node is found,
        (EXCLUSIVE_CHOICE between the node and hidden activity) group
        becomes parallel to the graph without the node.

    Returns
    -------
    graph : Graph

    References
    ----------
    S.J.J. Leemans, D. Fahland, and W.M.P. van der Aalst. Discovering Block-Structured
    Process Models From Event Logs - A Constructive Approach. 2013

    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.396.197&rep=rep1&type=pdf
    """
    miner = InductiveMiner(data_holder, parallel_activity)
    miner.apply()
    return miner.graph
