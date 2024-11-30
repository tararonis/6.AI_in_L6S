from numpy import zeros
from pandas import DataFrame

from sberpm.visual._types import NodeType


class TokenReplay:
    """
    Conformance Checking Using Token-Based Replay.

    Features of the algorithm:
     - Before replaying the trace one token is produced and placed at the start_event.
     Then the start_event acts like a usual place, some tokens can be added to it (missing ones).
     - After replaying the end_event is supposed to contain at least one token,
     if its not there, it is created (missing token). Then the token at the end_event is consumed.
     If some tokens are present at the end event after that, they are considered to be the remaining ones.

    Parameters
    ----------
    data_holder: sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    graph: sberpm.visual._graph.Graph
        Petri-net of the process.

    Attributes
    ----------
    _data_holder: sberpm.DataHolder
        Object that contains the event log and the names of its necessary columns.

    _graph: sberpm.visual._graph.Graph
        Petri-net of the process.

    result: pd.DataFrame, shape[0]=number_of_unique_ids
        The result of the token replay for every event trace.
        Columns: data_holder.id_column: id,
                 data_holder.activity_column: the event trace,
                 'c': number of consumed tokens of the event trace,
                 'p': number of produced tokens of the event trace,
                 'm': number of missing tokens of the event trace,
                 'r': number of remaining tokens of the event trace,
                 'fitness': fitness value of the event trace.

    mean_fitness: float
        Mean of all calculated fitness values for every event trace.

        =mean(fitness_1, fitness_2, ... , fitness_n),
        where fitness_i - fitness value of the i-th event trace.

    average_fitness: float
        Value that is calculated using the total numbers of consumed, produced, missing and remaining tokens
        after replaying all the event traces.

        =calc_fitness((c_1 + ... + c_n), (p_1 + ... + p_n), (m_1 + ... + m_n), (r_1 + ... + r_n)))
        where c_i, p_i, m_i, r_i - numbers of consumed, produced, missing and remaining tokens of the i-th event trace.

    """

    def __init__(self, data_holder, graph):
        self._data_holder = data_holder
        self._graph = graph
        self.result = None
        self.mean_fitness = None
        self.average_fitness = None

    def apply(self):
        """
        Calculates token replay.
        """
        unique_id_number = self._data_holder.data[self._data_holder.id_column].nunique()
        c_list = zeros((unique_id_number,), dtype=int)
        p_list = zeros((unique_id_number,), dtype=int)
        m_list = zeros((unique_id_number,), dtype=int)
        r_list = zeros((unique_id_number,), dtype=int)

        cached_traces = {}

        replay_graph = self._create_replay_graph()
        traces = self._data_holder.get_grouped_columns(self._data_holder.activity_column)
        for i, trace in enumerate(traces):
            if trace not in cached_traces:
                c, p, m, r = self._calc_coeffs(trace, replay_graph)
                cached_traces[trace] = [c, p, m, r]
            else:
                c, p, m, r = cached_traces[trace]
            c_list[i] = c
            p_list[i] = p
            m_list[i] = m
            r_list[i] = r

        fitness_list = self._calc_fitness(c_list, p_list, m_list, r_list)  # np.array, shape=(unique_id_number, )
        df = DataFrame(
            {
                self._data_holder.id_column: self._data_holder.grouped_data[self._data_holder.id_column],
                self._data_holder.activity_column: traces,
                "c": c_list,
                "p": p_list,
                "m": m_list,
                "r": r_list,
                "fitness": fitness_list,
            }
        )
        self.mean_fitness = fitness_list.mean()
        self.average_fitness = self._calc_fitness(c_list.sum(), p_list.sum(), m_list.sum(), r_list.sum())

        self.result = df

    def _create_replay_graph(self):
        """
        Transforms the graph (Petri-net) to a different representation
        that will be suitable for replaying the event traces.

        Returns
        -------
        replay_graph: ReplayGraph

        """
        nodes = {}
        start_event = None
        end_event = None
        for gnode in self._graph.get_nodes():
            if gnode.type == NodeType.TASK:
                n = ReplayActivity()
            else:
                n = ReplayPlace()
                if gnode.type == NodeType.START_EVENT:
                    start_event = n
                elif gnode.type == NodeType.END_EVENT:
                    end_event = n
            nodes[gnode.id] = n
        for edge in self._graph.get_edges():
            source_node = nodes[edge.source_node.id]
            target_node = nodes[edge.target_node.id]
            source_node.outgoing.append(target_node)
            target_node.incoming.append(source_node)
        return ReplayGraph(nodes, start_event, end_event)

    # TODO refactor
    @staticmethod
    def _calc_coeffs(trace, replay_graph):
        """
        Replays the given trace using the given Petri-net.

        Parameters
        ----------
        trace: list of str
            The event trace.

        replay_graph: ReplayGraph
            Representation of the Petri-net suitable for replaying event traces.

        Returns
        -------
        c, p, m, r: int
            Number of consumed, produced, missing and remaining tokens respectively.
        """
        replay_graph.start_event.token_num = 1
        c, p, m, r = 0, 1, 0, 0
        for act in trace:
            node = replay_graph.nodes[act]
            if len(node.incoming) == 0 or len(node.outgoing) == 0:
                continue

            # Remove a token from every incoming node
            for incoming_place in node.incoming:
                if incoming_place.has_token():
                    incoming_place.remove_token()
                else:
                    m += 1

            c += len(node.incoming)
            # Add a token to every outgoing node
            for outgoing_place in node.outgoing:
                outgoing_place.add_token()
            p += len(node.outgoing)

        # End_event must have a token, consume it.
        if replay_graph.end_event.has_token():
            replay_graph.end_event.remove_token()
        else:
            m += 1
        c += 1

        # Check for remaining tokens
        for node in replay_graph.nodes.values():
            if type(node) == ReplayPlace:
                r += node.token_num
                node.token_num = 0

        return c, p, m, r

    @staticmethod
    def _calc_fitness(consumed_tokens, produced_tokens, missing_tokens, remaining_tokens):
        """
        Calculates fitness.
        Fitness shows how well the graph describes the business process or a single event trace.

        Parameters
        ----------
        consumed_tokens : int or ndarray
            Number of consumed tokens.
        produced_tokens : int or ndarray
            Number of produced tokens.
        missing_tokens : int or ndarray
            Number of missing tokens.
        remaining_tokens : int or ndarray
            Number of remaining tokens.

        Returns
        -------
        fitness: float or ndarray of float
        """
        return ((1 - missing_tokens / consumed_tokens) + (1 - remaining_tokens / produced_tokens)) / 2


class ReplayAbstractNode:
    def __init__(self):
        """
        Abstract node that is suitable for token replay.

        Attributes
        ----------
        incoming: list of ReplayAbstractNode
            Incoming nodes.

        outgoing: list of ReplayAbstractNode
            Outgoing nodes.
        """
        self.incoming = []
        self.outgoing = []


class ReplayPlace(ReplayAbstractNode):
    def __init__(self):
        """
        Representation of a place that is suitable for token replay.

        Attributes
        ----------
        token_num: int
            Number of tokens that are present in the place.
        """
        super().__init__()
        self.token_num = 0

    def has_token(self):
        """
        Checks whether the place has at least one token.

        Returns
        -------
        result: bool
            True, if the place has at least one token, False otherwise.
        """
        return self.token_num > 0

    def remove_token(self):
        """
        Removes a token from the place.
        """
        self.token_num -= 1

    def add_token(self):
        """
        Adds a token to the place.
        """
        self.token_num += 1


class ReplayActivity(ReplayAbstractNode):
    """
    Representation of an activity node that is suitable for token replay.
    """


class ReplayGraph:
    def __init__(self, nodes, start_event, end_event):
        """
        Representation of a Petri-net that is suitable for token replay.

        Parameters
        ----------
        nodes: dict of {str: ReplayAbstractNode}
            Nodes that represent a Petri-net.

        start_event: ReplayPlace
            Node that represents a start_event.

        end_event: ReplayPlace
            Node that represents an end_event.

        Attributes
        ----------
        The same.
        """
        self.nodes = nodes
        self.start_event = start_event
        self.end_event = end_event
