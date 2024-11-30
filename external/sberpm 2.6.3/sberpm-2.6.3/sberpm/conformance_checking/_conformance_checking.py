from math import sqrt

from sberpm.conformance_checking._token_replay import TokenReplay
from sberpm.visual._types import GraphType, NodeType


class ConformanceChecking:
    """
    Class for conformance checking.
    It presents a number of metrics showing how well
    the graph describes the event log it was built from.

    Parameters
    ----------
    dh: sberpm.DataHolder
        DataHolder of log file.

    graph: sberpm.visual.Graph
        Petri Net.

    Attributes
    ---------
    mean_fitness: float
        Mean fitness of model. Calculated with TokenReplay algorithm.

    average_fitness: float
        Average fitness of model. Calculated with TokenReplay algorithm.

    simplicity: float
        Simplicity of model. Calculated as inverse arc degree.

    generalization: float
        Generalization of model.

    precision: float
        Precision of model.

    Examples
    --------
    >>> from sberpm.conformance_checking import ConformanceChecking
    >>> miner = sberpm.miners.AlphaMiner(dh)
    >>> miner.apply()
    >>> conf_check = ConformanceChecking(dh, miner.graph)
    >>> conf_check.get_conformance_checking()
    """

    def __init__(self, dh, graph):
        if graph.type != GraphType.PETRI_NET:
            raise TypeError(f"Graph must be a Petri Net, but got {graph.type}")
        self._value_counts = dh.get_grouped_data(dh.activity_column)[dh.activity_column].value_counts()
        self._graph = graph
        self._unique_act = dh.get_unique_activities()
        self._m = len(graph.get_nodes())
        self._token_replay = TokenReplay(dh, graph)
        self.dh = dh

        self.mean_fitness = None
        self.average_fitness = None
        self.simplicity = None
        self.generalization = None
        self.precision = None

    def get_conformance_checking(self):
        """
        Return parameters of conformance checking as a dictionary.
        """
        if self._check_params():
            self._apply()

        return {
            "mean_fitness": self.mean_fitness,
            "average_fitness": self.average_fitness,
            "precision": self.precision,
            "generalization": self.generalization,
            "simplicity": self.simplicity,
        }

    def print_conformance_checking(self):
        """
        Print parameters of conformance checking.
        """
        if self._check_params():
            self._apply()
        print("mean_fitness:\t", self.mean_fitness)
        print("average_fitness:", self.average_fitness)
        print("precision:\t", self.precision)
        print("generalization:\t", self.generalization)
        print("simplicity:\t", self.simplicity)

    def get_fitness_df(self):
        """
        Return TokenReplay result.
        """
        if self._check_params():
            self._apply()
        return self._token_replay.result

    def _apply(self):
        """
        Calculate metrics of conformance checking.
        """
        self.mean_fitness, self.average_fitness = self._get_fitness()
        self.precision = self._get_precision()
        self.generalization = self._get_generalization()
        self.simplicity = self._get_simplicity()

    def _check_params(self):
        """
        Check parameters on being None.
        """
        return (
            self.mean_fitness is None
            or self.average_fitness is None
            or self.precision is None
            or self.generalization is None
            or self.simplicity is None
        )

    def _get_generalization(self):
        """
        Calculate generalization of a model.

        1 - sum{nodes}(sqrt(#executions)**(-1)) / #nodes
        """
        node_count = {
            node: (
                1
                / sqrt(
                    sum(
                        count * trace.count(node)
                        for trace, count in zip(self._value_counts.index, self._value_counts)
                    )
                )
            )
            for node in self._unique_act
        }

        return 1 - sum(node_count.values()) / (self._m - 2)

    def _get_simplicity(self):
        """
        Calculated simplicity of model.

        1 / (1 + max(mean_num_of_in_and_out_edges_of_a_node + 2, 0))
        """
        total_edges_per_node = sum(
            len(node.input_edges) + len(node.output_edges) for node in self._graph.nodes.values()
        )

        # '2' is needed in the formula so that a simplicity of an ideal graph
        # whose nodes have one input and one output edges (2 edges in total) equals 1
        return 1 / (1.0 + max(total_edges_per_node / len(self._graph.nodes) - 2, 0))

    def _get_fitness(self):
        """
        Calculate fitness of a model.
        """
        self._token_replay.apply()
        return self._token_replay.mean_fitness, self._token_replay.average_fitness

    def _get_precision(self):
        """
        Calculate precision of a model.
        """
        sum_at = 0
        sum_ee = 0
        dict_nodes = self._graph.nodes
        for trace, count in zip(self._value_counts.index, self._value_counts):
            log_trace = set(trace)
            model_trace = set()
            for act in [NodeType.START_EVENT] + list(trace):
                self._add_possible_out_act(model_trace, dict_nodes[act].output_edges)
            model_trace.remove(NodeType.END_EVENT)
            sum_at += count * len(model_trace)
            sum_ee += count * len(model_trace.difference(log_trace))
        return 1 - sum_ee / sum_at

    @staticmethod
    def _add_possible_out_act(model_trace, output_edges):
        for out in output_edges:
            if out.target_node.type == NodeType.PLACE:
                for b in out.target_node.id.split(" -> ")[1].split(","):
                    model_trace.add(b)
            else:
                model_trace.add(out.target_node.id)
