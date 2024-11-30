from enum import Enum
from pickle import dump
from typing import Dict, Tuple

from sberpm.miners.mining_utils import _inductive_utils as utils


class ProcessTreeNodeType(str, Enum):
    """
    Possible types of the nodes in a process tree.
    """

    EXCLUSIVE_CHOICE = "EXCLUSIVE_CHOICE"
    SEQUENTIAL = "SEQUENTIAL"
    PARALLEL = "PARALLEL"
    LOOP = "LOOP"
    SINGLE_ACTIVITY = "SINGLE_ACTIVITY"
    FLOWER = "FLOWER"

    def __str__(self) -> str:
        return self.value


class ProcessTreeNode:
    """
    Node of a process tree.

    It might be an operator or represent a real activity (or an artificial activity === hidden activity).

    If the constructor takes:
    - graph, start_nodes, end_nodes: a type needs to be determined
    - type, no label: it is either an operator or a hidden activity
    - type, label: it is a real process activity.

    Parameters
    ----------
    parallel_activity: bool, default=True
        If Tue and a normal cut during an iteration is not found, try to find a node
        so that the graph without it has a cut. If such node is found,
        (EXCLUSIVE_CHOICE between the node and hidden activity) group
        becomes parallel to the graph without the node.

    graph: Graph
        Graph.

    start_nodes: set of str
        Start nodes of the graph.

    end_nodes: set of str
        Start nodes of the graph.

    type_: str of ProcessTreeNodeType
        Type of the process tree node.

    label: str or None
        If this process tree node represents a real activity, its name, else None.
    """

    def __init__(
        self,
        parallel_activity,
        graph=None,
        start_nodes=None,
        end_nodes=None,
        type_=None,
        label=None,
    ):
        self.id = str(hash(self)) + str(hash(graph))
        self.parallel_act = parallel_activity

        if (graph is None or start_nodes is None or end_nodes is None) and type_ is None:
            raise ValueError("Either {graph, start_nodes and end_nodes} or {type} should be given.")

        self.graph = graph
        self.start_nodes = start_nodes
        self.end_nodes = end_nodes
        self.nodes_metric_names = set()
        self.nodes_metric_data = dict()  # {metric_name: {node:metric_value}}
        self.type = type_
        self.label = label
        self.children = []
        self.apply()

    def unite_operators(self):
        """
        Unites nodes that represent SEQUENTIAL or EXCLUSIVE_CHOICE
        operators if possible to simplify the process tree.
        """
        if self.type in [
            ProcessTreeNodeType.SEQUENTIAL,
            ProcessTreeNodeType.EXCLUSIVE_CHOICE,
            ProcessTreeNodeType.PARALLEL,
        ]:
            again = True
            while again:
                again = False
                new_children = []
                for ch in self.children:
                    if ch.type == self.type:
                        new_children += ch.children
                        again = True
                    else:
                        new_children.append(ch)
                self.children = new_children
        for ch in self.children:
            ch.unite_operators()

    def apply(self):
        """
        Determines a type of the process tree node if not set.
        """
        # Type of the node has already been set.
        if self.type is not None:
            return

        # SINGLE_ACTIVITY
        if len(self.graph.nodes) == 1:
            if len(self.graph.edges) == 0:
                node_id = list(self.graph.nodes.keys())[0]
                self.process_single_real_activity(node_id)
            else:  # there is a self loop, edge: node -> node
                self.process_loop_with_hidden_activity()

            return
        # EXCLUSIVE_CHOICE
        found, node_groups = find_exclusive_choice_cut(self.graph)

        if found:
            self.process_exclusive_choice(node_groups)
            return

        # SEQUENTIAL
        found, node_groups, hidden_activity_mask = find_sequential_cut(
            self.graph, self.start_nodes, self.end_nodes
        )

        if found:
            self.process_sequential(node_groups, hidden_activity_mask)
            return

        # PARALLEL
        found, node_groups = find_parallel_cut(self.graph, self.start_nodes, self.end_nodes)

        if found:
            self.process_parallel(node_groups)
            return

        # LOOP
        found, node_groups = find_loop_cut(self.graph, self.start_nodes, self.end_nodes)

        if found:
            self.process_loop(node_groups)
            return

        # Additional check that removes one node from the graph
        # and checks whether this new graph has a cut. If it has,
        # the (graph with only this node OR hidden activity) group
        # is considered  to be parallel to the new graph.
        if self.parallel_act:
            found, node_to_remove, new_graph_cut_type, new_graph_cut_data = find_cut_without_one_node(
                self.graph, self.start_nodes, self.end_nodes
            )
            if found:
                self.process_special_parallel(node_to_remove, new_graph_cut_type, new_graph_cut_data)
                return

        self.type = ProcessTreeNodeType.FLOWER
        self.children = [
            ProcessTreeNode(self.parallel_act, type_=ProcessTreeNodeType.SINGLE_ACTIVITY, label=node)
            for node in self.graph.nodes.keys()
        ]

        return

    def process_single_real_activity(self, node_id):
        """
        Make this node a leaf node of the process tree
        representing a real single activity.

        Parameters
        ----------
        node_id: str
            Id/name of an activity.
        """
        self.type = ProcessTreeNodeType.SINGLE_ACTIVITY
        self.label = node_id

    def process_loop_with_hidden_activity(self):
        """
        Make this node a loop operator for a real activity and a hidden activity.
        """
        # loop with a hidden activity
        self.type = ProcessTreeNodeType.LOOP
        node_id = list(self.graph.nodes.keys())[0]
        self.children = [
            ProcessTreeNode(self.parallel_act, type_=ProcessTreeNodeType.SINGLE_ACTIVITY, label=node_id),
            ProcessTreeNode(self.parallel_act, type_=ProcessTreeNodeType.SINGLE_ACTIVITY),
        ]

    def process_exclusive_choice(self, node_groups):
        """
        Make this node an exclusive choice operator and launch recursion.

        Parameters
        ----------
        node_groups: list of set of str
            Groups of activities the graph will be split into.
        """
        self.type = ProcessTreeNodeType.EXCLUSIVE_CHOICE
        for graph, start_nodes, end_nodes in utils.cut_graph(
            self.graph, self.start_nodes, self.end_nodes, node_groups
        ):
            self.children.append(ProcessTreeNode(self.parallel_act, graph, start_nodes, end_nodes))

    def process_sequential(self, node_groups, hidden_activity_mask):
        """
        Make this node a sequential operator and launch recursion.

        Parameters
        ----------
        node_groups: list of set of str
            Groups of activities the graph will be split into.

        hidden_activity_mask: list of bool or None
            If an element is True, the corresponding node group should be connected
            to the hidden activity via the EXCLUSIVE_CHOICE operator.
        """
        self.type = ProcessTreeNodeType.SEQUENTIAL
        for (graph, start_nodes, end_nodes), hidden_activity in zip(
            utils.cut_graph(self.graph, self.start_nodes, self.end_nodes, node_groups), hidden_activity_mask
        ):
            if hidden_activity:  # not the last group that has an end node that is not a start node
                tree_node = ProcessTreeNode(self.parallel_act, type_=ProcessTreeNodeType.EXCLUSIVE_CHOICE)
                tree_node.children = [
                    ProcessTreeNode(self.parallel_act, type_=ProcessTreeNodeType.SINGLE_ACTIVITY),
                    ProcessTreeNode(self.parallel_act, graph, start_nodes, end_nodes),
                ]
                self.children.append(tree_node)
            else:
                self.children.append(ProcessTreeNode(self.parallel_act, graph, start_nodes, end_nodes))

    def process_parallel(self, node_groups):
        """
        Make this node a parallel operator and launch recursion.

        Parameters
        ----------
        node_groups: list of set of str
            Groups of activities the graph will be split into.
        """
        self.type = ProcessTreeNodeType.PARALLEL
        for graph, start_nodes, end_nodes in utils.cut_graph(
            self.graph, self.start_nodes, self.end_nodes, node_groups
        ):
            self.children.append(ProcessTreeNode(self.parallel_act, graph, start_nodes, end_nodes))

    def process_loop(self, node_groups):
        """
        Make this node a loop operator and launch recursion.

        Parameters
        ----------
        node_groups: list of set of str
            Groups of activities the graph will be split into.
        """
        self.type = ProcessTreeNodeType.LOOP
        for graph, start_nodes, end_nodes in utils.cut_graph(
            self.graph, self.start_nodes, self.end_nodes, node_groups
        ):
            self.children.append(ProcessTreeNode(self.parallel_act, graph, start_nodes, end_nodes))

    def process_special_parallel(self, node_to_remove, new_graph_cut_type, new_graph_cut_data):
        """
        Make this node a parallel operator between
        (EXCLUSIVE_CHOICE between the graph of the only node_to_remove and hidden activity)
        and (the graph without the node_to_remove).

        Parameters
        ----------
        node_to_remove: str
            Name of the node that must be cut out from the graph.

        new_graph_cut_type: str of ProcessTreeNodeType
            Cut type that exists in the graph without the node_to_remove.

        new_graph_cut_data: object
            Information about the cut in the graph without the node_to_remove.

        """
        self.type = ProcessTreeNodeType.PARALLEL

        # Child 1: EXCLUSIVE_CHOICE between graph with node_to_remove and hidden activity
        graph_ntr = utils.create_dfg()
        graph_ntr.add_node(node_to_remove, node_to_remove)
        if (node_to_remove, node_to_remove) in self.graph.edges:
            graph_ntr.add_edge(node_to_remove, node_to_remove)
        start_nodes_ntr = self.start_nodes.intersection([node_to_remove])
        end_nodes_ntr = self.end_nodes.intersection([node_to_remove])
        ch1 = ProcessTreeNode(self.parallel_act, type_=ProcessTreeNodeType.EXCLUSIVE_CHOICE)
        ch1.children = [
            ProcessTreeNode(self.parallel_act, graph_ntr, start_nodes_ntr, end_nodes_ntr),
            ProcessTreeNode(self.parallel_act, type_=ProcessTreeNodeType.SINGLE_ACTIVITY),
        ]

        # Child 2: graph without the node_to_remove
        new_graph = utils.create_graph_without_nodes(self.graph, [node_to_remove])
        new_start_nodes = self.start_nodes.difference([node_to_remove]).union(
            [node for node, node_obj in new_graph.nodes.items() if len(node_obj.input_edges) == 0]
        )
        new_end_nodes = self.end_nodes.difference([node_to_remove]).union(
            [node for node, node_obj in new_graph.nodes.items() if len(node_obj.output_edges) == 0]
        )

        ch2 = ProcessTreeNode(
            self.parallel_act, new_graph, new_start_nodes, new_end_nodes, type_=new_graph_cut_type
        )

        # Start the appropriate recursion for ch2
        if new_graph_cut_type == ProcessTreeNodeType.SINGLE_ACTIVITY:
            if new_graph_cut_data is None:
                ch2.process_loop_with_hidden_activity()
            else:
                ch2.label = new_graph_cut_data
        elif new_graph_cut_type == ProcessTreeNodeType.EXCLUSIVE_CHOICE:
            ch2.process_exclusive_choice(new_graph_cut_data)
        elif new_graph_cut_type == ProcessTreeNodeType.SEQUENTIAL:
            node_groups, hidden_activities_mask = new_graph_cut_data
            ch2.process_sequential(node_groups, hidden_activities_mask)
        elif new_graph_cut_type == ProcessTreeNodeType.PARALLEL:
            ch2.process_parallel(new_graph_cut_data)
        elif new_graph_cut_type == ProcessTreeNodeType.LOOP:
            ch2.process_loop(new_graph_cut_data)
        else:
            raise ValueError(f"Unknown type: {new_graph_cut_type}")

        self.children = [ch1, ch2]

    def save(self, file_name):
        """
        Save the graph to a file.

        Parameters
        ----------
        file_name: str
            Name of the file.
        """
        with open(file_name, "wb") as f:
            dump(self, f)

    def get_nodes(self) -> Dict[str, "ProcessTreeNode"]:
        d = {self.id: self}
        for ch in self.children:
            d.update(ch.get_nodes())

        return d

    def get_edges(self) -> Dict[Tuple[str, str], Tuple["ProcessTreeNode", "ProcessTreeNode"]]:
        d = {}
        for ch in self.children:
            d[(self.id, ch.id)] = (self, ch)
            d.update(ch.get_edges())

        return d

    def add_node_metric(self, metric_name, metric_data):
        """
        Adds a metric to the 'task' nodes.

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        metric_data : dict of {str: number}
            Metric values of 'task' nodes. Key: id of the 'task' node, value: value of the metric.
            If there is at least one node in given metric_data that is not contained in the graph,
            the metric will not be added.
        """
        self.nodes_metric_names.add(metric_name)
        self.nodes_metric_data[metric_name] = metric_data

    def contains_node_metric(self, node_metric_name):
        """
        Checks whether graph contains given node's metric.

        Parameters
        ----------
        node_metric_name : str
            Name of the metric.

        Returns
        -------
        result : bool
            Returns True if given node's metric's name is preset in graph, False otherwise.
        """
        return node_metric_name in self.nodes_metric_names

    def clear_node_metrics(self):
        """
        Remove all the metrics from the nodes in the graph
        """
        self.nodes_metric_names.clear()
        self.nodes_metric_data = dict()

    def remove_node_metric(self, metric_name: str):
        """
        Remove metric_name from the nodes_metric_names

        Parameters
        ----------
        metric_name : str
            Name of the metric.
        """
        self.nodes_metric_names.remove(metric_name)
        self.nodes_metric_data.pop(metric_name)


def find_exclusive_choice_cut(graph):
    """
    Find an exclusive choice cut.

    Parameters
    ----------
    graph: Graph
        Graph.

    Returns
    -------
    result: bool
        True if found, False otherwise.

    node_groups: list of set of str or None
        If found, returns a list of node groups that the graph should be split into.
    """
    connected_comps = utils.get_weakly_connected_components(graph)
    return [True, connected_comps] if len(connected_comps) > 1 else [False, None]


# TODO refactor
def find_sequential_cut(graph, start_nodes, end_nodes):
    """
    Find a sequential cut.

    Parameters
    ----------
    graph: Graph
        Graph.

    start_nodes: set of str
        Start nodes of the graph.

    end_nodes: set of str
        Start nodes of the graph.

    Returns
    -------
    result: bool
        True if found, False otherwise.

    node_groups: list of set of str or None
        If found, returns a list of consecutive node groups that the graph should be split into.

    hidden_activity_mask: list of bool or None
        If an element is True, the corresponding node group should be connected
        to the hidden activity using the EXCLUSIVE_CHOICE operator.
    """
    strongly_connected_comps = utils.get_strongly_connected_components(graph)
    # Each node of the ggraph is a strongly connected component
    # Name of a node is a string number of a component in strongly_connected_comps list
    ggraph, start_gnodes, end_gnodes = utils.get_graph_with_grouped_nodes(
        graph, start_nodes, end_nodes, strongly_connected_comps
    )

    # Divide all the gnodes in sequential groups (the groups will be modified later if needed)
    unvisited_gnodes = set(ggraph.nodes.keys())
    visited_gnodes = set()

    curr_group = {node for node in start_gnodes if len(ggraph.nodes[node].input_edges) == 0}

    if not curr_group:
        return False, None, None

    groups = [curr_group]
    visited_gnodes = visited_gnodes.union(curr_group)
    unvisited_gnodes.difference_update(curr_group)  # remove curr_group nodes from unvisited_gnodes
    while True:
        curr_group = set()
        for gnode in unvisited_gnodes:
            in_gnodes = {edge.source_node.id for edge in ggraph.nodes[gnode].input_edges}

            if in_gnodes.issubset(visited_gnodes):
                curr_group.add(gnode)

        if not curr_group:
            break
        groups.append(curr_group)
        visited_gnodes = visited_gnodes.union(curr_group)
        unvisited_gnodes.difference_update(curr_group)

    if len(groups) < 2:
        return False, None, None

    # Unite groups if
    # 1. there are edges between non-neighboring groups (long edges) or
    # 2. if a group has an end activity that is not a start activity (it is only possible
    #    for the last group to have such activities)
    if len(groups) > 2:
        end_nonstart_gnodes = end_gnodes.difference(start_gnodes)

        for i in range(len(groups) - 1):
            # Long edges
            target_gnodes = [
                edge.target_node.id for gnode in groups[i] for edge in ggraph.nodes[gnode].output_edges
            ]

            # TODO check inductive_miner = InductiveMiner(data_holder)

            bad_target_gnodes = [
                node for node in target_gnodes if node not in groups[i] and node not in groups[i + 1]
            ]
            if len(bad_target_gnodes) != 0:
                # find the target group of the longest edge and unite all groups in-between
                farthest_group_index = max(
                    {
                        ind
                        for ind, group in enumerate(groups[i + 2 :], i + 2)
                        if len(group.intersection(bad_target_gnodes)) != 0
                    }
                )
                groups[i] = groups[i].union(*[groups[j] for j in range(i + 1, farthest_group_index)])

                for j in range(i + 1, farthest_group_index)[::-1]:
                    del groups[j]
                continue

            # End non-start activity
            if len(groups[i].intersection(end_nonstart_gnodes)) and len(groups) != 2:
                groups[i] = groups[i].union(groups[i + 1])
                del groups[i + 1]
                continue

    # Find groups that should be connected with the hidden activity via the EXCLUSIVE_CHOICE operator
    first_end_node_group_idx = next(
        (i for i in range(len(groups)) if len(end_gnodes.intersection(groups[i])) != 0), None
    )  # for loop assignment with Node default value

    last_start_node_group_idx = next(
        (i for i in range(len(groups))[::-1] if len(start_gnodes.intersection(groups[i])) != 0), None
    )  # for loop assignment with Node default value

    hidden_activity_mask = [
        i < last_start_node_group_idx or i > first_end_node_group_idx for i in range(len(groups))
    ]

    # Transform groups of strongly connected components into groups of real nodes
    node_groups = []
    for group in groups:
        nodes_in_subgraph = set()
        for gnode in group:  # gnode = number of a component
            nodes = strongly_connected_comps[int(gnode)]
            nodes_in_subgraph = nodes_in_subgraph.union(nodes)
        node_groups.append(nodes_in_subgraph)

    return True, node_groups, hidden_activity_mask


def find_parallel_cut(graph, start_nodes, end_nodes):
    """
    Find a parallel cut.

    Parameters
    ----------
    graph: Graph
        Graph.

    start_nodes: set of str
        Start nodes of the graph.

    end_nodes: set of str
        Start nodes of the graph.

    Returns
    -------
    result: bool
        True if found, False otherwise.

    node_groups: list of set of str or None
        If found, returns a list of node groups that the graph should be split into.
    """
    inv_graph = utils.create_inverted_graph(graph)
    connected_components = utils.get_weakly_connected_components(inv_graph)

    if len(connected_components) < 2:
        return False, None
    ok = utils.check_each_node_group_has_start_end_nodes(connected_components, start_nodes, end_nodes)
    if not ok:
        return False, None

    return True, connected_components


def find_loop_cut(graph, start_nodes, end_nodes):
    """
    Find a loop cut.

    Parameters
    ----------
    graph: Graph
        Graph.

    start_nodes: set of str
        Start nodes of the graph.

    end_nodes: set of str
        Start nodes of the graph.

    Returns
    -------
    result: bool
        True if found, False otherwise.

    node_groups: list of set of str or None
        If found, returns a list of node groups that the graph should be split into.
    """
    start_end_nodes = start_nodes | end_nodes
    small_graph = utils.create_graph_without_nodes(graph, start_end_nodes)
    connected_components = utils.get_weakly_connected_components(small_graph)
    main_comp = start_end_nodes

    good_comps = []
    for comp in connected_components:
        in_nodes_from_other_comps = {
            edge.source_node.id for node in comp for edge in graph.nodes[node].input_edges
        }.difference(comp)
        out_nodes_from_other_comps = {
            edge.target_node.id for node in comp for edge in graph.nodes[node].output_edges
        }.difference(comp)

        if in_nodes_from_other_comps.issubset(end_nodes) and out_nodes_from_other_comps.issubset(start_nodes):
            good_comps.append(comp)
        else:
            main_comp = main_comp | comp

    if not good_comps:
        return False, None

    good_comps.insert(0, main_comp)

    return True, good_comps


def find_cut_without_one_node(graph, start_nodes, end_nodes):
    """
    Try to find a node so that the graph without it
    has a cut.

    Parameters
    ----------
    graph: Graph
        Graph.

    start_nodes: set of str
        Start nodes of the graph.

    end_nodes: set of str
        Start nodes of the graph.

    Returns
    -------
    result: bool
        True if found, False otherwise.

    node_to_remove: str

    cut_type: str of ProcessTreeNodeType or None
        Type of the cut if one is found.

    cut_data: object or None
        Information about the cut if one is found.
    """
    for node_to_remove in sorted(graph.nodes.keys()):  # sorted so that different launches give the same result
        new_graph = utils.create_graph_without_nodes(graph, [node_to_remove])
        new_start_nodes = start_nodes.difference([node_to_remove])
        new_end_nodes = end_nodes.difference([node_to_remove])
        found, cut_type, cut_data = find_any_cut(new_graph, new_start_nodes, new_end_nodes)
        if found:
            return found, node_to_remove, cut_type, cut_data

    return False, None, None, None


def find_any_cut(graph, start_nodes, end_nodes):
    """
    Find a cut for a given graph.

    Parameters
    ----------
    graph: Graph
        Graph.

    start_nodes: set of str
        Start nodes of the graph.

    end_nodes: set of str
        Start nodes of the graph.

    Returns
    -------
    result: bool
        True if found, False otherwise.

    cut_type: str of ProcessTreeNodeType or None
        Type of the cut if one is found.

    cut_data: object or None
        Information about the cut if one is found.
    """
    # SINGLE_ACTIVITY
    if len(graph.nodes) == 1:
        if len(graph.edges) != 0:
            return True, ProcessTreeNodeType.SINGLE_ACTIVITY, None

        node_id = list(graph.nodes.keys())[0]
        return True, ProcessTreeNodeType.SINGLE_ACTIVITY, node_id
    # EXCLUSIVE_CHOICE
    found, node_groups = find_exclusive_choice_cut(graph)
    if found:
        return True, ProcessTreeNodeType.EXCLUSIVE_CHOICE, node_groups

    # SEQUENTIAL
    found, node_groups, hidden_activity_mask = find_sequential_cut(graph, start_nodes, end_nodes)
    if found:
        return True, ProcessTreeNodeType.SEQUENTIAL, [node_groups, hidden_activity_mask]

    # PARALLEL
    found, node_groups = find_parallel_cut(graph, start_nodes, end_nodes)
    if found:
        return True, ProcessTreeNodeType.PARALLEL, node_groups

    # LOOP
    found, node_groups = find_loop_cut(graph, start_nodes, end_nodes)
    if found:
        return True, ProcessTreeNodeType.LOOP, node_groups

    return False, None, None
