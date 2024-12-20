from graphviz import Digraph
from IPython.display import HTML

from sberpm.miners.mining_utils import ProcessTreeNode, ProcessTreeNodeType
from sberpm.visual import NodeType, utils


class GvNode:
    """
    Represents a node object with graphviz's parameters.

    Parameters
    ----------
    id_ : str
        Id of the node.
    label : str
        Label of the node.
    shape : {'box', 'circle', 'diamond'}, default=None
        Shape of the node.
    style : {'filled'}, default=None
        Style of the node.
    fillcolor : str (real color: 'red', 'blue',... or hexadecimal representation (GRB model): '#AA5500')
        Fillcolor of the node.

    Attributes
    ----------
    The same as Parameters.
    """

    def __init__(self, id_, label, shape=None, style=None, fillcolor=None):
        self.id = id_
        self.label = label
        self.shape = shape
        self.style = style
        self.fillcolor = fillcolor


def remove_bad_symbols(string):
    """
    Removes bad symbols from the names of the nodes as graphviz can fail while parsing the .gv-file.

    Parameters
    ----------
    string: str
        String (node's name or label).

    Returns
    -------
    result: str
        String without bad symbols.
    """
    bad_symbols = [":", "\\"]
    res = string
    for s in bad_symbols:
        res = res.replace(s, " ")
    return res


def _get_gv_node(node, node_color_dict):
    """
    Chooses visualisation parameters for a node.

    Parameters
    ----------
    node : Node
        Node object.

    Returns
    -------
    gv_node: GvNode
        Object that contains a node's visualization parameters.
    """
    node_id = remove_bad_symbols(node.id)
    node_label = remove_bad_symbols(node.label)
    if node.type == NodeType.TASK:
        node_label_with_metrics = utils.add_metrics_to_node_label(
            node_label, node.metrics
        )
        node_color_by_metric = node_color_dict.get(node.id, None)
        gv_node = GvNode(
            node_id,
            node_label_with_metrics,
            shape="box",
            style="filled",
            fillcolor=node_color_by_metric,
        )
    elif node.type == NodeType.START_EVENT:
        gv_node = GvNode(
            node_id,
            node_label,
            shape="circle",
            style="filled",
            fillcolor="green",
        )
    elif node.type == NodeType.END_EVENT:
        gv_node = GvNode(
            node_id, node_label, shape="circle", style="filled", fillcolor="red"
        )
    elif node.type == NodeType.EXCLUSIVE_GATEWAY:
        gv_node = GvNode(node_id, "X", shape="diamond")
    elif node.type == NodeType.PARALLEL_GATEWAY:
        gv_node = GvNode(node_id, "+", shape="diamond")
        print("huinia")
    elif node.type == NodeType.PLACE:
        gv_node = GvNode(node_id, node_label)
    elif node.type == NodeType.PARALLEL_GATEWAY_BLUE:
        gv_node = GvNode(
            node_id, "||", shape="diamond", style="filled", fillcolor="blue"
        )
        print(gv_node)
    else:
        raise RuntimeError(f'Wrong node type: "{node.type}"')
    return gv_node


class GraphvizPainter:
    """
    Class represents a graph visualizer that uses a graphviz visualizing tool.

    Attributes
    ----------
    _digraph: Digraph
        A graphviz's object that represents a graph.

    Examples
    --------
    >>> import pandas as pd
    >>> from sberpm import DataHolder
    >>> from sberpm.miners import HeuMiner
    >>> from sberpm.visual import GraphvizPainter
    >>>
    >>> # Create data_holder
    >>> df = pd.DataFrame({
    ...     'id_column': [1, 1, 2],
    ...     'activity_column':['st1', 'st2', 'st1'],
    ...     'dt_column':[123456, 123457, 123458]})
    >>> data_holder = DataHolder(df, 'id_column', 'activity_column', 'dt_column')
    >>>
    >>> # Create graph using a miner algorithm
    >>> miner = HeuMiner(data_holder)
    >>> miner.apply()
    >>>
    >>> # Visualize graph
    >>> painter = GraphvizPainter()
    >>> painter.apply(miner.graph)
    >>>
    >>> # Or visualize graph with metrics
    >>> graph = miner.graph
    >>> graph.add_node_metric('count', {'st1': 2, 'st2': 1})
    >>> painter = GraphvizPainter()
    >>> painter.apply(graph, node_style_metric='count')
    >>>
    >>> # Save and show result
    >>> painter.write_graph('graph.svg', 'svg')
    >>> painter.show()  # Works in Jupyter Notebook
    """

    def __init__(self):
        self._digraph = None

    def apply(
        self,
        graph,
        node_style_metric=None,
        edge_style_metric=None,
        hide_disconnected_nodes=False,
    ):
        """
        Visualizes the given graph.
        (Creates graphviz's object that can be displayed or saved to file.)

        Parameters
        ----------
        graph : Graph or ProcessTreeNode
            Graph object.

        node_style_metric: str
            Name of the node's metric that will influence the colour of the nodes.
            If None or given metric in not contained in the given graph, nodes will have the same colour.
            Is not used if graph is a ProcessTreeNode object.

        edge_style_metric: str
            Name of the edge's metric that will influence the thickness of the edges.
            If None or given metric in not contained in the given graph, edges will have the same width.
            Is not used if graph is a ProcessTreeNode object.

        hide_disconnected_nodes: bool, default=False
            If True, nodes without any input and output edges will not be displayed.
            Is not used if graph is a ProcessTreeNode object.
        """
        if isinstance(graph, ProcessTreeNode):
            self._apply_process_tree(graph)
            return

        self._digraph = Digraph()

        node_color_dict = utils.calc_nodes_colors_by_metric(graph, node_style_metric)
        edge_width_dict = utils.calc_edges_widths_by_metric(graph, edge_style_metric)

        for node in graph.get_nodes():
            if not (
                hide_disconnected_nodes
                and len(node.output_edges) == 0
                and len(node.input_edges) == 0
            ):
                gv_node = _get_gv_node(node, node_color_dict)
                self._add_node_to_digraph(gv_node)

        for n1n2, edge in graph.edges.items():
            self._digraph.edge(
                remove_bad_symbols(edge.source_node.id),
                remove_bad_symbols(edge.target_node.id),
                penwidth=str(edge_width_dict[n1n2])
                if n1n2 in edge_width_dict
                else None,
                label=str(edge.metrics[edge_style_metric])
                if edge_style_metric in edge.metrics
                else None,
            )

    def _apply_process_tree(self, root_node):
        """
        Graphviz visualizer for ProcessTreeNode class.

        Parameters
        ----------
        root_node: ProcessTreeNode
        """
        digraph = Digraph()

        metrics = root_node.nodes_metric_data

        # Add nodes
        label_dict = {
            ProcessTreeNodeType.EXCLUSIVE_CHOICE: "X",
            ProcessTreeNodeType.SEQUENTIAL: "->",
            ProcessTreeNodeType.PARALLEL: "||",
            ProcessTreeNodeType.LOOP: "*",
            ProcessTreeNodeType.FLOWER: "?",
        }

        node2gvnode = dict()
        GraphvizPainter._add_process_tree_nodes(
            digraph,
            root_node,
            label_dict,
            node2gvnode,
            metrics,
        )

        # Add edges
        GraphvizPainter._add_process_tree_edges(digraph, root_node, node2gvnode)

        self._digraph = digraph

    @staticmethod
    def _add_process_tree_nodes(
        digraph: Digraph, node, label_dict, node2gvnode, metrics
    ):
        def get_node_metrics(nodes_metric_data: dict, node_name: str):
            """
            Remove metric_name from the nodes_metric_names

            Parameters
            ----------
            node_name : str
                Name of the metric.
            """
            node_metrics = dict()

            for metric_name, metrics in nodes_metric_data.items():
                for node, metric_value in metrics.items():
                    if node == node_name:
                        node_metrics[metric_name] = metric_value
            return node_metrics

        if node.type == ProcessTreeNodeType.SINGLE_ACTIVITY:
            if node.label is not None:
                node_id = node.label
                label = utils.add_metrics_to_node_label(
                    label=node.label,
                    metrics=get_node_metrics(metrics, node.label),
                )
                color = "white"
            else:
                node_id = f"{node.type}_{len(node2gvnode)}"
                label = ""
                color = "black"
            shape = "box"
        else:
            node_id = f"{node.type}_{len(node2gvnode)}"
            label = label_dict[node.type]
            shape = "circle"
            color = "white"

        node2gvnode[node] = node_id
        digraph.node(node_id, label, shape=shape, fillcolor=color, style="filled")

        for n in node.children:
            GraphvizPainter._add_process_tree_nodes(
                digraph, n, label_dict, node2gvnode, metrics=metrics
            )

    @staticmethod
    def _add_process_tree_edges(digraph: Digraph, node, node2gvnode):
        n1 = node2gvnode[node]
        for node2 in node.children:
            n2 = node2gvnode[node2]
            digraph.edge(n1, n2)
        for node2 in node.children:
            GraphvizPainter._add_process_tree_edges(digraph, node2, node2gvnode)

    def _add_node_to_digraph(self, gv_node):
        """
        Adds a GvNode to the graph.

        Parameters
        ----------
        gv_node: GvNode
            Represents a node object with graphviz's parameters.
        """
        self._digraph.node(
            gv_node.id,
            shape=gv_node.shape,
            fillcolor=gv_node.fillcolor,
            style=gv_node.style,
            label=gv_node.label,
        )

    def save(self, filename, format, prog="dot"):
        """
        Saves a graph visualization to file.

        Parameters
        ----------
        filename : str
            Name of the file to save the result to.

        format : {'gv', 'svg', 'png', 'pdf'}
            Format of the file.

        prog : {'dot', 'neato', ...}, default='dot'
            Graphviz's engine used to render the visualization.
        """
        self._digraph.__setattr__("engine", prog)
        binary_data = self._digraph.pipe(format=format)
        with open(filename, mode="wb") as f:
            f.write(binary_data)

    def write_graph(self, filename, format, prog="dot"):
        """
        Saves a graph visualization to file.

        Parameters
        ----------
        filename : str
            Name of the file to save the result to.

        format : {'gv', 'svg', 'png', 'pdf'}
            Format of the file.

        prog : {'dot', 'neato', ...}, default='dot'
            Graphviz's engine used to render the visualization.
        """
        self.save(filename, format, prog)

    def show(self, fit_window: bool = True):
        """
        Shows visualization of the graph in Jupyter Notebook.

        Parameters
        ----------
        fit_window: bool, default=True
            If True:
                if the picture width is bigger than the width of the window,
                picture size will be lowered so that its width
                fits the maximum width of the window
            if False:
                picture will be shown without changes.

        Returns
        -------
        digraph : IPython.core.display.HTML
            Graph in HTML format.
        """
        if fit_window:
            return HTML(self._digraph.pipe(format="svg").decode())
        else:
            return self._digraph
