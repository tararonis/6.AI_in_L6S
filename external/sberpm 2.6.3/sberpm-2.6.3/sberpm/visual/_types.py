from enum import Enum


class GraphType(str, Enum):
    """
    Types of graph
    """

    PETRI_NET = "Petri-Net"
    DFG = "DFG"
    BPMN = "BPMN"

    def __str__(self) -> str:
        return self.value


class NodeType(str, Enum):
    """
    Types of nodes
    """

    START_EVENT = "startevent"
    END_EVENT = "endevent"
    TASK = "task"
    PLACE = "place"
    PARALLEL_GATEWAY = "parallel_gateway"
    EXCLUSIVE_GATEWAY = "exclusive_gateway"
    PARALLEL_GATEWAY_BLUE = "parallel_gateway_blue"

    def __str__(self) -> str:
        return self.value
