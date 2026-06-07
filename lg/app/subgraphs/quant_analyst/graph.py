from langgraph.graph import END, START, StateGraph

from .quant import QuantInput, auditor_node, quant_node, route_audit_subgraph, route_quant


def quant_subgraph():
    builder = StateGraph(QuantInput)
    builder.add_node("quant_node", quant_node)
    builder.add_node("auditor_node", auditor_node)

    builder.add_edge(START, "quant_node")
    builder.add_conditional_edges("quant_node", route_quant, ["auditor_node", "quant_node"])
    builder.add_conditional_edges(
        "auditor_node", route_audit_subgraph, ["quant_node", END]
    )

    return builder.compile()
