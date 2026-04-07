from .quant import QuantInput, quant_node, auditor_node, route_quant
from langgraph.graph import START, StateGraph

# In the Subgraph definition
def quant_subgraph():
    builder = StateGraph(QuantInput)
    builder.add_node("quant_node", quant_node)
    builder.add_node("auditor_node", auditor_node)
    
    builder.add_edge(START, "quant_node")
    builder.add_conditional_edges("quant_node", route_quant, ["auditor_node", "quant_node"])
    
    return builder.compile()
