from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from app.services.langgraph.checkpointer import checkpointer

from app.core.state import MasterState
from app.subgraphs.planning.logic import planning_architect
from app.subgraphs.planning.orchestrator import scheduler, route_to_agents
from app.subgraphs.planning.plan_validator import plan_validator, route_valid_plan
from app.subgraphs.planning.query_validator import query_validator, route_valid_query

from app.subgraphs.research.node import researcher
from app.subgraphs.vector_db.vector_db import vector_db
from app.subgraphs.analyst.analyst import analyst
from app.subgraphs.quantitative.quant import quant

from app.subgraphs.research.node import research_agent


# AGENT_REGISTRY = {
#     "research": researcher,
#     "quant_sandbox": quant,
#     "vector_db": vector_db,
#     "analyst": analyst
# }


# # Add nodes and edges 
builder = StateGraph(MasterState)
builder.add_node("query_validator", query_validator)
builder.add_node("planning_architect", planning_architect)
builder.add_node("plan_validator", plan_validator)
builder.add_node("scheduler", scheduler)
builder.add_node("research", research_agent)
builder.add_node("quant_sandbox", quant)
builder.add_node("vector_db", vector_db)
builder.add_node("analyst", analyst)

# Logic
builder.add_edge(START, "query_validator")
builder.add_conditional_edges("query_validator", route_valid_query, ["planning_architect", END])
builder.add_edge("planning_architect", "plan_validator")
builder.add_conditional_edges("plan_validator", route_valid_plan, ["planning_architect", "scheduler"])
builder.add_conditional_edges("scheduler", route_to_agents, ["research", "quant_sandbox", "vector_db", "analyst", END])
builder.add_edge("research", "scheduler")
builder.add_edge("quant_sandbox", "scheduler")
builder.add_edge("vector_db", "scheduler")
builder.add_edge("analyst", "scheduler")
# builder.add_edge(END)

# Compile
graph = builder.compile(
    checkpointer=checkpointer,
    # interrupt_before=["query_validator"]
)

# We tell the graph: "If the query is invalid, STOP and wait for the human."
