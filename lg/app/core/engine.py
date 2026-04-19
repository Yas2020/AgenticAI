from langgraph.graph import END, START, StateGraph
from app.services.langgraph_postgres.checkpointer import checkpointer

from app.core.state import MasterState
from app.subgraphs.orchestration.planner import planning_architect
from app.subgraphs.orchestration.scheduler import scheduler, route_to_agents
from app.subgraphs.orchestration.plan_validator import plan_validator, route_valid_plan
from app.subgraphs.orchestration.query_validator import query_validator, route_valid_query

# from app.subgraphs.research.node import researcher
# from app.subgraphs.vector_db.vector_db import vector_db
# from app.subgraphs.analyst.analyst import analyst
# from app.subgraphs.quant_analyst.quant import quant

from app.subgraphs.research.node import research_agent
from app.subgraphs.quant_analyst.graph import quant_subgraph
from app.subgraphs.quant_analyst.quant import route_audit
from app.subgraphs.analyst.analyst import analyst_agent


# # Add nodes and edges 
builder = StateGraph(MasterState)
builder.add_node("query_validator", query_validator)
builder.add_node("planning_architect", planning_architect)
builder.add_node("plan_validator", plan_validator)
builder.add_node("scheduler", scheduler)
builder.add_node("research", research_agent)
builder.add_node("quant_analyst", quant_subgraph())
# builder.add_node("quant_analyst", quant)
# builder.add_node("vector_db", vector_db)
builder.add_node("analyst", analyst_agent)

# Logic
builder.add_edge(START, "query_validator")
builder.add_conditional_edges("query_validator", route_valid_query, ["planning_architect", END])
builder.add_edge("planning_architect", "plan_validator")
builder.add_conditional_edges("plan_validator", route_valid_plan, ["planning_architect", "scheduler"])
builder.add_conditional_edges("scheduler", route_to_agents, ["research", "quant_analyst", "analyst", END])
builder.add_edge("research", "scheduler")
# builder.add_edge("vector_db", "scheduler")
builder.add_edge("analyst", "scheduler")
# builder.add_edge("quant_analyst", "scheduler")
builder.add_conditional_edges("quant_analyst", route_audit, ["scheduler", "quant_analyst"])


# Compile
graph = builder.compile(
    checkpointer=checkpointer,
)