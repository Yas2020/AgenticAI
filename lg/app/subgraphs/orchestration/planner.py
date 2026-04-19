from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from app.core.state import MasterState
from app.schemas.task import Task

### LLM
planner = ChatOpenAI(model="gpt-4o", temperature=0)

    
class ResearchDAG(BaseModel):
    """The output of the Architect node."""
    tasks: List[Task]
    strategy_rationale: str = Field(description="Why this sequence was chosen")
    estimated_tokens: int
    

# --- The Nodes ---
# 2. 'vector_db': Internal proprietary research and historical reports. REMOVED


PLANNING_SYSTEM_PROMPT = """
Role: Lead Investment Strategist & AI Architect.
Goal: Decompose a high-level investment query into a Directed Acyclic Graph (DAG) of specialized tasks.

Each task must contain a concise but specific description of the work being performed.

Example of task description:
"Search for the last 3 earnings reports and recent news for NVDA"

Context:
You have access to the following specialized agents:
1. 'research': Real-time market data, news, and SEC filings via MCP.
2. 'quant_sandbox': Python-based PAL (Program-Aided Language) for Monte Carlo, DCF, and technical indicators.
3. 'analyst': Synthesizes outputs from research and quant tasks into an investment thesis or report.

Instructions:
1. BREAKDOWN: Divide the query into atomic, dependent tasks. 
2. DEPENDENCIES: A task must list the IDs of tasks that provide its required input. 
3. PARALLELISM: Identify tasks that can run simultaneously (e.g., multiple searching web).
4. QUANT VALIDATION: If the query involves valuation, growth forecasts, or risk assessment, include a 'quant_sandbox' task to compute numerical validation (Monte Carlo, DCF, indicators).

Think step-by-step about the required information flow before generating the DAG.

Output Requirements:
Return a valid DAG where 'depends_on' refers to a list of integer IDs.

CONSTRAINT: Generate between 3 and 8 tasks.
"""

async def planning_architect(state: MasterState):
    """
    Architect: Translates the user query into a Directed Acyclic Graph of tasks.
    """
    # Use a high-reasoning model (Claude 3.5 Sonnet / DeepSeek-R1)
    user_query = state["messages"][-1].content
    
    # Use a high-reasoning model (Claude 3.5 Sonnet or GPT-4o)
    structured_planner = planner.with_structured_output(ResearchDAG)
    
    dag_output = await structured_planner.ainvoke([
        SystemMessage(content=PLANNING_SYSTEM_PROMPT),
        HumanMessage(content=f"Generate a research DAG for: {user_query}")
    ])
    
    # Update state: Store the tasks and the rationale
    return {
        "plan": dag_output.tasks, 
        "current_focus": "Planning Complete",
        "messages": [AIMessage(content=f"Plan generated: {dag_output.strategy_rationale}")]
    }