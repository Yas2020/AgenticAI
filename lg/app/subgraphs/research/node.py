import asyncio
import traceback
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.schemas.task import TaskUpdate
from app.core.state import MasterState
from app.schemas.artifact import Artifact
from app.services.mcp.mcp_clients import mcp_manager # Import the shared instance


class AgentInput(MasterState):
    task_id: int
    
class FinancialMetrics(BaseModel):
    revenue: float | None = None
    net_income: float | None = None
    pe_ratio: float | None = None
    
class ResearchSummary(BaseModel):
    key_findings: List[str]
    financial_metrics: FinancialMetrics
    sources: List[str]


RESEARCHER_PROMPT = """
Role: Senior Equity Research Analyst.
Context: You are researching {topic}.
Task: Execute the assigned task and extract high-signal financial data.
Guardrail: Ignore SEO spam and marketing fluff. Focus on SEC filings, earnings transcripts, and reputable news.
"""


# 1. Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Get MCP server for research
mcp_research = mcp_manager["research"]

async def research_agent(state: AgentInput):
    # Find the task assigned to this agent with specific task id
    task = next((t for t in state["plan"] if t.id == state["task_id"]), None)
    if task is None:
        return {"plan": [TaskUpdate(task_id=state["task_id"], status="failed", error_message="Task not found")]}
    topic = state.get("topic", "General Investment")
    
    try: 
        # 1. Get the ALREADY OPEN session (or connect if it's the first time)
        print("DEBUG: Loading tools from session...")
        # Use cached tools if possible
        research_tools = await mcp_research.get_tools()
        print("Tools Received. Here are the tools:")
        print(research_tools)
        
        # 1. Setup history
        messages = [
            SystemMessage(content=RESEARCHER_PROMPT.format(topic=topic)),
            HumanMessage(content=f'''Task: {task.description}\nUse the available web search tool to gather information first.
            Then summarize the findings into the required schema.''')
        ]

        # 2. Bind tools to the base model
        tool_model = model.bind_tools(research_tools)
        
        # 3. Let the model to choose tools and prepare args for them - dont add structure output yet so model is focused with this tool
        response = await tool_model.ainvoke(messages)
        # If the model decides to use the tool, response is an AIMessage empty content with tool call extra kwargs
        messages.append(response) # Keep the assistant's request in history
        
        # 4. If the model generated tool_calls, loop through them. Use the tool objects directly to talk to the MCP server.
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Find the matching tool from your research_tools list
                selected_tool = next(t for t in research_tools if t.name == tool_call["name"])
                
                # This triggers the actual MCP protocol call to the server
                tool_output = await selected_tool.ainvoke(tool_call["args"])
                
                # Add the result to history so the model sees what it found
                messages.append(ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=str(tool_output)
                ))
        
        extractor = model.with_structured_output(ResearchSummary)
   
        # 5. Use an LLM to "Sift" the raw results into an Artifact
        # We use a structured output to ensure the Quantitative agent can read it later
        extraction = await extractor.ainvoke(messages)
        content_str = extraction.model_dump_json() 

        # Append it as an AI Message so the next node knows the "Assistant" said this
        messages.append(AIMessage(content=content_str))

        # 4. Return the standard response
        return {
            "artifacts": [
                Artifact(
                    artifact_type="web_research",
                    source=task.agent, 
                    content=extraction.model_dump(), # Nested dict is fine
                    task_id=state["task_id"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    success=True,
                    error=None
                )
            ],
            "plan": [
                TaskUpdate(
                    id=state["task_id"], 
                    status="completed", 
                    error_message=None
                )
            ],
            "messages": messages
        }
    except Exception as e:
        traceback.print_exc() # This will show the EXACT line and error in your terminal
        return {"plan": [
            TaskUpdate(
                id=state["task_id"],
                status="failed",
                error_message=str(e)
            )
        ]}
    




##################################
############# Mock Agent #########
################################## 

async def researcher(state: AgentInput):
    """
    A generic mock node to test parallel execution and state merging.
    """
    # 1. Find the task assigned to this agent with specific task id
    # Your scheduler should have already set one to 'running'
    task = next(t for t in state["plan"] if t.id == state["task_id"])
    
    print(f"--- [MOCK] Executing specific task {task.id}: {task.description} ---")
    
    # 2. Simulate 'Work' (Network latency)
    await asyncio.sleep(1) 
    
    # 3. Create a Dummy Artifact
    mock_artifact = Artifact(
        artifact_type="researcher", 
        task_id=task.id,
        source=task.agent,
        content=f"Mock data for {task.description}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        success=True,
        error=None
    )
    
    # 4. Return the 'Receipt' for the Reducer
    return {
        "artifacts": [mock_artifact],
        "plan": [TaskUpdate(id=task.id, status="completed", error_message=None)]
    }
    