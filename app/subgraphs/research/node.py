import asyncio
from datetime import datetime
from app.schemas.task import TaskUpdate
from app.core.state import MasterState
from app.schemas.artifact import Artifact
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

class AgentInput(MasterState):
    task_id: int


### LLM
extractor = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

RESEARCHER_PROMPT = """
Role: Senior Equity Research Analyst.
Context: You are researching {topic}.
Task: Execute the assigned task and extract high-signal financial data.
Guardrail: Ignore SEO spam and marketing fluff. Focus on SEC filings, earnings transcripts, and reputable news.
"""

async def research_agent(state: AgentInput):
    # 1. Find the task assigned to this agent with specific task id
    # Your scheduler should have already set one to 'running'
    task = next(t for t in state["plan"] if t.id == state["task_id"])
    topic = state.get("topic", "General Investment")
    
    try:
        # 2. Call the MCP Tool (Web Search)
        # This assumes you have an mcp_service helper
        search_results = await mcp_service.call_tool(
            "web_search", 
            {"query": f"{topic}: {task.description}"}
        )
        
        # 3. Use an LLM to "Sift" the raw results into an Artifact
        # We use a structured output to ensure the Quantitative agent can read it later
        extraction = await extractor.with_structured_output(Artifact).ainvoke([
            SystemMessage(content=RESEARCHER_PROMPT.format(topic=topic)),
            HumanMessage(content=f"Raw Data: {search_results}")
        ])
        
        # 4. Return the standard 'Senior' response
        return {
            "artifacts": [
                Artifact(
                    artifact_type="research",
                    source=task.agent, 
                    content=extraction, 
                    task_id=state["task_id"],
                    timestamp=datetime.now().isoformat()
                )
            ],
            "plan": [TaskUpdate(task_id=state["task_id"], status="completed")]
        }
    except Exception as e:
        return {"plan": [
            TaskUpdate(
                task_id=state["task_id"],
                status="failed",
                error_message=str(e)
            )
        ]}


############# Mock Agent ######### 

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
        timestamp=datetime.now().isoformat()
    )
    
    # 4. Return the 'Receipt' for the Reducer
    return {
        "artifacts": [mock_artifact],
        "plan": [TaskUpdate(id=task.id, status="completed")]
    }
    