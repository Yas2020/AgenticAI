import os
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
from app.schemas.task import TaskUpdate
from app.core.state import MasterState
from app.schemas.artifact import Artifact
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI


class AnalystInput(MasterState):
    task_id: int
    
# 5. Call the "Big Gun" model for synthesis
model = ChatOpenAI(model="gpt-4o", temperature=0)

def format_research(artifacts, source):
    formatted = []
    for a in artifacts:
        if a.source == source and isinstance(a.content, dict):
            # Transform dict into a readable block
            summary = "\n".join([f"- {k}: {v}" for k, v in a.content.items()])
            formatted.append(f"RESEARCH DATA:\n{summary}")
        else:
            formatted.append(str(a.content))
    return "\n\n".join(formatted)


async def analyst_agent(state: MasterState):
    
    # 2. Pull the quant task from the global state
    task = next(t for t in state["plan"] if t.id == state["task_id"])
    if task is None:
        return {"plan": [TaskUpdate(task_id=state["task_id"], status="failed", error_message="Task not found")]}
    
    # 3. Gather all context
    research_docs = format_research(state["artifacts"], "research")
    quant_results = format_research(state["artifacts"], "quant_analyst")
    
    # 4. Construct the high-stakes prompt
    prompt = f"""
    You are a Senior Investment Analyst. You must synthesize the following data into a final report.
    
    ### RESEARCH FINDINGS:
    {research_docs}
    
    ### QUANTITATIVE VALUATION (JSON):
    {quant_results}
    
    ### YOUR MISSION:
    1. CROSS-CHECK: Does the revenue/growth in the Quant math match the Research facts? 
    2. VALUATION: Is the DCF value realistic compared to the market context?
    3. FINAL VERDICT: Provide a 'BUY', 'HOLD', or 'SELL' recommendation.
    
    ### FORMAT:
    Output your report in clean Markdown. End with a section titled 'EVALUATION' 
    where you grade the accuracy of the preceding agents.
    """
    
    response = await model.ainvoke([SystemMessage(content=prompt)])
    
    # 6. Save the markdown report
    run_id = str(uuid.uuid4())[:8]
    base_artifacts_path = Path("/app/artifacts/final_report") 
    run_dir = base_artifacts_path / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
    os.chmod(run_dir, 0o777)
    
    report_path = run_dir / "final_report.md"
    
    metadata_header = f"""---
    
    Run ID: {run_id}
    Model: {model.model_name}
    Timestamp: {datetime.now().isoformat()}
---

"""
    
    report_path.write_text(metadata_header + response.content)
    
    # 7. Save the Final Report as a unique artifact
    return {
        "artifacts": [
            Artifact(
                artifact_type="final_report", 
                task_id=task.id,
                source=task.agent,
                content=response.content,
                timestamp=datetime.now().isoformat(),
                success=True,
                error=None
            )
        ],
        "plan": [TaskUpdate(id=task.id, status="completed", error_message=None)]
    }




##################################
######### Mock Agent #############
##################################

async def analyst(state: AnalystInput):
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
        artifact_type="analyst_report", 
        task_id=task.id,
        source=task.agent,
        content=f"Mock data for {task.description}",
        timestamp=datetime.now().isoformat(),
        success=True,
        error=None
    )
    
    # 4. Return the 'Receipt' for the Reducer
    return {
        "artifacts": [mock_artifact],
        "plan": [TaskUpdate(id=task.id, status="completed", error_message=None)]
    }