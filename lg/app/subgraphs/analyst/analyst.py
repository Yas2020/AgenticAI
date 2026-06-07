import asyncio
from datetime import datetime, timezone

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.schemas.task import TaskUpdate
from app.core.state import MasterState
from app.schemas.artifact import Artifact
from app.services.artifacts.evidence import (
    build_source_citations,
    format_evidence_bundle,
)
from app.services.artifacts.paths import write_final_report


class AnalystInput(MasterState):
    task_id: int
    
# 5. Call the "Big Gun" model for synthesis
model = ChatOpenAI(model="gpt-4o", temperature=0)


async def analyst_agent(state: MasterState):
    
    # 2. Pull the quant task from the global state
    task = next(t for t in state["plan"] if t.id == state["task_id"])
    if task is None:
        return {"plan": [TaskUpdate(id=state["task_id"], status="failed", error_message="Task not found")]}
    
    # 3. Gather all context
    research_docs = format_evidence_bundle(state["artifacts"], sources=("research",))
    quant_results = format_evidence_bundle(state["artifacts"], sources=("quant_analyst",))
    citation_index, _sources = build_source_citations(state["artifacts"])

    failed_upstream = [
        t
        for t in state["plan"]
        if t.status == "failed" and t.id != task.id
    ]
    failure_notes = ""
    if failed_upstream:
        lines = [
            f"- Task {t.id} ({t.agent}): {t.error_message or 'failed'}"
            for t in failed_upstream
        ]
        failure_notes = (
            "### UPSTREAM TASK FAILURES (do not invent data to replace these):\n"
            + "\n".join(lines)
            + "\n"
        )

    prompt = f"""
    You are a Senior Investment Analyst. Synthesize the evidence into a final investment report.

    ### SOURCE INDEX (cite every factual claim as [^1], [^2], etc. matching the numbers below):
    {citation_index}

    ### RESEARCH FINDINGS:
    {research_docs}

    ### QUANTITATIVE VALUATION (JSON):
    {quant_results or "(no quantitative output — explain limitation in prose)"}

    {failure_notes}
    ### YOUR MISSION:
    1. CROSS-CHECK: Does quant math align with research where both exist?
    2. If upstream tasks failed, state uncertainty explicitly — do not fabricate missing figures.
    3. VALUATION: Assess realism vs context when data exists.
    4. FINAL VERDICT: BUY / HOLD / SELL when sufficient evidence; otherwise "INSUFFICIENT DATA".

    ### FORMAT:
    - Clean Markdown with inline citations [^1], [^2], etc. matching SOURCE INDEX.
    - End with a "## References" section listing each [^n] URL or source label.
    - End with a section titled 'EVALUATION' grading preceding agents.
    """
    
    response = await model.ainvoke([SystemMessage(content=prompt)])

    query = next(
        (
            getattr(msg, "content", "")
            for msg in state.get("messages", [])
            if getattr(msg, "content", None)
        ),
        "",
    )
    report_path = write_final_report(
        response.content,
        topic=state.get("topic", "equity research"),
        query=query if isinstance(query, str) else str(query),
        model_name=model.model_name,
        artifacts=state.get("artifacts", []),
    )
    print(f"Final report written to {report_path}")

    return {
        "artifacts": [
            Artifact(
                artifact_type="final_report",
                task_id=task.id,
                source=task.agent,
                content=response.content,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=True,
                error=None,
            )
        ],
        "plan": [TaskUpdate(id=task.id, status="completed", error_message=None)],
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