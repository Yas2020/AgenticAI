import asyncio
import traceback
import json
from datetime import datetime, timezone
from typing import Literal, Optional

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Send

from app.schemas.task import Task, TaskUpdate
from app.core.state import MasterState
from app.schemas.artifact import Artifact
from app.services.artifacts.evidence import format_evidence_bundle
from app.services.mcp.mcp_clients import mcp_manager

MAX_ITERATION = 2


class QuantInput(MasterState):
    """Subgraph-local state for quant execution and audit reflection loops."""

    task_id: int
    retry_count: int = 0
    audit_status: Optional[Literal["passed", "failed"]] = None
    audit_feedback: Optional[str] = None

# 1. Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Get MCP server for research
mcp_quant = mcp_manager["quant"]

# 3. System Prompt (The PAL Specialist)
QUANT_ANALYST_PROMPT = """You are a Senior Quantitative Analyst. 
Your goal is to write deterministic Python code to solve financial tasks.
- Use 'numpy' and 'pandas' for calculations.
- Use 'matplotlib' for charts.
- You have access to a pre-defined string variable called ARTIFACT_DIR. DO NOT import os or sys. To save your plots, use:
plt.savefig(ARTIFACT_DIR + '/plot.png')
- Use data found by the Researcher provided in the Context below.

Context from Researcher: {research_context}

Task: {task_description}

### MANDATORY RULES:
1. NO HALLUCINATIONS: You are strictly forbidden from using "placeholder," "mock," or "example" data. 
2. CONTEXT-ONLY: If a specific data point (like Data Center Revenue) is present in the RESEARCH_CONTEXT, you MUST use those exact figures.
3. FAIL-FAST: If the data is missing from the context, do not invent it. State "DATA_MISSING" in your code comments.
4. Call the `execute_quant_code` tool with your Python script in the `code` argument. Do not paste code in the chat — only via the tool.
5. UNIT STANDARD: You MUST use the units found in the Research Data (usually Billions). Do not convert to Millions unless the research specifically uses Millions.
6. MATH > TEXT: If the research provides both a growth rate (e.g., 75%) and the raw revenue numbers, and they do not mathematically align, use the raw revenue numbers to calculate your own growth rate.
7. EXPLAIN DISCREPANCIES: If you find a data conflict, add a discrepancy_note key to your summary dictionary explaining your choice.
8. Your script MUST terminate with these exact lines of code. No exceptions.
Do not print any other text, explanations, or summaries outside of this JSON block.

Your script must end with:
import json
# Ensure all numpy types or decimals are converted to standard floats/ints first
print(json.dumps(summary, indent=2))
"""


def _last_quant_artifact(state: QuantInput) -> Artifact | None:
    return next(
        (a for a in reversed(state["artifacts"]) if a.artifact_type == "quantitative_analyst"),
        None,
    )


def _build_quant_messages(
    state: QuantInput, task: Task, research_context: str
) -> list[BaseMessage]:
    system = SystemMessage(
        content=QUANT_ANALYST_PROMPT.format(
            research_context=research_context,
            task_description=task.description,
        )
    )
    initial = HumanMessage(
        content=(
            f"Task: {task.description}\n"
            "Write the Python analysis script and call execute_quant_code with it. "
            "Do not reply with code in plain text."
        )
    )

    audit_feedback = state.get("audit_feedback")
    if audit_feedback:
        last = _last_quant_artifact(state)
        code = ""
        if last and isinstance(last.content, dict):
            code = last.content.get("code") or ""
        return [
            system,
            initial,
            HumanMessage(
                content=(
                    "The Auditor rejected your previous attempt.\n\n"
                    f"Auditor feedback:\n{audit_feedback}\n\n"
                    f"Previous code:\n```python\n{code}\n```\n\n"
                    "Fix the issues and call execute_quant_code with a corrected script."
                )
            ),
        ]

    if state.get("retry_count", 0) > 0:
        last = _last_quant_artifact(state)
        err = "Execution failed with no error details."
        if last and isinstance(last.content, dict):
            err = (
                last.content.get("stderr")
                or last.content.get("error")
                or err
            )
        return [
            system,
            initial,
            HumanMessage(
                content=(
                    "Your previous script failed to execute successfully.\n\n"
                    f"Error output:\n{err}\n\n"
                    "Fix the script and call execute_quant_code again."
                )
            ),
        ]

    return [system, initial]


async def quant_node(state: QuantInput):
    """
    The node responsible for triggering the PAL (Program-Aided Language) flow.
    """
    
    # 1. Pull the quant task from the global state
    task = next(t for t in state["plan"] if t.id == state["task_id"])
    if task is None:
        return {"plan": [TaskUpdate(id=state["task_id"], status="failed", error_message="Task not found")]}
        
    # 2. Context Injection: Summarize what the researcher found
    # This prevents the Quant agent from having to 'search' the whole history
    research_context = format_evidence_bundle(state["artifacts"], sources=("research",))
    
    messages = _build_quant_messages(state, task, research_context)
    
    try: 
        # 4. Get the tools from the ALREADY OPEN session
        print("DEBUG: Loading tools from session")
        # Use cached tools if possible
        quant_tools = await mcp_quant.get_tools()
        print(f"Tools Received. Here are the tools:\n{quant_tools}")
        
        # 5. Bind tools to the base model
        tool_model = model.bind_tools(quant_tools)
        
        # 6. Let the model to choose tools, prepare args and number of times they call tools etc... But you may also force it to use the tool "execute_quant_code"
        response = await tool_model.ainvoke(
            messages, 
            # tool_choice={"type": "function", "function": {"name": "execute_quant_code"}}
        )
        # If the model decides to use the tool, response is an AIMessage empty content with tool call extra kwargs
        messages.append(response) # Keep the assistant's request in history
        
        # 7. Pull out the generated code for later inspection by our auditor
        generated_code = None
        # Find the specific tool call for code execution
        quant_call = next(
            (tc for tc in response.tool_calls if tc["name"] == "execute_quant_code"), 
            None
        )
        if quant_call:
            # tool_calls[i]["args"] is a dictionary of the arguments passed to the tool
            generated_code = quant_call["args"].get("code")
        
        # 8. If the model generated tool_calls, loop through them. Use the tool objects directly to talk to the MCP server.
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Find the matching tool from your research_tools list
                selected_tool = next(t for t in quant_tools if t.name == tool_call["name"])
                
                # This triggers the actual MCP protocol call to the server
                mcp_response = await selected_tool.ainvoke(tool_call["args"])
                
                # 9. Create the ToolMessage and add it to the message history
                # This ties the result back to the specific ID the LLM generated
                messages.append(
                    ToolMessage(content=str(mcp_response), 
                    tool_call_id=tool_call["id"]
                    )
                )
            # 10. FINAL STEP: Call the LLM one last time 
            # Let the model see the results (Optional, but good for context) - no need to add the return to messages
            # await tool_model.ainvoke(messages)
        
            # 11. Access the first element of the list (mcp returns a list TextContent()) and get the 'text' content
            if mcp_response and len(mcp_response) > 0:
                raw_tool_output = mcp_response[0].get("text", "") 
            else:
                raw_tool_output = "{}" # Fallback
            
            # 12. Parse that text (which is your JSON string from the server) into a dict
            try:
                mcp_data = json.loads(raw_tool_output)
            except json.JSONDecodeError:
                # Handle cases where the server might have crashed or returned non-JSON
                mcp_data = {"status": "error", "stdout": raw_tool_output, "stderr": "Failed to parse JSON"}
            
            # 13. Return the standard response
            success = mcp_data.get("status") == "success"
            update: dict = {
                "artifacts": [
                    Artifact(
                        artifact_type="quantitative_analyst",
                        source=task.agent, 
                        content={
                            "code": generated_code,
                            "stdout": mcp_data.get("stdout"),
                            "stderr": mcp_data.get("stderr"),
                            "results": mcp_data.get("result"),
                            "plots": mcp_data.get("artifacts"),
                            "run_id": mcp_data.get("run_id"),
                        },
                        task_id=state["task_id"],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        success=success,
                        error=None
                    )
                ],
                "messages": messages
            }
            if not success:
                update["retry_count"] = state.get("retry_count", 0) + 1
            return update
            
        # If the model doesnt make a tool call in its response, its not doing what its supposed to do! 
        return {
            "artifacts": [
                Artifact(
                    artifact_type="quantitative_analyst",
                    source=task.agent,
                    content={"error": "Agent failed to trigger the quant code execution tool."},
                    task_id=state["task_id"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    success=False,
                    error=None
                )
            ],
            "retry_count": state.get("retry_count", 0) + 1,
            "messages": messages,
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
 
def route_quant(state: QuantInput):
    last_artifact = state["artifacts"][-1]
    if not last_artifact.success:
        retry_count = state.get("retry_count", 0)
        if retry_count < MAX_ITERATION:
            return Send("quant_node", state)
        return "auditor_node"

    return "auditor_node"
   

AUDITOR_SYSTEM_PROMPT = """
You are a Lead Financial Auditor. Your sole mission is to ensure the Quantitative Agent's simulation is FACTUALLY ACCURATE and LOGICALLY SOUND.

INPUT DATA FOR VERIFICATION:
1. RESEARCH DATA: The ground truth extracted from filings/news.
2. QUANT CODE: The Python script the agent wrote to process this data. Note: The string variable ARTIFACT_DIR is a pre-defined. The code must not define it! 
3. STDOUT: The actual output of that script.

YOUR CRITICAL ERROR LIST:
- DATA HALLUCINATION: Did the agent use 'Mock' or 'Example' numbers instead of the specific figures in the Research Data? (Check billions vs millions!)
- SILENT FAILURE: Did the script run but fail to print the results in a structured format (JSON)?
- LOGIC ERROR: Is the formula for DCF, WACC, or CAGR mathematically incorrect?
- UNIT MISMATCH: Is the code mixing 'Thousands' and 'Millions' incorrectly?

RESPONSE PROTOCOL:
- If the code is perfect: Respond with 'PASS' and a brief summary of the findings.
- If there is an error: Respond with 'FAIL' followed by a specific 'CORRECTION_INSTRUCTION'. 
  Example: 'FAIL: You used 2.5B for revenue, but Research Artifact shows 26.1B. Update your variable and re-run.'

Important Rule:
- Unit Check: 39,100M, 39.1B, and 39,100,000,000 are the SAME value. Do not fail for these.
- Discrepancy Check: If the agent provided a discrepancy_note in the JSON, and that note explains why a calculation (like 329% growth) differs from the text (75% growth) based on the raw revenue, ACCEPT THE MATH and PASS.
- JSON Extraction: You must look at the string inside STDOUT. If that string contains a valid JSON block, even if it has other text around it, it is a PASS on the format check.


RESEARCH DATA FOUND:
    {research_content}
    
    CODE EXECUTED:
    {code}
    
    STDOUT RESULT:
    {stdout}
"""

async def auditor_node(state: QuantInput):
    # 1. If the task was marked "failed" by agent node, nothing left for auditor to check. 
    # It is the scheduler job to decide the next step
    task = next(t for t in state["plan"] if t.id == state["task_id"])
    if task.status == "failed":
        return {
            "audit_status": "passed",
            "audit_feedback": "Task marked failed by Quant Analyst - No Audition Needed"
        }

    quant_artifact = next(
        a for a in reversed(state["artifacts"]) if a.artifact_type == "quantitative_analyst"
    )
    if not quant_artifact.success and state.get("retry_count", 0) >= MAX_ITERATION:
        content = (
            quant_artifact.content
            if isinstance(quant_artifact.content, dict)
            else {}
        )
        return {
            "plan": [
                TaskUpdate(
                    id=state["task_id"],
                    status="failed",
                    error_message=(
                        content.get("error")
                        or content.get("stderr")
                        or "Quant execution failed after max retries"
                    ),
                )
            ],
            "audit_status": "failed",
            "audit_feedback": "Quant execution retries exhausted",
        }
        
    # 2. Get the latest artifacts from quant node and research node
    research_context = format_evidence_bundle(state["artifacts"], sources=("research",))
    
    # 3. Extract code and result
    code = quant_artifact.content.get("code")
    stdout = quant_artifact.content.get("stdout")

    # 4. Auditor Prompt
    messages = []
    auditor_prompt = AUDITOR_SYSTEM_PROMPT.format(
        research_content= research_context,
        code = code,
        stdout = stdout
    )
    
    response = await model.ainvoke([SystemMessage(content=auditor_prompt)])
    messages.append(response.content)
    
    if "PASS" in response.content:
        return {
            "plan": [
                TaskUpdate(
                    id=state["task_id"], 
                    status="completed", 
                    error_message=None
                )
            ],
            "audit_status": "passed",
            "messages": messages
        }
    retry_count = state.get("retry_count", 0)
    if retry_count < MAX_ITERATION:
        return {
            "audit_status": "failed",
            "audit_feedback": response.content,
            "retry_count": retry_count + 1,
            "messages": messages,
        }

    print("MAX RETRIES REACHED: Moving on despite audit failure.")
    return {
        "plan": [
            TaskUpdate(
                id=state["task_id"],
                status="failed",
                error_message=response.content,
            )
        ],
        "audit_status": "failed",
        "audit_feedback": response.content,
        "retry_count": retry_count + 1,
        "messages": messages,
    }


def route_audit_subgraph(state: QuantInput):
    if state.get("audit_status") == "passed":
        return END

    task = next(t for t in state["plan"] if t.id == state["task_id"])
    if task.status == "failed":
        return END

    if state.get("retry_count", 0) >= MAX_ITERATION:
        return END

    return "quant_node"



#################################
######### Mock Quant ############
#################################
 

async def quant(state: QuantInput):
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
        artifact_type="quant", 
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