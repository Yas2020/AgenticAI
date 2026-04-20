## Overview

Built a production-oriented multi-agent system for automated investment research using modern agentic AI patterns.

The system demonstrates:

- multi-agent coordination and task decomposition
- tool-augmented reasoning (web search + code execution)
- workflow orchestration using DAG-based planning
- persistence and long-running execution
- evaluation and self-reflection loops

**Example task:**
User query → “Analyze NVIDIA and generate an investment report”

**Output:**
Structured markdown report with qualitative analysis and quantitative insights (charts, simulations)

## Why This Matters

This project demonstrates how modern ML systems move beyond single-model pipelines into **coordinated agent systems** that:

- combine reasoning, tools, and execution
- handle long-running workflows
- enforce reliability via validation loops
- separate planning from execution

This mirrors real-world production systems where ML models interact with multiple services and require orchestration, monitoring, and fault tolerance.

## Architecture

The system follows an **Orchestrator–Worker (hierarchical) design**:

- **Orchestrator**
  - Planner → generates DAG of tasks
  - Scheduler → manages execution and dependencies

- **Research Agent**
  - Performs web search via MCP tools

- **Quant Agent**
  - Generates and executes Python code for analysis

- **Auditor**
  - Validates outputs and enforces correctness (reflection loop)

- **Analyst**
  - Produces final investment report

I used LangGraph for implementation. 

![](./images/graph.png)


### User Query Validator
User query is first validated for safety, clearance and relevance to the topic. This is done by a small model. If not passed, the user is ask to edit the query (HITL). After maximum number of iteration of failures, graph ends.

### Orchestrator
Controls the full workflow using a graph-based execution model.

**Framework:** LangGraph

Responsibilities:
- state management
- task orchestration
- retries and failure handling
- agent coordination

#### Planner

- Converts user query into a **DAG of dependent tasks**
- Tasks are assigned to specialized agents (Research, Quant, Analyst)
- Plan is validated for:
  - DAG correctness (no cycles)
  - valid dependencies

If the plan is valid, it is sent to the scheduler to be distributed among agents. 

#### Scheduler

- Tracks task states: `pending → ready → running → completed/failed`
- Dispatches tasks to agents when dependencies are satisfied
- Handles:
  - retries
  - failure escalation (human-in-the-loop)
- Terminates when all tasks complete

### Research Agent
This agent's job is to gather fresh data from the web or retrieve documents for database. In this application, it only performs web search using **MCP tools**. It summarizes information into the researcher artifacts.

- Tools: **Tavily Search** or DuckDuckGo are great for structured research.
- Strategy: This agent might receive multiple independent tasks from the scheduler for multiple searches from different angles (e.g., competitors, financial, recent news) which will be conducted in parallel. Then the result will be summarized and saved into the researcher's artifact. 

### Quant subgraph
This subgraph consists of Quant agent, a deterministic validator and the Auditor agent.

#### Quant Analyst
Quant agent implements Program-Aided Language (PAL) Models. It is the "Calculation" agent that should never "hallucinate" math; it should write and run code instead. It can run data analytics 

- Tools: A Python REPL or Sandbox environment provided via an MCP server 
- Environment (Quant Sandbox): Uses Modal Sandboxes to execute untrusted code safely without crashing the main graph
- Role: Transforms raw financial statements from the Research agent into visualizations, charts, or complex ratio analyses. It can run Monte Carlo simulations, data analytics tools to provide quantitative support for the final report.  

#### Auditor
Perform self-reflection task. Checks for hallucinations, logical or runtime errors, discrepancy, and provides feedback. Judges the results from Quant node for required outputs, validity of code and data used in the code, discrepancies between the results from Quant agent and research data. PASS or FAIL if requirements not met.

If failed, the task will be returned to the Quant Agent with the feedback from Auditor
- This cycle repeats for a MAX_ITERATION allowed
- After max retires reached, the task is marked failed by Auditor and returned to the scheduler

Responsibilities:

• verify results
• detect hallucinations
• suggest retries

Often used in Evaluator‑Optimizer loops.

Pattern:

Agent A → generates output
Agent B → critiques output
Agent A → improves result


### Analyst
The Analyst produces a "Candidate Report". The analyst doesn't search; it "thinks" over the gathered data.

- Strategy: Use a "Chain of Thought" prompt to ensure it doesn't skip steps when transforming raw data into investment insights.
- Role: Acts as the data scientist's assistant, looking for underlying trends and risks.

The final report in markdown provides insight into the investment the user is asking for.


### Memory Component
Stores intermediate results and long‑term information.

- short‑term conversation memory
- vector database (not implemented here)
- Stores graph checkpoints.
    - Backend:
        - Postgres
    - Allows:
        - resume runs: LangGraph automatically restores the last checkpoint.
            - Run crashes during Analyst agent.
            - Next invocation resumes from Analyst node.
        - crash recovery
        - long‑running tasks
    - This automatically stores:
        - node state
        - graph transitions
        - outputs
        - thread memory


### Reliability Improvements
To make system reliable, it must address:

#### Semantic Guardrails
Use Pydantic schemas to enforce strict agent communication.

If an agent sends malformed data:

- Graph triggers retry logic.

#### Token Budget Monitor
- Prevents infinite reasoning loops.
- Tracks cumulative token cost.
- System self‑terminates if limits exceeded.

#### Traceability
- Observability tools allow debugging of reasoning decisions.

- **LangSmith** traces show agent thought chains.


## Other Considerations
### Notebooks
Notebooks are used to communicate with the graph - of course its not ideal because of

- SSE streams stall
- event loop blocking
- poor concurrency

```json
Notebook → HTTP request → LangGraph API (FastAPI container) → MCP server → external APIs
```


### MCP Session Management

Problem:
- Closing MCP sessions per request causes
`ClosedResourceError`.

Solution:
- Use persistent session tied to application lifetime.

MCPManager
- Responsibilities:
    - maintain single async MCP session
    - cache tools
    - manage connection lifecycle
    - Concurrency Protection
        - Use `asyncio.Lock` during initialization.
        - Prevents race conditions when multiple requests load tools.

- MCP startup should be integrated into FastAPI lifespan. This guarantees:
    - MCP session ready before requests
    - checkpointer initialized


### FastAPI Runtime Layer
Responsibilities:

- initialize services
- expose API endpoints
- run graphs
- SSE Streaming 
    - Persistent runs using checkpointer


Responsibilities:
- streaming responses using SSE
- trigger async graph execution
- pass thread_id
- production deployment path

It separates orchestration from experimentation.

### Why MCP
MCP standardizes tool discovery and execution.

Benefits:

- decouples agents from tool implementations
- enables external tool services
- future compatibility with agent ecosystems

### Why Postgres Checkpointing
Postgres provides:
- reliability
- persistence
- ability to resume workflows

This is essential for long-running agent pipelines.

#### Parallel Execution
LangGraph supports parallel users via thread_id.

Example:

- User A → thread_id = nvidia
- User B → thread_id = tesla

Each run isolated.



## Project Limitations (Honest Critique)

- Limited Evaluation Framework
The project lacks automated evaluation metrics.
    - Future improvement:
        - integrate evaluation datasets
        - automatic grading of agent outputs

- Limited Observability
Current tracing is minimal.
    - Future improvement:
        - integrate LangSmith
        - visualize agent execution traces

- No Cost Monitoring
LLM token usage not tracked.
    - Future improvement:
        - token budget monitoring
        - cost dashboards

- No Automatic Retry Policies
Currently failures require manual inspection.
    - Future improvement:
        - structured retry strategies
        - fallback models


- No replayable runs
    - run IDs for debugging
    - agent execution timeline


### run it

## Run

```bash
# Start API
uvicorn app.main:app --reload

```



