from fastapi import FastAPI, Request
import json
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from langchain_core.load import dumpd
from app.schemas.api import GraphRequest
from app.core.engine import graph   # your compiled graph
from app.services.mcp.mcp_client import mcp_manager

from contextlib import asynccontextmanager
from app.services.langgraph.checkpointer import checkpointer, pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    async with pool:
        # 1️⃣ Setup Postgres checkpointer tables
        await checkpointer.setup()
        print("✅ LangGraph checkpointer tables initialized.")

        # 2️⃣ Start MCP session
        await mcp_manager.startup()
        print("✅ MCP session initialized.")

        # 3️⃣ App is now fully ready for agents to run safely
        yield  

    # --- Shutdown Logic ---
    await mcp_manager.shutdown()
    print("🛑 Application shutting down...")

# Initialize FastAPI with the lifespan
app = FastAPI(title="LangGraph API", lifespan=lifespan)


@app.post("/graph-stream")
async def run_graph_stream(req: GraphRequest, request: Request):
    inputs = {
        "messages": [HumanMessage(content=m.content) for m in req.messages],
        "topic": req.topic,
    }
    
    # Ensure this contains {"configurable": {"thread_id": "..."}}
    config = req.thread.model_dump() if req.thread else {}

    async def event_generator():
        last_event = None
        try:
            async for event in graph.astream_events(
                inputs,
                config=config,
                version="v2" # Recommended version
            ):
                last_event = event
                
                # Filter for relevant events
                if event["event"] in ["on_chain_start", "on_chain_end", "on_tool_start", "on_tool_end", "on_llm_end"]:
                    yield f"data: {json.dumps(dumpd(event))}\n\n"

                if await request.is_disconnected():
                    break
            
            # Send final event as a string
            final_data = json.dumps({"event": "final_state", "data": dumpd(last_event)})
            yield f"data: {final_data}\n\n"

        except Exception as e:
            # Errors must also be yielded as data strings
            error_data = json.dumps({"event": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"

    return EventSourceResponse(event_generator())
