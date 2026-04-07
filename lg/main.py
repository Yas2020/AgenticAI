import json
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from app.schemas.api import GraphRequest
from langchain_core.load import dumpd
from app.core.engine import graph   # your compiled graph
from app.services.mcp.mcp_clients import mcp_manager
from contextlib import asynccontextmanager
from app.services.langgraph_postgres.checkpointer import checkpointer, pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    async with pool:
        # 1️⃣ Setup Postgres checkpointer tables
        await checkpointer.setup()
        print("✅ LangGraph checkpointer tables initialized.")

        # 2️⃣ Start MCP session
        for manager in mcp_manager.values():
            await manager.startup()
        print("✅ MCP session initialized for all clients.")

        # 3️⃣ App is now fully ready for agents to run safely
        yield  

    # --- Shutdown Logic ---
    for manager in mcp_manager.values():
        await manager.shutdown()
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
        async for chunk in graph.astream(
            inputs, 
            config=config, 
            stream_mode="updates", 
            version="v2"
        ):
            # chunk["data"] is a dict: { "node_name": { "state_key": "new_value" } }
            for node_name, state_update in chunk["data"].items():
                # 1. Guard against NoneType state updates
                if state_update is None:
                    continue
                # Extract just the message content if it exists
                if isinstance(state_update, dict) and "messages" in state_update and state_update["messages"]:
                    last_msg = state_update["messages"][-1]
                    # Extracts text from Human/AI/Tool messages
                    content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                else:
                    # 2. If it's not a message (like an Artifact or Dict), 
                    # use dumpd to make it JSON-safe or just str() it
                    content = dumpd(state_update) # Fallback for non-message nodes

                yield json.dumps({
                    "node": node_name,
                    "content": content,
                    "state": dumpd(state_update) 
                })

    return EventSourceResponse(event_generator())

    # async def event_generator():
    #     last_event = None
    #     try:
    #         async for event in graph.astream_events(
    #             inputs,
    #             config=config,
    #             version="v2" # Recommended version
    #         ):
    #             last_event = event
                
    #             # Filter for relevant events
    #             if event["event"] in ["on_chain_start", "on_chain_end", "on_tool_start", "on_tool_end", "on_llm_end"]:
    #                 yield f"data: {json.dumps(dumpd(event))}\n\n"

    #             if await request.is_disconnected():
    #                 break
            
    #         # Send final event as a string
    #         final_data = json.dumps({"event": "final_state", "data": dumpd(last_event)})
    #         yield f"data: {final_data}\n\n"

    #     except Exception as e:
    #         # Errors must also be yielded as data strings
    #         error_data = json.dumps({"event": "error", "message": str(e)})
    #         yield f"data: {error_data}\n\n"

    # return EventSourceResponse(event_generator())
