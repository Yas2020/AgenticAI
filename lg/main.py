import json
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage
from app.schemas.api import GraphRequest
from langchain_core.load import dumpd
from app.core.engine import graph
from app.services.mcp.mcp_clients import mcp_manager
from contextlib import asynccontextmanager
from app.services.langgraph_postgres.checkpointer import checkpointer, pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with pool:
        await checkpointer.setup()
        print("✅ LangGraph checkpointer tables initialized.")

        for manager in mcp_manager.values():
            await manager.startup()
        print("✅ MCP session initialized for all clients.")

        yield

    for manager in reversed(list(mcp_manager.values())):
        await manager.shutdown()
    print("🛑 Application shutting down...")


app = FastAPI(title="LangGraph API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/graph-stream")
async def run_graph_stream(req: GraphRequest, request: Request):
    inputs = {
        "messages": [HumanMessage(content=m.content) for m in req.messages],
        "topic": req.topic,
    }
    config = req.thread.model_dump() if req.thread else {}

    async def event_generator():
        try:
            async for chunk in graph.astream(
                inputs,
                config=config,
                stream_mode="updates",
                version="v2",
            ):
                if await request.is_disconnected():
                    break

                for node_name, state_update in chunk["data"].items():
                    if state_update is None:
                        continue

                    if (
                        isinstance(state_update, dict)
                        and "messages" in state_update
                        and state_update["messages"]
                    ):
                        last_msg = state_update["messages"][-1]
                        content = (
                            last_msg.content
                            if hasattr(last_msg, "content")
                            else str(last_msg)
                        )
                    else:
                        content = dumpd(state_update)

                    yield json.dumps(
                        {
                            "event": "update",
                            "node": node_name,
                            "content": content,
                            "state": dumpd(state_update),
                        }
                    )

            yield json.dumps({"event": "done"})
        except Exception as e:
            yield json.dumps({"event": "error", "message": str(e)})

    return EventSourceResponse(event_generator())
