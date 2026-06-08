import json

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app.core.engine import graph
from app.schemas.api import FeedbackRequest, GraphRequest, GraphResumeRequest
from app.services.api.sse import stream_graph
from app.services.langgraph_postgres.checkpointer import checkpointer, pool
from app.services.mcp.mcp_clients import mcp_manager
from app.services.observability.langfuse_tracer import is_langfuse_enabled
from app.services.observability.user_feedback import record_user_feedback


def _graph_config(req_thread) -> dict:
    config = req_thread.model_dump() if req_thread else {}
    configurable = config.setdefault("configurable", {})
    thread_id = configurable.get("thread_id")

    if is_langfuse_enabled() and thread_id:
        try:
            from langfuse.langchain import CallbackHandler

            config["callbacks"] = [CallbackHandler()]
            metadata = dict(config.get("metadata", {}))
            metadata.update(
                {
                    "langfuse_session_id": thread_id,
                    "langfuse_tags": ["app_run", "investment_research"],
                }
            )
            config["metadata"] = metadata
        except Exception:
            pass

    return config


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
    if not req.thread or not req.thread.configurable.get("thread_id"):
        raise HTTPException(
            status_code=400,
            detail="thread.configurable.thread_id is required for HITL resume and checkpointing.",
        )

    inputs = {
        "messages": [HumanMessage(content=m.content) for m in req.messages],
        "topic": req.topic,
    }
    config = _graph_config(req.thread)

    return EventSourceResponse(stream_graph(graph, inputs, config, request))


@app.post("/graph-resume")
async def resume_graph_stream(req: GraphResumeRequest, request: Request):
    thread_id = req.thread.configurable.get("thread_id")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread.configurable.thread_id is required.")

    config = _graph_config(req.thread)
    return EventSourceResponse(
        stream_graph(graph, Command(resume=req.resume.strip()), config, request)
    )


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    return record_user_feedback(
        thread_id=req.thread_id,
        rating=req.rating,
        comment=req.comment,
        trace_id=req.trace_id,
    )
