# app/services/mcp/mcp_client.py
import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Dict

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)


def _benign_shutdown_error(exc: BaseException) -> bool:
    """MCP streamable-http often raises GeneratorExit/TaskGroup noise on close."""
    if isinstance(exc, (GeneratorExit, asyncio.CancelledError, RuntimeError)):
        message = str(exc).lower()
        if "cancel scope" in message or "different task" in message:
            return True
    if isinstance(exc, BaseExceptionGroup):
        return all(_benign_shutdown_error(e) for e in exc.exceptions)
    return isinstance(exc, GeneratorExit)


class MCPManager:
    def __init__(self, url, name: str = "mcp"):
        self.url = url
        self.name = name
        self.session = None
        self.tools = None
        self._stack: AsyncExitStack | None = AsyncExitStack()
        self._lock = asyncio.Lock()

    async def startup(self, timeout: float | None = 30.0):
        """Initializes the persistent MCP session."""
        async with self._lock:
            if self.session is not None:
                return

            self._stack = AsyncExitStack()

            print(f"Starting MCPManager ({self.name})...")
            print(f"Connecting to {self.url}...")

            async def _connect() -> None:
                streams = await self._stack.enter_async_context(
                    streamable_http_client(self.url)
                )
                read_stream, write_stream = streams[0], streams[1]
                self.session = await self._stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await self.session.initialize()

            try:
                if timeout is None:
                    await _connect()
                else:
                    await asyncio.wait_for(_connect(), timeout=timeout)
                print(f"Successfully initialized MCP session ({self.name}).")
            except Exception as e:
                await self.shutdown()
                raise RuntimeError(
                    f"MCP startup failed for {self.name} at {self.url}: {e}"
                ) from e

    async def get_session(self):
        """Return the already-open MCP session"""
        if self.session is None:
            raise RuntimeError("MCP session not initialized. Call startup() first.")
        return self.session

    async def get_tools(self):
        """Load tools once and cache"""
        if self.tools is None:
            session = await self.get_session()
            print("Loading MCP tools...")
            self.tools = await load_mcp_tools(session)
        return self.tools

    async def shutdown(self):
        """Close MCP session and HTTP transport."""
        async with self._lock:
            if self.session is None and self._stack is None:
                return

            stack = self._stack
            self.session = None
            self.tools = None
            self._stack = None

            if stack is None:
                return

            try:
                await stack.aclose()
            except BaseException as exc:
                if not _benign_shutdown_error(exc):
                    logger.warning(
                        "MCP shutdown warning for %s: %s", self.name, exc, exc_info=exc
                    )

            self._stack = AsyncExitStack()
            print(f"MCPManager shutdown complete ({self.name}).")


# global instance
mcp_manager: Dict[str, MCPManager] = {
    "research": MCPManager(
        os.getenv("RESEARCH_MCP_URL", "http://mcp-research:3000/mcp"),
        name="research",
    ),
    "quant": MCPManager(
        os.getenv("QUANT_MCP_URL", "http://mcp-quant:3001/mcp"),
        name="quant",
    ),
}
