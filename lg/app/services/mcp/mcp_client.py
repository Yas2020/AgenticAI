# app/services/mcp/mcp_client.py
import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from langchain_mcp_adapters.tools import load_mcp_tools

class MCPManager:
    def __init__(self):
        self.session = None
        self.tools = None
        self._stack = None  # Will be created at startup
        self._lock = asyncio.Lock() # <-- concurrency guard fro the first session when multiple agnets want to open session

    async def startup(self):
        """Call this at FastAPI startup to initialize MCP session"""
        async with self._lock:
            if self.session is not None:
                return

            print("Starting MCPManager...")
            self._stack = AsyncExitStack()
            await self._stack.__aenter__()  # Enter the stack manually

            print("Connecting to Research Server for the first time...")
            streams = await self._stack.enter_async_context(
                streamable_http_client("http://mcp-research:3000/mcp")
            )
            read_stream, write_stream = streams[0], streams[1]
            self.session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()
            await asyncio.sleep(0.1)  # let background tasks start
            print("Successfully initialized MCP session.")

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
        """Call this at FastAPI shutdown to close MCP session"""
        if self._stack:
            await self._stack.aclose()
        self.session = None
        self.tools = None
        self._stack = None
        print("MCPManager shutdown complete.")

# global instance
mcp_manager = MCPManager()