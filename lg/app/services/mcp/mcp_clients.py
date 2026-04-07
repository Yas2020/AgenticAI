# app/services/mcp/mcp_client.py
import asyncio
from typing import Dict
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from langchain_mcp_adapters.tools import load_mcp_tools

class MCPManager:
    def __init__(self, url):
        self.url = url 
        self.session = None
        self.tools = None
        self._stack = AsyncExitStack()
        self._lock = asyncio.Lock() # <-- concurrency guard for the first session when multiple agnets want to open session

    async def startup(self):
        """Initializes the persistent MCP session"""
        async with self._lock:
            if self.session is not None:
                return

            print("Starting MCPManager...")
            print("Connecting to Research Server for the first time...")
            try:
                # 1. Enter the HTTP transport context
                # Note: streamable_http_client typically returns (read, write, get_session_id)
                streams = await self._stack.enter_async_context(
                    streamable_http_client(self.url)
                )
                read_stream, write_stream = streams[0], streams[1]
                
                # 2. Enter the Session context
                # This 'enter_async_context' triggers the background receive loops
                self.session = await self._stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await self.session.initialize()
                print("Successfully initialized MCP session.")
            except Exception as e:
                await self.shutdown()
                raise e

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
mcp_manager: Dict[str, MCPManager] = {
    "research": MCPManager("http://mcp-research:3000/mcp"),
    "quant": MCPManager("http://mcp-quant:3001/mcp")
}