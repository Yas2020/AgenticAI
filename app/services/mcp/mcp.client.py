import os
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters


async def get_research_tools():
    # Define how to start your local MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_servers/web_search/research_server.py"],
        env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
    )
    
    # This helper automatically discovers the @mcp.tool() functions
    return await load_mcp_tools(server_params)