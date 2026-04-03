# mcp_servers/search/server.py
import httpx
import os
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timezone
from app.schemas.artifact import Artifact

import logging
import sys

# Configure logging to show internal MCP message routing
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# You can specifically target the MCP server logger for less noise
logging.getLogger("mcp.server").setLevel(logging.DEBUG)
logging.getLogger("fastmcp").setLevel(logging.DEBUG)


# Create the server
mcp = FastMCP("InvestmentResearch", host="0.0.0.0", port=3000)

@mcp.tool()
async def search_market_data(query: str) -> Artifact:
    """
    Search the web for real-time investment data, SEC filings, and news.
    Returns a structured artifact.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key, 
                    "query": query, 
                    "search_depth": "advanced",
                    "max_results": 5
                },
            )
            response.raise_for_status()
            data = response.json()
            results = [f"Source: {r['url']}\nContent: {r['content']}" 
                       for r in data.get("results", [])]

            content = "\n\n---\n\n".join(results) if results else "No relevant results found."

            return Artifact(
                source="Tavily API",
                content=content,
                timestamp=datetime.now(timezone.utc),
            )

    except Exception as e:
        return Artifact(
            source="Tavily API",
            content="",
            timestamp=datetime.now(timezone.utc),
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
    )