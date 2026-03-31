from mcp.server.fastmcp import FastMCP
import httpx
import os

# Create the server
mcp = FastMCP("InvestmentResearch")

@mcp.tool()
async def search_market_data(query: str) -> str:
    """
    Search the web for real-time investment data, SEC filings, and news.
    """
    # Replace with your API of choice (Tavily is great for agentic search)
    api_key = os.getenv("TAVILY_API_KEY")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "search_depth": "advanced"}
        )
        data = response.json()
        
    # Clean up the output for the LLM
    results = [f"Source: {r['url']}\nContent: {r['content']}" for r in data.get("results", [])]
    return "\n\n---\n\n".join(results)

if __name__ == "__main__":
    mcp.run(transport="stdio") # Use stdio for local agent communication