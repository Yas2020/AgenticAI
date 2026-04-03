# 2. mcp_servers/research/Dockerfile (Streaming HTTP MCP)
FROM python:3.11-slim

WORKDIR /app

# Install MCP Python SDK and search tools
RUN pip install --no-cache-dir mcp[server] httpx beautifulsoup4

# # Copy your specific research server code
COPY . .

# MCP Servers over HTTP usually need a specific port
EXPOSE 3000

# Replace 'server.py' with your entry point
# Using --transport sse for Streaming HTTP
CMD ["python", "server.py", "--port", "3000"]
