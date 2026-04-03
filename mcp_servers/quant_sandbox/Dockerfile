# 3. mcp_servers/quant_sandbox/Dockerfile (Execution Environment)
FROM python:3.11-slim

WORKDIR /app

# Install Data Science stack for the Quant Agent
RUN pip install --no-cache-dir \
    mcp[server] \
    pandas \
    numpy \
    matplotlib \
    yfinance \
    scipy \
    statsmodels

COPY . .

EXPOSE 3001

# Quant sandbox usually needs to execute code strings
CMD ["python", "server.py", "--transport", "sse", "--port", "3001"]
