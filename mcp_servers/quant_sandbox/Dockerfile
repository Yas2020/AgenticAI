# mcp_servers/quant_sandbox/Dockerfile (Execution Environment)
FROM python:3.11-slim

# Install system dependencies for scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

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


# Use an argument so you can change it at build time if needed
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create the user with the SPECIFIC IDs from your host
RUN groupadd -g ${GROUP_ID} quantgroup && \
    useradd -u ${USER_ID} -g quantgroup -m quantuser

# Ensure the artifacts directory exists and is owned by our new user
RUN mkdir -p /app/artifacts && chown quantuser:quantgroup /app/artifacts

# Quant sandbox usually needs to execute code strings
CMD ["python", "server.py", "--port", "3001"]
