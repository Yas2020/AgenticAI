FROM python:3.13-slim

# CRITICAL: Force logs to show up in real-time
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure LangSmith tracing is active by default in the container
ENV LANGCHAIN_TRACING_V2=true

WORKDIR /app

# Install system dependencies (needed for some database drivers)
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*


# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start with uvicorn. 
# '--log-level debug' helps verify tracing status on startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
