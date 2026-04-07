from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

DB_URI = "postgresql://postgres:postgres@postgres:5432/langgraph"

# Create a global pool and checkpointer instance
pool = AsyncConnectionPool(
    conninfo=DB_URI, 
    max_size=20,
    open=False,
    kwargs={"autocommit": True}
)
checkpointer = AsyncPostgresSaver(pool)

print("LangGraph tables created.")