# scripts/test_graph.py
import asyncio
from app.core.engine import app # Your compiled LangGraph
from langchain_core.messages import HumanMessage

async def run_test():
    config = {"configurable": {"thread_id": "test_1"}}
    initial_input = {"messages": [HumanMessage(content="Analyze NVIDIA's Q3 revenue.")]}
    
    # Stream the graph execution
    async for event in app.astream(initial_input, config=config):
        for node_name, output in event.items():
            print(f"\n--- Node: {node_name} ---")
            if "plan" in output:
                # Check if the statuses are moving correctly
                statuses = [f"ID {t.id}: {t.status}" for t in output["plan"]]
                print(f"Task Statuses: {statuses}")

if __name__ == "__main__":
    asyncio.run(run_test())