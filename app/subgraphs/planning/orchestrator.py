
from app.core.state import MasterState
from langgraph.graph import END
from langgraph.constants import Send
from langgraph.types import Command

def scheduler(state: MasterState):
    """
        Scheduler: Updates the status of tasks whose dependencies are 'complete' to 'ready'. 
        Tasks are assigned by the planning_architect.
        Scheduler acts as Dynamic Router: It looks for 'ready' tasks and dispatches them to their specific agent nodes.
    """
    
    # If all tasks complete, nothing left to do 
    if all(t.status == "completed" for t in state["plan"]):
        return {}
    
    task_map = {t.id: t for t in state["plan"]}

    for task in state["plan"]:
        if task.status == "pending" and all(task_map[dep].status == "completed" for dep in task.depends_on):
            task.status = "ready"

    ##############   
    # Todo: Decide what to do with failed tasks! retires? ... logic of retires
    ##############

    return {
        "plan": [{"id": t.id, "status": "running"} for t in state["plan"] if t.status == "ready"]
    }



def route_to_agents(state: MasterState):
    
    # If nothing is running and everything is completed, NOW return END
    if all(t.status == "completed" for t in state["plan"]):
        return END
    
    # This is where the magic happens for the Mermaid graph
    running_tasks = [t for t in state["plan"] if t.status == "running"]
    
    # Fan out tasks to the corresponding agents
    return [
        Send(task.agent, {**state, "task_id": task.id}) 
        for task in running_tasks
    ]

    
    
    # return Command(
    #     # update={"some_state_key": "some_value"}, # Optional state update
    #     goto=[
    #         Send(task.agent, {**state, "task_id": task.id}) 
    #         for task in state["plan"] if task.status == "running"
    #     ]
    # )

    
 
