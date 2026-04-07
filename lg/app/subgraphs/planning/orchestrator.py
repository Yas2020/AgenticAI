from langgraph.graph import END
from langgraph.constants import Send
from app.core.state import MasterState


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

    # Update the status to running - we use dict here not TaskUpdate which is reserved for agent nodes only - it can not updpate to "running"
    return {
        "plan": [{"id": t.id, "status": "running", "error_message": None} for t in state["plan"] if t.status == "ready"]
    }



def route_to_agents(state: MasterState):
    
    # If nothing is running and everything is completed, NOW return END
    if all(t.status == "completed" for t in state["plan"]):
        return END
    
    # Fan out tasks to the corresponding agents
    return [
        Send(t.agent, {**state, "task_id": t.id}) 
        for t in state["plan"] if t.status == "running"
    ]


    
    
    # return Command(
    #     # update={"some_state_key": "some_value"}, # Optional state update
    #     goto=[
    #         Send(task.agent, {**state, "task_id": task.id}) 
    #         for task in state["plan"] if task.status == "running"
    #     ]
    # )

    
 
