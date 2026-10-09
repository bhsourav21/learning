from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    task: str
    result: str
    next_agent: str

def supervisor(state: State) -> Literal["goto_researcher", "goto_writer"]:
    next_agent = "goto_researcher" if "research" in state["task"] else "goto_writer"
    return {"next_agent": next_agent}

def route_supervisor(state: State) -> State:
    return state["next_agent"]

def researcher(state: State) -> State:
    return{"result": f"Researcher gathered facts for {state['task']}"}

def writer(state: State) -> State:
    return{"result": f"Wrote a draft on {state['task']}"}

graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "goto_researcher": "researcher",
        "goto_writer": "writer"
    }
)
graph.add_edge("researcher", END)
graph.add_edge("writer", END)

app = graph.compile()
print(app.invoke({"task": "research on Dr. Amartya Sen", "result": "", "next_agent": ""}))
print()
print(app.invoke({"task": "write a para on Dr. Amartya Sen"}))