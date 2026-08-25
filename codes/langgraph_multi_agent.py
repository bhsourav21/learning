from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    task: str
    result: str
    next_agent: str

def supervisor(state: State) -> State:
    next_agent = "researcher" if "research" in state["task"] else "writer"
    return {"next_agent": next_agent}

def route_supervisor(state: State) -> State:
    return state["next_agent"]

def researcher(state: State) -> State:
    return {"result": f'[researcher] gathered facts for {state["task"]}'}

def writer(state: State) -> State:
    return {"result": f'[writer] drafted text for {state["task"]}'}

graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_supervisor, \
                            { \
                                "researcher": "researcher", \
                                "writer": "writer" \
                            },)
graph.add_edge("researcher", END)
graph.add_edge("writer", END)

app = graph.compile()
print(app.invoke({"task": "research the history of chess", "result": "", "next_agent": ""}))
print(app.invoke({"task": "write a poem about autumn", "result": "", "next_agent": ""}))
# print(app.invoke({"task": "research the history of chess", "result": ""}))
# print(app.invoke({"task": "write a poem about autumn", "result": ""}))