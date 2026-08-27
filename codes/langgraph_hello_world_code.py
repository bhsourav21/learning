from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    greeting: str

def node(state: State) -> State:
    return {"greeting": f'Hello {state["greeting"]}'}

graph = StateGraph(State)
graph.add_node("node_1", node)
graph.add_edge(START, "node_1")
graph.add_edge("node_1", END)

app = graph.compile()
result = app.invoke({"greeting": "Sourav"})
print(result)
