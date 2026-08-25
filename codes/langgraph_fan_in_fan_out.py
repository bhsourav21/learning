from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Annotated
import operator
import time

class State(TypedDict):
    topic: str
    angles: Annotated[list[str], operator.add]

def angle_a(state: State) -> State:
    print("executed the node angle_a")
    return {"angles": [f"[history_angle] {state["topic"]}"]}

def angle_b(state: State) -> State:
    time.sleep(20)
    print("executed the node angle_b")
    return {"angles": [f"[economic_angle] {state["topic"]}"]}

def combine(state: State) -> State:
    return {}

graph = StateGraph(State)
graph.add_node("angle_a", angle_a)
graph.add_node("angle_b", angle_b)
graph.add_node("combine", angle_b)
graph.add_edge(START, "angle_a")
graph.add_edge(START, "angle_b")
graph.add_edge("angle_a", "combine")
graph.add_edge("angle_b", "combine")
graph.add_edge("combine", END)

app = graph.compile()
print(app.invoke({
    "topic": "trade routes",
    "angles": []
}))
