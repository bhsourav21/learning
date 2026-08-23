from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    greeting: str

def say_hello(state: State) -> State:
    return {"greeting": f"Hello, {state['greeting']}!"}

graph = StateGraph(State)
graph.add_node("hello", say_hello)
graph.add_edge(START, "hello")
graph.add_edge("hello", END)

app = graph.compile()
print(app.invoke({"greeting": "world"}))
