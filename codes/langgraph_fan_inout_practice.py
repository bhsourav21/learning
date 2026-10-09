import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage 

class State(TypedDict):
    topic: str
    messages: Annotated[list[str], operator.add]

def first_a(state: State) -> State:
    return {"messages": [f"first_a for topic: {state['topic']}"]}

def first_b(state: State) -> State:
    return {"messages": [f"first_b for topic: {state['topic']}"]}

def combine(state: State) -> State:
    return {}

graph = StateGraph(State)

graph.add_node("first_a", first_a)
graph.add_node("first_b", first_b)
graph.add_node("combine", combine)
graph.add_edge(START, "first_a")
graph.add_edge(START, "first_b")
graph.add_edge("first_a", "combine")
graph.add_edge("first_b", "combine")
graph.add_edge("combine", END)

app = graph.compile()
result = app.invoke({"topic": "This is summer"})
print(result)





