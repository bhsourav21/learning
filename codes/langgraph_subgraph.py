from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Annotated
import operator
import time

class InnerState(TypedDict):
    value: int

def double(state: InnerState) -> InnerState:
    return{"value": state["value"] * 2}

inner = StateGraph(InnerState)
inner.add_node("double", double)
inner.add_edge(START, "double")
inner.add_edge("double", END)
inner_app = inner.compile()

class OuterState(TypedDict):
    value: int

outer = StateGraph(OuterState)
outer.add_node("doubling_subgraph", inner_app)
outer.add_edge(START, "doubling_subgraph")
outer.add_edge("doubling_subgraph", END)
outer_app = outer.compile()

print(outer_app.invoke({"value": 5}))

