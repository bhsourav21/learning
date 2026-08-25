from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class InnerState(TypedDict):
    amount: int

def double(state: InnerState) -> InnerState:
    return {"amount": state["amount"] * 2}

inner = StateGraph(InnerState)
inner.add_node("double", double)
inner.add_edge(START, "double")
inner.add_edge("double", END)
inner_app = inner.compile()

class OuterState(TypedDict):
    number: int

def run_doubling_subgraph(state: OuterState) -> OuterState:
    """Different schemas can't be composed directly -- translate in, invoke, translate out"""
    inner_result = inner_app.invoke({"amount": state["number"]})
    return {"number": inner_result["amount"]}

outer = StateGraph(OuterState)
outer.add_node("doubling_subgraph", run_doubling_subgraph)
outer.add_edge(START, "doubling_subgraph")
outer.add_edge("doubling_subgraph", END)
outer_app = outer.compile()

print(outer_app.invoke({"number": 5}))
