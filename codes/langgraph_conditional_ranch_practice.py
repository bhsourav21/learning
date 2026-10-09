from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    amount: int
    route: str

def check_amount(state: State) -> State:
    return {}

def route_by_amount(state: State) -> Literal["auto_approved", "manual_review_needed"]:
    return "auto_approved" if state["amount"] < 40 else "manual_review_needed"

def auto_approve(state: State) -> State:
    return{"route": "auto-approved"}

def manual_review(state: State) -> State:
    return{"route": "Need to be reviewed manually"}

graph = StateGraph(State)
graph.add_node("check_amount", check_amount)
graph.add_node("auto_approve", auto_approve)
graph.add_node("manual_review", manual_review)
graph.add_edge(START, "check_amount")
graph.add_conditional_edges(
    "check_amount", route_by_amount, {"auto_approved": "auto_approve", "manual_review_needed": "manual_review"},
)
graph.add_edge("auto_approve", END)
graph.add_edge("manual_review", END)

app = graph.compile()
print(app.invoke({"amount": 100, "route": ""}))
print(app.invoke({"amount": 40, "route": ""}))
print(app.invoke({"amount": 39, "route": ""}))