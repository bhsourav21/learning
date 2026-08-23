from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal

class State(TypedDict):
    amount: float
    route: str

def check_amount(state: State) -> State:
    return {}

def check_amount_func(state: State) -> Literal["auto_approved", "need_manual_review"]:
    return "auto_approved" if state["amount"] < 100 else "need_manual_review"

def auto_approve(state: State) -> State:
    return {"route": "auto approved"}

def human_review(state: State) -> State:
    return {"route": "Sent for human review"}

graph = StateGraph(State)
graph.add_node("check_amount", check_amount)
graph.add_node("auto_approve", auto_approve)
graph.add_node("human_review", human_review)
graph.add_edge(START, "check_amount")
graph.add_conditional_edges(
    "check_amount", check_amount_func, {"auto_approved": "auto_approve", "need_manual_review": "human_review"}, 
)
graph.add_edge("auto_approve", END)
graph.add_edge("human_review", END)

app = graph.compile()
print(app.invoke({"amount": 42}))
print(app.invoke({"amount": 420}))