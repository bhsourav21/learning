from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from typing import Literal
from langchain_core.tools import tool

@tool
def check_policy(amount: float) -> str:
    """Check company policy for an expense amount"""
    return "needs_review" if amount > 500 else "auto_ok"

class ExpenseState(TypedDict):
    amount: float
    memo: str
    polic_result: str
    decision: str

def check_expense(state: ExpenseState) -> ExpenseState:
    result = check_policy.invoke({"amount": state["amount"]})
    return {"polic_result": result}

def route_after_check(state: ExpenseState) -> Literal["auto_approve", "ask_human"]:
    return "auto_approve" if state["polic_result"] == "auto_ok" else "ask_human"

def auto_approve(state: ExpenseState) -> ExpenseState:
    return {"decision": f"Auto-approved ${state["amount"]} for {state["memo"]}"}

def ask_human(state: ExpenseState) -> ExpenseState:
    human_says = interrupt(
        {
            "question": f"Expense of ${state['amount']} for {state['memo']} exceeds policy. Approve?"
        }
    )
    verdict = "Approved" if human_says else "Rejected"
    return {"decision": f"{verdict} by human review: ${state["amount"]} for {state["memo"]}."}

graph = StateGraph(ExpenseState)
graph.add_node("check_expense", check_expense)
graph.add_node("auto_approve", auto_approve)
graph.add_node("ask_human", ask_human)
graph.add_edge(START, "check_expense")
graph.add_conditional_edges(
    "check_expense", route_after_check,
    {
        "auto_approve": "auto_approve", 
        "ask_human": "ask_human"
    }
)
graph.add_edge("auto_approve", END)
graph.add_edge("ask_human", END)

app = graph.compile(checkpointer=InMemorySaver())

# Case 1: small amount, no human needed
cfg1 = {"configurable":{"thread_id": "expense-1"}}
out1 = app.invoke({
    "amount": 120,
    "memo": "team lunch",
},
cfg1
)

#Case 2: large amount, pause for a human
cfg2 = {"configurable":{"thread_id": "expense-2"}}
out2 = app.invoke({
    "amount": 900,
    "memo": "conference travel",
},
cfg2
)

print("Case 2: (paused):", "__interrupt__" in out2)

out2b = app.invoke(Command(resume=False), cfg2)
print("Case 2: (resumed):", out2b["decision"])

