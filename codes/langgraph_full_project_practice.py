from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

@tool
def check_policy(amount: float) -> str:
    """Check company policy for expense amount."""
    if amount > 500:
        return "needs_review"
    else:
        return "auto_ok"

class ExpenseState(TypedDict):
    amount: float
    memo: str
    policy_result: str
    decision: str

def check_expense(state: ExpenseState) -> ExpenseState:
    result = check_policy.invoke({"amount": state["amount"]})
    return{"policy_result": result}

def route_after_check(state: ExpenseState) -> Literal["auto_approve", "ask_human"]:
    if state["policy_result"] == "needs_review":
        return "ask_human"
    else:
        return "auto_approve"

def auto_approve(state: ExpenseState) -> ExpenseState:
    return {"decision": f"Auto approved for {state['amount']} for memo {state["memo"]}"}

def ask_human(state: ExpenseState) -> ExpenseState:
    human_says = interrupt(
        {
            "question": f"Expense of ${state['amount']} for {state['memo']} has been received. Approve?"
        }
    )
    verdict = "Approved" if human_says else "Rejected"
    return{"decision": verdict}


graph = StateGraph(ExpenseState)
graph.add_node("check_expense", check_expense)
graph.add_node("auto_approve", auto_approve)
graph.add_node("ask_human", ask_human)
graph.add_edge(START, "check_expense")
graph.add_conditional_edges(
    "check_expense",
    route_after_check,
    {
        "ask_human": "ask_human",
        "auto_approve": "auto_approve"
    }
)
graph.add_edge("ask_human", END)
graph.add_edge("auto_approve", END)

app = graph.compile(checkpointer=InMemorySaver())

cfg1 = {
    "configurable": {
        "thread_id": "exp-1"
    }
}

cfg2 = {
    "configurable": {
        "thread_id": "exp-2"
    }
}

def run_case(label, payload, cfg, approve=True):
    """Invoke the graph, and if it paused for a human, answer and resume."""
    out = app.invoke(payload, cfg)

    # A graph that hit interrupt() returns a "__interrupt__" key instead of
    # finishing. Its value is a LIST of Interrupt objects (a node can raise
    # more than one), each carrying the payload we passed to interrupt().
    if "__interrupt__" in out:
        for itr in out["__interrupt__"]:
            print(f"{label}: PAUSED -> {itr.value['question']}")
        print(f"{label}: resuming with approve={approve}")

        # Command(resume=X) makes X the RETURN VALUE of the interrupt() call
        # inside ask_human, so the node carries on from exactly where it
        # stopped. cfg is required: thread_id is what identifies which
        # paused run to resume, which is why a checkpointer is mandatory.
        out = app.invoke(Command(resume=approve), cfg)

    print(f"{label}: {out['decision']}")
    return out

out1 = run_case("case1", {"amount": 120, "memo": "fun",
                          "policy_result": "", "decision": ""}, cfg1)
out2 = run_case("case2", {"amount": 900, "memo": "travel",
                          "policy_result": "", "decision": ""}, cfg2, approve=True)






