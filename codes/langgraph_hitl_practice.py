from typing import TypedDict, Annotated
from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    request: str
    approved: bool

def request_approval(state: State) -> State:
    decision = interrupt({"question": f"Approve this? {state['request']}"})
    return {"approved": decision}

graph = StateGraph(State)
graph.add_node("request_approval", request_approval)
graph.add_edge(START, "request_approval")
graph.add_edge("request_approval", END)

app = graph.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "approval_1"}}

out = app.invoke({"request": "return $500", "approved": False}, config)
print(f"response:{out}")

out2 = app.invoke(Command(resume=True), config)
print(f"resumed: {out2}")


