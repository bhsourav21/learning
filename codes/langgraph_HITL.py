from typing import TypedDict
from langgraph.types import Command, interrupt

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated

class State(TypedDict):
    request: str
    approved: str

def request_approval(state: State) -> State:
    decision = interrupt({"question": f"Approve this? {state['request']}"})
    return {"approved": decision}

graph = StateGraph(State)
graph.add_node("request_approval", request_approval)
graph.add_edge(START, "request_approval")
graph.add_edge("request_approval", END)

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "user_thread_1"}}

out = app.invoke({"request": "refund $500", "approved": False}, config)
print(f"paused:{out}")

out2 = app.invoke(Command(resume='no'), config)
print(f"resumed:{out2}")
