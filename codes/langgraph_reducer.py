from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]

def add_greeting(state: State) -> State:
    return {"messages": [AIMessage(content="this is ai message")]}

def add_greeting_1(state: State) -> State:
    return {"messages": [AIMessage(content="this is ai message 1")]}

graph = StateGraph(State)
graph.add_node("add_greeting", add_greeting)
graph.add_node("add_greeting_1", add_greeting_1)
graph.add_edge(START, "add_greeting")
graph.add_edge("add_greeting", "add_greeting_1")
graph.add_edge("add_greeting_1", END)

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="this is human message")]})

for m in result["messages"]:
    print(type(m).__name__, "->", m.content)