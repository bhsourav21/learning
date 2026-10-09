from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage 

class State(TypedDict):
    messages: Annotated[list, add_messages]

# class State(TypedDict):
#     messages: str

def add_greeting(state: State) -> State:
    return {"messages": [AIMessage(content="hi there!")]}

graph = StateGraph(State)
graph.add_node("add_greeting", add_greeting)
graph.add_edge(START, "add_greeting")
graph.add_edge("add_greeting", END)

app = graph.compile()
result = app.invoke({"messages": HumanMessage(content="hello!")})

for m in result["messages"]:
    print(type(m).__name__, "->", m.content)





