from typing import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]

def echo(state: State) -> State:
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"You said:{last}")]}

graph = StateGraph(State)
graph.add_node("echo", echo)
graph.add_edge(START, "echo")
graph.add_edge("echo", END)

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

config_1 = {"configurable": {"thread_id": "user-42"}}
config_2 = {"configurable": {"thread_id": "user-43"}}

app.invoke({"messages": [HumanMessage(content="first message")]}, config_1)
app.invoke({"messages": [HumanMessage(content="first message")]}, config_2)
app.invoke({"messages": [HumanMessage(content="second message")]}, config_2)

state = app.get_state(config_1)
print(f'messages remembered for thread 1:{len(state.values["messages"])}')

state = app.get_state(config_2)
print(f'messages remembered for thread 2:{len(state.values["messages"])}')

