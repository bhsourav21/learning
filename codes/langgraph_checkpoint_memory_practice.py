from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    text: Annotated[list, add_messages]

def node1(state: State) -> State:
    return {"text": [f"{state['text']}_1"]}

def node2(state: State) -> State:
    return {"text": [f"{state['text']}_2"]}

def node3(state: State) -> State:
    return {"text": [f"{state['text']}_3"]}

def node4(state: State) -> State:
    return {"text": [f"{state['text']}_4"]}

def node5(state: State) -> State:
    return {"text": [f"{state['text']}_5"]}

def node6(state: State) -> State:
    return {"text": [f"{state['text']}_6"]}

graph = StateGraph(State)
graph.add_node("node1", node1)
graph.add_node("node2", node2)
graph.add_node("node3", node3)
graph.add_node("node4", node4)
graph.add_node("node5", node5)
graph.add_node("node6", node6)

graph.add_edge(START, "node1")
graph.add_edge("node1", "node2")
graph.add_edge("node2", "node3")
graph.add_edge("node3", "node4")
graph.add_edge("node4", "node5")
graph.add_edge("node5", "node6")
graph.add_edge("node6", END)    

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

config1 = {
    "configurable": {
        "thread_id": "user_1"
    }
}

config2 = {
    "configurable": {
        "thread_id": "user_2"
    }
}

app.invoke({"text": "Hello"}, config1)
app.invoke({"text": "Hello"}, config2)

state1 = app.get_state(config1)
state2 = app.get_state(config2)

print(f"Messages remembered for user1: {state1.values["text"]}")
print(f"Messages remembered for user2: {state2.values["text"]}")