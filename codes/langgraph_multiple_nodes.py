from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    text: str

def shout(state: State) -> State:
    return {"text": state["text"].upper()}

def exclaim(state: State) -> State:
    return {"text": state["text"] + "!!!"}

graph = StateGraph(State)
graph.add_node("shout", shout)
graph.add_node("exclaim", exclaim)
graph.add_edge(START, "shout")
graph.add_edge("shout", "exclaim")
graph.add_edge("exclaim", END)

app = graph.compile()
print(app.invoke({"text": "i am in lower case"}))

