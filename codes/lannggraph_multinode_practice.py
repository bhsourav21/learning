from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    text: str

def upper(state: State) -> State:
    return {"text": f"{state['text'].upper()}"}

def exclaim(state: State) -> State:
    return {"text": f"{state['text']}!!"}


graph = StateGraph(State)
graph.add_node("upper", upper)
graph.add_node("exclaim", exclaim)
graph.add_edge(START, "upper")
graph.add_edge("upper", "exclaim")
graph.add_edge("exclaim", END)

app = graph.compile()
print(app.invoke({"text": "my name is sourav bhattacharya"}))