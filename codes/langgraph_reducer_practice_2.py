
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int
    log: Annotated[list[str], operator.add]
    # log: list[str]

def increment(state: State) -> State:
    return {
        "count": state["count"] + 1,
        "log": [f"Count is now {state["count"] + 1}"]
    }

def should_continue(state: State) -> Literal["continue", "end"]:
    if state["count"] < 5:
        return "continue"
    else:
        return "end"

graph = StateGraph(State)
graph.add_node("increment", increment)
graph.add_edge(START, "increment")
graph.add_conditional_edges(
    "increment", 
    should_continue,
    {
        "continue": "increment",
        "end": END
    }
)

app = graph.compile()
print(app.invoke({"count": 0, "log": []}))