from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Annotated
import operator
from typing import Literal

class State(TypedDict):
    count: int
    log: Annotated[list[str], operator.add]
    # log: Annotated[list[str], add_messages]

def increment(state: State) -> State:
    return {"count": state["count"] + 1, "log": [f"count is now {state['count'] + 1}"]}

def should_continue(state: State) -> Literal["increment", "__end__"]:
    return "increment" if state["count"] < 3 else "__end__"

graph = StateGraph(State)
graph.add_node("increment", increment)
graph.add_edge(START, "increment")
graph.add_conditional_edges(
    "increment", should_continue, {"increment": "increment", "__end__": END}
)

app = graph.compile()
print(app.invoke({"count": 0, "log": []}))

