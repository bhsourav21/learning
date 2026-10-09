from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, add_messages
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

MODEL = "qwen3:4b"

# LangGraph needs a LangChain chat model, not the raw `openai` SDK client -
# .bind_tools() below is a LangChain method. ChatOpenAI speaks the same
# OpenAI-compatible protocol, so pointing base_url at Ollama is all it takes.
#
# max_tokens matters here: qwen3 is a reasoning model and spends its budget
# on hidden thinking FIRST. Too small a budget and the visible reply (and
# its tool calls) never arrive.
model = ChatOpenAI(
    model=MODEL,
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
    max_tokens=3000,
)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city"""
    return f"Its sunny in {city}."

TOOLS = [get_weather]

class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: State) -> State:
    model_with_tools = model.bind_tools(TOOLS)
    response = model_with_tools.invoke(state["messages"])
    # add_messages is a reducer: it APPENDS what you return to the existing
    # list rather than replacing it. Return a list for clarity.
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode([get_weather]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()
result = app.invoke({
    "messages": [HumanMessage(content="What is the weather in Paris?")]
})

for m in result["messages"]:
    print(type(m).__name__, "->", m.content)