import os
from typing import Annotated, TypedDict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=os.getenv("OPENAI_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Real web search (DuckDuckGo, no api key required)"""
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as e:
        raise RuntimeError(f"web_search failed: {e}")

    print(f"[web_search] live DuckDuckGo results for '{query}':")
    if not results:
        print("  (no results)")
        return "No results found"

    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def answer(state: State) -> State:
    model_with_tools = llm.bind_tools([web_search])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("answer", answer)
graph.add_node("tools", ToolNode([web_search]))
graph.add_edge(START, "answer")
graph.add_conditional_edges("answer", tools_condition) # -> "tools" or END, built in
graph.add_edge("tools", "answer")                       # loop back after running tools

app = graph.compile()
print(app.get_graph().draw_mermaid())

result = app.invoke({"messages": [HumanMessage(content="Who won the Wimbledon in 2026?")]})

for m in result["messages"]:
    print(type(m).__name__, "->", getattr(m, "content", None))
