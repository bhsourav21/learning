import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=os.getenv("OPENAI_API_KEY"))

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."

agent = create_agent(llm, tools=[get_weather])
result = agent.invoke({"messages": [HumanMessage(content="Weather in Paris?")]})
print(result["messages"][-1].content)



class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: State) -> State:
    model_with_tools = llm.bind_tools([get_weather])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode([get_weather]))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition) # -> "tools" or END, built in
graph.add_edge("tools", "agent")                      # loop back after running tools

app = graph.compile()
print(app.get_graph().draw_mermaid())

# result = app.invoke({"messages": [HumanMessage(content="What's the weather in Paris?")]})
result = app.invoke({"messages": [HumanMessage(content="What's the capital of France?")]})

for m in result["messages"]:
    print(type(m).__name__, "->", getattr(m, "content", None))
