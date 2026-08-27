import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=os.getenv("OPENAI_API_KEY"))

ROLE_TO_MESSAGE = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}

class State(TypedDict):
    role: str
    question: str
    answer: str

def question(state: State) -> State:
    return {
        "role": "user",
        "question": state["question"]
    }

def answer(state: State) -> State:
    message_cls = ROLE_TO_MESSAGE[state["role"]]
    response = llm.invoke([message_cls(content=state["question"])])
    return {"answer": response.content}


graph= StateGraph(State)
graph.add_node("question", question)
graph.add_node("answer", answer)
graph.add_edge(START, "question")
graph.add_edge("question", "answer")
graph.add_edge("answer", END)

app = graph.compile()
result = app.invoke({"question": "What's the capital of India?"})
print(result)
