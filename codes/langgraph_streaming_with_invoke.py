import operator
import os
from typing import Annotated, Literal, TypedDict

from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_openai import ChatOpenAI

load_dotenv()
colorama_init(autoreset=True)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

MAX_ITERATIONS = 5

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_results: Annotated[list[str], operator.add]
    iteration_count: int
    final_answer: str

THINK_SYSTEM_PROMPT = (
    "You are a research agent answering the user's question through a fixed sequence "
    f"of {MAX_ITERATIONS} investigation steps, one step per turn. You will see your own "
    "past 'Thought:' lines and the 'Observation:' that followed each one.\n"
    "Rules for this turn:\n"
    "- Output exactly one new sentence, starting with 'Thought:'.\n"
    "- It must name a SPECIFIC sub-question or angle you have not investigated in an earlier "
    "Thought yet (check the history before choosing).\n"
    "- Never restate a previous Thought and never answer the original question here -- a "
    "separate final step handles that."
)

FINAL_SYSTEM_PROMPT = (
    "You are the same research agent. All investigation steps are complete. Using only the "
    "Thought/Observation history in this conversation, write the final answer to the "
    "original question. Start with 'Final Answer:' and keep it to 2-3 sentences."
)

def think(state: State) -> State:
    response = llm.invoke([SystemMessage(content=THINK_SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [AIMessage(content=response.content.strip())]}

def act(state: State) -> State:
    step_number = state["iteration_count"] + 1
    last_thought = state["messages"][-1].content
    # Simulates a tool call's output -- kept local/deterministic so this demo
    # doesn't depend on a second external tool integration.
    result = f"Observation: [step {step_number}] simulated finding for -- {last_thought}"
    return {
        "messages": [HumanMessage(content=result)],
        "tool_results": [result],
        "iteration_count": step_number,
    }

def should_continue(state: State) -> Literal["think", "finalize"]:
    return "finalize" if state["iteration_count"] >= MAX_ITERATIONS else "think"

def finalize(state: State) -> State:
    response = llm.invoke([SystemMessage(content=FINAL_SYSTEM_PROMPT), *state["messages"]])
    return {"final_answer": response.content.strip(), "messages": [AIMessage(content=response.content.strip())]}

graph = StateGraph(State)
graph.add_node("think", think)
graph.add_node("act", act)
graph.add_node("finalize", finalize)
graph.add_edge(START, "think")
graph.add_edge("think", "act")
graph.add_conditional_edges("act", should_continue, {"think": "think", "finalize": "finalize"})
graph.add_edge("finalize", END)

app = graph.compile()

initial_state = {
    "messages": [HumanMessage(content="What's the tallest mountain in the world?")],
    "tool_results": [],
    "iteration_count": 0,
    "final_answer": "",
}

print("=== invoke: nothing prints here -- think/act/think/act/.../finalize all run silently ===")
print("=== (this call blocks until the ENTIRE graph -- all 5 steps -- has finished) ===\n")
final_state = app.invoke(initial_state)
print("=== invoke returned -- only NOW do we have anything to show, and it's all at once ===\n")

# There is no per-node signal here, so the only way to tell Thought/Tool/Final apart is to
# walk the final message history and infer it from shape: HumanMessage after the first one
# is an Observation, AIMessage before iteration_count hits MAX is a Thought, the very last
# AIMessage is the Final Answer. Compare to langgraph_streaming.py, where the node name
# itself ("think"/"act"/"finalize") tells you the event type with zero guessing.
for i, message in enumerate(final_state["messages"][1:], start=1):
    if isinstance(message, HumanMessage):
        print(f"{Fore.YELLOW}{Style.BRIGHT}[TOOL CALL]{Style.RESET_ALL} {Fore.YELLOW}{message.content}")
    elif message.content == final_state["final_answer"]:
        print(f"{Fore.GREEN}{Style.BRIGHT}[FINAL ANSWER]{Style.RESET_ALL} {Fore.GREEN}{message.content}")
    else:
        print(f"{Fore.CYAN}{Style.BRIGHT}[THOUGHT]{Style.RESET_ALL} {Fore.CYAN}{message.content}")

# Why this can't show progress live, unlike langgraph_streaming.py's app.stream(...):
# invoke() runs the whole graph -- think, act, think, act, ..., finalize -- inside one
# blocking call and only returns after the LAST node finishes. All the printing above happens
# in a tight loop after the fact, replaying history from the final state; during the actual
# run (which can take several seconds per LLM call x5) the terminal shows nothing at all, so
# there's no way to watch a specific step happen or interrupt mid-way based on what step 2
# said.
