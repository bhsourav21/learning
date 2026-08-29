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

EVENT_STYLE = {
    "think": (Fore.CYAN, "THOUGHT"),
    "act": (Fore.YELLOW, "TOOL CALL"),
    "finalize": (Fore.GREEN, "FINAL ANSWER"),
}

print("=== streaming a 5-step think -> act loop, one event per node update ===\n")
for event in app.stream(initial_state, stream_mode="updates"):
    for node_name, output in event.items():
        color, label = EVENT_STYLE[node_name]
        if node_name == "think":
            text = output["messages"][-1].content
        elif node_name == "act":
            text = output["tool_results"][-1]
        else:
            text = output["final_answer"]
        print(f"{color}{Style.BRIGHT}[{label}]{Style.RESET_ALL} {color}{text}")

# Reflection (2 sentences):
# Streaming turns a multi-second black box into a running commentary, so the user can tell
# the agent is making progress -- and on what -- instead of staring at a blank terminal
# wondering if it's stuck. That visibility also lets a human interrupt or redirect early
# (e.g. spot a bad plan at step 2) rather than waiting for the whole task to finish.

# System-prompt iteration note:
# The first version of THINK_SYSTEM_PROMPT just said "output the next reasoning step,
# starting with Thought:" -- no length limit, no instruction to hold off on the answer.
# Streamed output showed two incoherence problems: step 1's "Thought" gave away the full
# final answer immediately ("The tallest mountain ... is Mount Everest ... 8,848.86 meters")
# instead of planning research, and every step was a multi-sentence paragraph instead of one
# reasoning step -- both defeat the point of watching a 5-step process unfold. Adding "output
# exactly one sentence", "name a SPECIFIC sub-question not covered by an earlier Thought",
# and "never answer the original question here" fixed both: the 5 streamed Thoughts became
# one-sentence, non-overlapping angles (geography, geology, climate, cultural history,
# climbing risks), with the actual answer only appearing in the FINAL ANSWER step.
