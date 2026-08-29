import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command, interrupt

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

MAX_ITERATIONS = 5

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_results: Annotated[list[str], operator.add]
    plan: str
    iteration_count: int
    final_answer: str

REQUIRED_FIELDS = ("messages", "tool_results", "plan", "iteration_count", "final_answer")

def check_state(node_name: str, state: State) -> None:
    """Verifies every node receives the full state shape, not a partial.stale one"""
    missing = [field for field in REQUIRED_FIELDS if field not in state]
    assert not missing, f"[{node_name}] missing required state fields: {missing}"

def make_plan(state: State) -> State:
    check_state("make_plan", state)
    question = state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=f"In one sentence, outline a plan to answer:{question}")])
    return {"plan": response.content}

def run_tool_step(state: State) -> State:
    check_state("run_tool_step", state)
    step_number = state["iteration_count"] + 1
    approved = interrupt(
        {
            "question": f"About to run research step {step_number}/{MAX_ITERATIONS} following '{state["plan"]}'. Continue?"
        }
    )
    print('hi1')
    if not approved:
        return {"final_answer": "Stopped by human before research finished."}

    # Simulates a tool call's output -- kept local/deterministic so this demo does not
    # depend on a second external tool integration
    result = f"[step {step_number}] simulated finding for plan: {state['plan']}"
    return {
        "tool_results": [result],
        "iteration_count": step_number
    }

def should_continue(state: State) -> Literal["run_tool_step", "finalize"]:
    if state.get("final_answer") or state["iteration_count"] >= MAX_ITERATIONS:
        return "finalize"
    else:
        return "run_tool_step"

def finalize(state: State) -> State:
    check_state("finalize", state)
    if state.get("final_answer"):
        return{}
    summary = "\n".join(state["tool_results"]) or "No research was gathered."
    return {"final_answer": f"Final answer based on:\n{summary}"}

graph = StateGraph(State)
graph.add_node("make_plan", make_plan)
graph.add_node("run_tool_step", run_tool_step)
graph.add_node("finalize", finalize)
graph.add_edge(START, "make_plan")
graph.add_edge("make_plan", "run_tool_step")
graph.add_conditional_edges(
    "run_tool_step", should_continue, {
        "run_tool_step": "run_tool_step",
        "finalize": "finalize"
    }
)
graph.add_edge("finalize", END)

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "agent_1"}}

initial_state = {
    "messages": [HumanMessage(content="What's the tallest mountain in the world?")],
    "tool_results": [],
    "plan": "",
    "iteration_count": 0,
    "final_answer": ""
}

print("=== run 1: start the agent, interrupt mid-run, resume from checkpoint (same process) ===")
out = app.invoke(initial_state, config)
while "__interrupt__" in out:
    print(f"paused -- {out['__interrupt__'][0].value['question']}")
    out = app.invoke(Command(resume=True), config)
print(f"finished: {out['final_answer']}\n")


print("=== run 2: interrupt again, then simulate a process restart before resuming ===")
out2 = app.invoke(initial_state, config)
print(f"out2['__interrupt__'][0]:{out2['__interrupt__'][0]}")
print(f"paused -- {out2['__interrupt__'][0].value['question']}")

# A real process restart wipes Python's heap; a *new* InMemorySaver reproduces
# that exactly, since its checkpoint store is just an inprocess dict. Same thread id,
# same graph shape -- the only missing is the earlier process memory.
print("-- process 'restarts' here --")
fresh_checkpointer = InMemorySaver()
fresh_app = graph.compile(checkpointer=fresh_checkpointer)

snapshot = fresh_app.get_state(config)
print(f"checkpoint found after restart: {bool(snapshot.values)} (values={snapshot.values}")

try:
    resumed = fresh_app.invoke(Command(resume=True), config)
    print(f"resume after restart succeeded: {resumed}")
except Exception as e:
    print(f"resume after restart FAILED as expected -- {type(e).__name__}: {e}")

print(
    "\nConclusion: InMemorySaver checkpoints live only in the processes RAM. "
    "A real restart loses them exactly like `fresh_checkpointer` above. "
    "Surviving a real restart needs a persistent checkpointer "
    "(e.g. SqliteSaver/PostgresSaver) pointed at the same DB file/connection."
)



# Question:
# check the output of @AI_study/learning/codes/langgraph_interrupt_3.py 
# why it says that [make_plan] missing required state fields: ['plan', 'iteration_count', 'final_answer']. 
# How does it find other two state fields - messages and tool_results?

# Answer:
# Confirmed empirically — the raw state make_plan receives after the simulated restart is exactly {'messages': [], 
# 'tool_results': []}. plan, iteration_count, final_answer aren't just empty, they're absent as keys entirely.

# Here's why: LangGraph builds one internal "channel" per state field, and which channel type it uses depends on 
# whether the field has a reducer.

# messages (Annotated[list, add_messages]) and tool_results (Annotated[list[str], operator.add]) have a reducer 
# attached. Reducer-backed channels (BinaryOperatorAggregate) need some starting value to combine future writes 
# into — so LangGraph seeds them at an empty list before anything is ever written. That's a real, defined value, 
# so it's present in state.
# plan, iteration_count, final_answer have no reducer — they're plain LastValue channels, which literally have 
# no value at all until the first write happens. There's no "empty string" or "zero" default; the channel is 
# just uninitialized.
# With fresh_checkpointer, thread "agent_1" has zero history — nothing was ever written to any channel. 
# Command(resume=True) finds no pending interrupt to resume, so it falls through to a fresh run from START. 
# But since no initial_state dict was passed either, LangGraph assembles make_plan's input purely from whatever 
# channels currently hold a value — the two reducer-backed ones (both still [], since even they were never written 
# to in this fresh checkpointer). The three LastValue fields simply don't exist as keys, which is exactly what 
# check_state catches.

# So the practical rule this surfaces: fields with a reducer always show up in state (even if empty); plain 
# scalar fields only exist once something writes to them. That's a real gap in this script too, worth knowing — 
# plan/iteration_count/final_answer have no safe default the way messages/tool_results do.

