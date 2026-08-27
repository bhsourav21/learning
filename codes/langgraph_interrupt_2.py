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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, api_key=os.getenv("OPENAI_API_KEY"))

MAX_ITERATIONS = 2

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_results: Annotated[list[str], operator.add]
    plan: str
    iteration_count: int
    final_answer: str

REQUIRED_FIELDS = ("messages", "tool_results", "plan", "iteration_count", "final_answer")

def check_state(node_name: str, state: State) -> None:
    """Verifies every node receives the full state shape, not a partial/stale one."""
    missing = [field for field in REQUIRED_FIELDS if field not in state]
    assert not missing, f"[{node_name}] missing required state fields: {missing}"

def make_plan(state: State) -> State:
    check_state("make_plan", state)
    question = state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=f"In one sentence, outline a plan to answer: {question}")])
    return {"plan": response.content}

def run_tool_step(state: State) -> State:
    check_state("run_tool_step", state)
    step_number = state["iteration_count"] + 1

    approved = interrupt(
        {"question": f"About to run research step {step_number}/{MAX_ITERATIONS} following plan '{state['plan']}'. Continue?"}
    )
    if not approved:
        return {"final_answer": "Stopped by human before research finished."}

    # Simulates a tool call's output -- kept local/deterministic so this demo
    # doesn't depend on a second external tool integration
    result = f"[step {step_number}] simulated finding for plan: {state['plan']}"
    return {
        "tool_results": [result],
        "iteration_count": step_number,
    }

def should_continue(state: State) -> Literal["run_tool_step", "finalize"]:
    if state.get("final_answer"):
        return "finalize"
    return "finalize" if state["iteration_count"] >= MAX_ITERATIONS else "run_tool_step"

def finalize(state: State) -> State:
    check_state("finalize", state)
    if state.get("final_answer"):
        return {}
    summary = "\n".join(state["tool_results"]) or "No research was gathered."
    return {"final_answer": f"Final answer based on:\n{summary}"}

graph = StateGraph(State)
graph.add_node("make_plan", make_plan)
graph.add_node("run_tool_step", run_tool_step)
graph.add_node("finalize", finalize)
graph.add_edge(START, "make_plan")
graph.add_edge("make_plan", "run_tool_step")
graph.add_conditional_edges(
    "run_tool_step", should_continue, {"run_tool_step": "run_tool_step", "finalize": "finalize"}
)
graph.add_edge("finalize", END)

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "agent-1"}}

initial_state = {
    "messages": [HumanMessage(content="What's the tallest mountain in the world?")],
    "tool_results": [],
    "plan": "",
    "iteration_count": 0,
    "final_answer": "",
}

print("=== run 1: start the agent, interrupt mid-run, resume from checkpoint (same process) ===")
out = app.invoke(initial_state, config)
while "__interrupt__" in out:
    print(f"paused -- {out['__interrupt__'][0].value['question']}")
    out = app.invoke(Command(resume=True), config)
print(f"finished: {out['final_answer']}\n")

print("=== run 2: interrupt again, then simulate a process restart before resuming ===")
out2 = app.invoke(initial_state, config)
print(f"paused -- {out2['__interrupt__'][0].value['question']}")

# A real process restart wipes Python's heap; a *new* InMemorySaver reproduces
# that exactly, since its checkpoint store is just an in-process dict. Same
# thread_id, same graph shape -- the only thing missing is the earlier process's memory.
print("-- process 'restarts' here --")
fresh_checkpointer = InMemorySaver()
fresh_app = graph.compile(checkpointer=fresh_checkpointer)

snapshot = fresh_app.get_state(config)
print(f"checkpoint found after restart: {bool(snapshot.values)} (values={snapshot.values})")

try:
    resumed = fresh_app.invoke(Command(resume=True), config)
    print(f"resume after restart succeeded: {resumed}")
except Exception as e:
    print(f"resume after restart FAILED as expected -- {type(e).__name__}: {e}")

print(
    "\nConclusion: InMemorySaver checkpoints live only in the process's RAM. "
    "A real restart loses them exactly like `fresh_checkpointer` above. "
    "Surviving a real restart needs a persistent checkpointer "
    "(e.g. SqliteSaver/PostgresSaver) pointed at the same DB file/connection."
)
