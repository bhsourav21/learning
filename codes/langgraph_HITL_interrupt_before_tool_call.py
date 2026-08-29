import operator
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, add_messages

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
    """Verifies every node receives the full state shape, not a partial/stale one"""
    missing = [field for field in REQUIRED_FIELDS if field not in state]
    assert not missing, f"[{node_name}] missing required state fields: {missing}"

def make_plan(state: State) -> State:
    check_state("make_plan", state)
    question = state["messages"][-1].content
    response = llm.invoke([HumanMessage(content=f"In one sentence, outline a plan to answer:{question}")])
    return {"plan": response.content}

def tool_node(state: State) -> State:
    """The actual tool execution step. No interrupt() call here -- the pause is
    handled entirely by interrupt_before=['tool_node'] at graph-compile time."""
    print("tool_node is executing!")  # should never print during the invoke() that pauses -- only after a resume
    check_state("tool_node", state)
    step_number = state["iteration_count"] + 1

    # Simulates a tool call's output -- kept local/deterministic so this demo does not
    # depend on a second external tool integration
    result = f"[step {step_number}] simulated finding for plan: {state['plan']}"
    return {
        "tool_results": [result],
        "iteration_count": step_number
    }

def should_continue(state: State) -> Literal["tool_node", "finalize"]:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "finalize"
    else:
        return "tool_node"

def finalize(state: State) -> State:
    check_state("finalize", state)
    summary = "\n".join(state["tool_results"]) or "No research was gathered."
    return {"final_answer": f"Final answer based on:\n{summary}"}

graph = StateGraph(State)
graph.add_node("make_plan", make_plan)
graph.add_node("tool_node", tool_node)
graph.add_node("finalize", finalize)
graph.add_edge(START, "make_plan")
graph.add_edge("make_plan", "tool_node")
graph.add_conditional_edges(
    "tool_node", should_continue, {
        "tool_node": "tool_node",
        "finalize": "finalize"
    }
)
graph.add_edge("finalize", END)

checkpointer = InMemorySaver()
# Static interrupt: the graph halts before entering tool_node every single time it's
# about to run -- unconditionally, with no interrupt() call needed inside the node.
app = graph.compile(checkpointer=checkpointer, interrupt_before=["tool_node"])


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("please answer y or n")


def propose_different_plan(question: str, rejected_plan: str) -> str:
    response = llm.invoke([
        HumanMessage(
            content=(
                f"A human reviewer rejected this plan: '{rejected_plan}' for answering: "
                f"{question}. In one sentence, propose a different approach."
            )
        )
    ])
    return response.content


def run_with_console_approval(initial_state: State, config: dict) -> State:
    out = app.invoke(initial_state, config)
    snapshot = app.get_state(config)

    print(f"snapshot:{snapshot}")
    # Static interrupt_before has no "__interrupt__" payload to check (that's the
    # dynamic interrupt() pattern from langgraph_interrupt_3.py) -- we check what node
    # is pending via get_state(config).next instead.
    while snapshot.next == ("tool_node",):
        step_number = snapshot.values["iteration_count"] + 1
        plan = snapshot.values["plan"]
        print(f"paused -- about to run research step {step_number}/{MAX_ITERATIONS} following '{plan}'.")

        if ask_yes_no("Approve this tool step?"):
            print("approved -> resuming with invoke(None, config)")
        else:
            question = snapshot.values["messages"][-1].content
            new_plan = propose_different_plan(question, plan)
            print(f"rejected -> asked the LLM for a different approach: '{new_plan}'")
            # Rewrite the plan in the checkpoint before resuming, so tool_node runs
            # against the revised approach instead of the rejected one.
            app.update_state(config, {"plan": new_plan})
            print("resuming with invoke(None, config)")

        out = app.invoke(None, config) # None -> don't inject any new input, just continue from wherever the checkpoint left off.
        snapshot = app.get_state(config)

    return out


print("=== flow 1: pause -> approve -> resume ===")
print(f"You'll be asked to approve {MAX_ITERATIONS} research steps. Answer 'y' each time to walk the approve-resume path.\n")
config1 = {"configurable": {"thread_id": "agent_1"}}
initial_state1 = {
    "messages": [HumanMessage(content="What's the tallest mountain in the world?")],
    "tool_results": [],
    "plan": "",
    "iteration_count": 0,
    "final_answer": ""
}
final1 = run_with_console_approval(initial_state1, config1)
print(f"finished: {final1['final_answer']}\n")


# 
# Question:
# langgraph_interrupt_3.py pauses with a dynamic interrupt() call inside run_tool_step,
# and resumes with Command(resume=True). Why doesn't this script use interrupt()/Command?
#
# Answer:
# interrupt_before is a *static*, graph-level pause -- LangGraph stops right before the
# named node runs, unconditionally, every time it's about to execute. Nothing inside
# tool_node calls interrupt(); there is no payload for a caller to read out of
# out["__interrupt__"], and no resume *value* the node is waiting on. So checking for a
# pause means reading get_state(config).next instead of "__interrupt__" in out, and
# resuming is just invoke(None, config) -- no Command wrapper needed.
#
# Rejection also works differently: since tool_node hasn't run yet, there's no tool
# output to override the way Command(resume=...) overrides an interrupt()'s return
# value. Instead we mutate the pending input directly with update_state(config, {"plan":
# new_plan}) before resuming, so when tool_node does run, it runs against the revised
# plan rather than the rejected one.
print("=== flow 2: pause -> reject -> retry with a different approach ===")
print(f"You'll be asked to approve {MAX_ITERATIONS} research steps. Answer 'n' at least once to see the LLM propose a different plan and retry.\n")
config2 = {"configurable": {"thread_id": "agent_2"}}
initial_state2 = {
    "messages": [HumanMessage(content="What's the deepest point in the ocean?")],
    "tool_results": [],
    "plan": "",
    "iteration_count": 0,
    "final_answer": ""
}
final2 = run_with_console_approval(initial_state2, config2)
print(f"finished: {final2['final_answer']}")

# Important concept:
# interrupt_before doesn't pause the program — it only pauses graph execution by returning early from 
# app.invoke(). LangGraph is a synchronous library call, not a background process that blocks waiting 
# for you to answer. It has no way to freeze your Python script and wait — the only thing it can do is:

# Run up to (but not including) tool_node.
# Persist that partial state to the checkpointer.
# Return control back to whatever called invoke() — normally, like any other function return.
# What happens after that return is entirely up to your code. If your code doesn't check 
# get_state(config).next and doesn't itself block on something (like input()), Python just 
# keeps executing statement by statement — run_with_console_approval returns out, finished:  
# gets printed with the stale final_answer, and the script moves on to flow 2, then exits. 
# That's not the interrupt failing — it's your code choosing not to react to it.

# The actual "pause and wait for a human" behavior was never LangGraph's job — it was the 
# while snapshot.next == ("tool_node",): ... ask_yes_no(...) ... app.invoke(None, config) loop 
# you commented out. input() is what blocks the terminal waiting for you to type 
# something; app.invoke(None, config) is what tells LangGraph "okay, proceed past 
# the checkpoint now." Take those out, and there's nothing left to create the appearance of a pause, 
# even though the underlying graph truly did stop before tool_node 
# (which the printed snapshot.next=('tool_node',) still proves).

# So: to restore the intended behavior, uncomment lines 112–129. The interrupt 
# itself was never broken — the code responsible for acting on it was.




# WHEN TO USE DYNAMIC INTERRUPT VS INTERRUPT_BEFORE

# the rule of thumb: use dynamic interrupt() when the node's whole job is 
# "pause and get a decision" and you're writing that node yourself 
# (like request_approval, or run_tool_step where you need the pause interleaved 
# with your own computed values). Use interrupt_before when the thing you want 
# to gate is a prebuilt or reusable component (like ToolNode) and the
#  approval requirement is a policy about when a node runs, not something 
# that belongs inside the node's own implementation.

