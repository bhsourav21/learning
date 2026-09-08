import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

MAX_RETRIES = 3
MAX_ITERATIONS = 15

@tool
def convert_currency(amount: float, pair: str) -> str:
    """Convert an amount between two currencies. Pass the currency pair as a single string."""
    print(f"[convert_currency] called with amount={amount!r} pair={pair!r}")
    rates = {"USD->EUR": 0.92, "USD->INR": 87.4, "EUR->USD": 1.09}
    if "->" not in pair:
        raise ValueError(
            f"unparsable pair {pair!r}: expected the exact format 'XXX->'YYY' "
            "using 3-LETTER ISO CODES, E.G. 'USD->EUR'"
        )
    if pair not in rates:
        raise KeyError(f"no rate for {pair!r}; supported pairs: {rates}")
    return f"{amount} {pair.split('->')[0]} = {round(amount * rates[pair], 2)} {pair.split('->')[1]}"

@tool
def check_inventory(sku: str) -> str:
    """Look up how many units of a SKU are in stock."""
    print(f"[check_inventory] called with sku={sku!r}")
    # Always fails -- stands in for a genuinely broken dependency (DB down, bad
    # credentials) that no amount of rewording the arguments can fix.
    raise RuntimeError("inventory service unreachable: connection refused on port 5432")

@tool
def roll_dice() -> int:
    """Roll a six-sided dice and return the result."""
    print("[roll_dice] rolled 3")
    # A loaded die: never raises, never returns 6. Asked to roll until a 6 comes up, the
    # agent loops forever on perfectly successful tool calls -- the exact failure mode
    # retry_count cannot see, because nothing ever throws.
    return 3

TOOLS = [convert_currency, check_inventory, roll_dice]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

class State(TypedDict):
    messages: Annotated[list, add_messages]
    retry_count: int
    iterations: int
    last_error: str

REQUIRED_FIELDS = ["messages", "retry_count", "iterations", "last_error"]

def check_state(node_name: str, state: State) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in state]
    assert not missing, f"{node_name} missing required state fields: {missing}"

def agent(state:State) -> State:
    check_state("agent", state)
    model_with_tools = llm.bind_tools(TOOLS)
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response], "iterations": state["iterations"] + 1}

def tool_node(state: State) -> State:
    check_state("agent", state)
    last_message = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    failures = 0
    latest_error = ""

    for call in last_message.tool_calls:
        name, args, call_id = call["name"], call["args"], call["id"]
        try:
            result = TOOLS_BY_NAME[name].invoke(args)
            print(f"[tool_node] {name} succeeded")
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id, name=name))
        except Exception as e:
            failures = failures + 1
            latest_error = f"{type(e).__name__}: {e}"
            attempt = state["retry_count"] + failures
            print(f"[tool_node] {name} FAILED (attempt {attempt}/{MAX_RETRIES}) -- {latest_error}")
            remaining = MAX_RETRIES - attempt
            retry_instruction = (
                f"You have {remaining} attempt(s) left. Do not give up yet and do not apologise to the "
                "user. Read the error, work out what shape the input should have been, and call the tool "
                "again now with different arguments."
                if remaining > 0
                else "No attempts left -- stop calling this tool."
            )
            tool_messages.append(
                ToolMessage(
                    content=(
                        f"TOOL CALL FAILED (attempt {attempt} of {MAX_RETRIES}) with {latest_error}\n"
                        f"{retry_instruction}"
                    ),
                    tool_call_id = call_id,
                    name = name,
                    status = 'error'
                )
            )

    # retry_count is monotonic across the whole run rather than reset on success:
    # a tool that alternates failing and succeeding would otherwise never hit the cap.
    return {
        "messages": tool_messages,
        "retry_count": state["retry_count"] + failures,
        "last_error": latest_error or state["last_error"],
    }

def route_after_agent(state: State) -> Literal["tools", "give_up", "__end__"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if state["iterations"] >= MAX_ITERATIONS:
            return "give_up"
        return "tools"
    return END

def route_after_tools(state: State) -> Literal["agent", "give_up"]:
    return "give_up" if state["retry_count"] >= MAX_RETRIES else "agent"

def give_up(state: State) -> State:
    check_state("give_up", state)
    if state["retry_count"] >= MAX_RETRIES:
        reason = f"The tool failed {MAX_RETRIES} times; the last error was: {state['last_error']}."
        print(f"[give_up] retry budget spent ({state['retry_count']}/{MAX_RETRIES})")
    else:
        reason = (
            f"You have used all {MAX_ITERATIONS} allowed turns without reaching an answer. "
            "The tool calls succeeded, but they were not getting you closer to one."
        )
        print(f"[give_up] turn budget spent ({state['iterations']}/{MAX_ITERATIONS})")

    last_message = state["messages"][-1]
    cancellations = []
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        cancellations = [
            ToolMessage(
                content="CANCELLED: the agent ran out of budget before this call was executed.",
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
            for call in last_message.tool_calls
        ]

    response = llm.invoke(
        state["messages"]
        + cancellations
        + [
            SystemMessage(
                content=(
                    f"{reason} Stop trying to use the tool. Tell the user plainly that you could "
                    "not complete the request, and say what went wrong."
                )
            )
        ]
    )
    return {"messages": cancellations + [response]}

graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_node("give_up", give_up)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "give_up": "give_up", END: END})
graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "give_up": "give_up"})
graph.add_edge("give_up", END)

app = graph.compile()
print(app.get_graph().draw_mermaid())

def run(label: str, question: str) -> None:
    print(f"\n=== {label} ===")
    result = app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "retry_count": 0,
            "iterations": 0,
            "last_error": "",
        }
    )
    print(f"-- transcript (retry_count={result['retry_count']} iterations={result['iterations']}) --")
    for m in result["messages"]:
        content = getattr(m, "content", None)
        calls = getattr(m, "tool_calls", None)
        suffix = f" tool_calls={[(c['name'], c['args']) for c in calls]}" if calls else ""
        print(f"{type(m).__name__} -> {content!r}{suffix}")


run("flow 1: tool raises, LLM reads the error and retries differently", "Convert 250 US dollars to euros.")
run("flow 2: tool always raises, retry budget runs out", "How many units of SKU ABC-123 do we have in stock?")
run(
    "flow 3: tool always succeeds but the agent never converges, turn budget runs out",
    "Roll the dice over and over until you get a 6. Do not stop or explain until you actually roll a 6.",
)