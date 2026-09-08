import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

MAX_RETRIES = 3      # how many times a *failing* tool call may be retried
MAX_ITERATIONS = 15  # how many agent turns the run may take, however well things are going

# Deliberately vague docstring: the model has to *discover* the required pair format
# from the error message instead of reading it off the tool schema. That's what makes
# the first attempt fail and the retry meaningful.
@tool
def convert_currency(amount: float, pair: str) -> str:
    """Convert an amount between two currencies. Pass the currency pair as a single string."""
    print(f"[convert_currency] called with amount={amount!r} pair={pair!r}")
    rates = {"USD->EUR": 0.92, "USD->INR": 87.4, "EUR->USD": 1.09}
    if "->" not in pair:
        raise ValueError(
            f"unparsable pair {pair!r}: expected the exact format 'XXX->YYY' "
            "using 3-letter ISO codes, e.g. 'USD->EUR'"
        )
    if pair not in rates:
        raise KeyError(f"no rate for {pair!r}; supported pairs: {sorted(rates)}")
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

REQUIRED_FIELDS = ("messages", "retry_count", "iterations", "last_error")

def check_state(node_name: str, state: State) -> None:
    """Verifies every node receives the full state shape, not a partial/stale one"""
    missing = [field for field in REQUIRED_FIELDS if field not in state]
    assert not missing, f"[{node_name}] missing required state fields: {missing}"

def agent(state: State) -> State:
    check_state("agent", state)
    model_with_tools = llm.bind_tools(TOOLS)
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response], "iterations": state["iterations"] + 1}

def tool_node(state: State) -> State:
    """Runs the pending tool calls. This is the retry node: instead of letting an
    exception escape and kill the graph, it converts the exception into a ToolMessage
    the LLM can read, so the next agent turn sees *why* it failed and can try
    differently."""
    check_state("tool_node", state)
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
            failures += 1
            latest_error = f"{type(e).__name__}: {e}"
            attempt = state["retry_count"] + failures
            print(f"[tool_node] {name} FAILED (attempt {attempt}/{MAX_RETRIES}) -- {latest_error}")
            # The error goes back to the LLM as the tool's *result*. Every tool_call id
            # must get a ToolMessage back or the next chat-completions request is
            # rejected -- an error reply still satisfies that contract.
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
                    tool_call_id=call_id,
                    name=name,
                    status="error",
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
    # Both halves guard the same thing: only an AIMessage that actually asked for a tool
    # should reach tool_node. `.tool_calls` is an AIMessage-only attribute, so on a
    # HumanMessage or ToolMessage the bare attribute access would raise AttributeError --
    # the isinstance check short-circuits before that happens. And an AIMessage that just
    # answered in prose has tool_calls == [], which is falsy, so the plain-answer turn
    # routes to END. Without the second half every agent turn would go to tool_node, which
    # would then iterate an empty list and hand the model back nothing.
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Two independent budgets, deliberately checked in two different places.
        # retry_count only moves when a tool *raises*, so it cannot see a run that loops
        # on successful calls -- that one would spin until GraphRecursionError. The turn
        # budget belongs here rather than in route_after_tools because this is the edge
        # every agent turn crosses, failing or not.
        if state["iterations"] >= MAX_ITERATIONS:
            return "give_up"
        return "tools"
    # Plain prose answer -- the model is done, so no budget check is needed.
    return END

def route_after_tools(state: State) -> Literal["agent", "give_up"]:
    # Budget spent -- stop handing the model another chance to call the tool.
    return "give_up" if state["retry_count"] >= MAX_RETRIES else "agent"

def give_up(state: State) -> State:
    """Shared exit for both budgets. Answers with the plain llm -- no tools bound -- so
    the model physically cannot emit another tool call, which is what guarantees the
    graph terminates rather than looping on a broken or unsatisfiable dependency."""
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

    # Arriving here from route_after_agent means the turn budget tripped on an AIMessage
    # whose tool_calls were never executed -- tool_node was skipped entirely. Those ids
    # still need replies before the next request, or it 400s with "must be followed by
    # tool messages" (see note 1 at the bottom). Arriving from route_after_tools instead,
    # tool_node has already answered every id and this loop is a no-op.
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
graph.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "give_up": "give_up", END: END}
)
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


# Flow 1: recoverable failure. The vague docstring pushes the model toward something
# like pair='USD to EUR'; the ValueError text names the required 'XXX->YYY' format, and
# the next agent turn calls the tool again with corrected arguments.
run("flow 1: tool raises, LLM reads the error and retries differently", "Convert 250 US dollars to euros.")

# Flow 2: unrecoverable failure. Nothing the model rewrites can reach a dead service,
# so it burns all 3 retries and give_up produces the final answer instead.
run("flow 2: tool always raises, retry budget runs out", "How many units of SKU ABC-123 do we have in stock?")

# Flow 3: no failure at all, but no progress either. Every roll_dice call succeeds, so
# retry_count stays 0 and the retry budget never fires -- MAX_ITERATIONS is the only
# thing standing between this and a GraphRecursionError.
run(
    "flow 3: tool always succeeds but the agent never converges, turn budget runs out",
    "Roll the dice over and over until you get a 6. Do not stop or explain until you actually roll a 6.",
)


# TWO BUDGETS, TWO FAILURE MODES
#
# retry_count and iterations look redundant and are not. retry_count only moves inside
# the `except` branch of tool_node, so it measures one thing: how many tool calls threw.
# That is the right guard for a broken dependency (flow 2), and it is blind to flow 3,
# where every single call succeeds and the agent still never finishes. A stub agent that
# loops on a tool returning "pong" runs thousands of turns with retry_count stuck at 0
# and dies with GraphRecursionError -- a traceback, not an answer.
#
# iterations is the guard that has no opinion about success: it counts agent turns and
# stops the run at MAX_ITERATIONS however well things appear to be going. LangGraph's own
# recursion_limit is a backstop for the same case, but it *raises* rather than returning,
# which is precisely what give_up exists to avoid.
#
# The two live in different routers on purpose. retry_count is checked in
# route_after_tools, since only a tool call can change it. iterations is checked in
# route_after_agent, the edge every agent turn crosses whether or not a tool ran.
#
#
# Why a hand-written tool node instead of the prebuilt ToolNode?
#
# ToolNode(TOOLS) already catches exceptions and feeds the error back as a ToolMessage
# (that's its handle_tool_errors default), so the "show the LLM the error" half comes
# for free. What it does *not* do is count anything: the graph would bounce between
# agent and tools forever against a permanently broken tool, until the recursion limit
# fires and raises GraphRecursionError -- a crash, not an answer. The retry budget has
# to live in state, which means a node that writes retry_count, which means writing the
# node. Everything else here is the standard tools loop from langgraph_tools.py.
#
# Two details that are easy to get wrong:
#
# 1. Every tool_call id in the AIMessage needs a matching ToolMessage before the next
#    model call, including the failed ones. Dropping the failed call's reply -- or
#    swallowing the exception and returning nothing -- makes the *next* request 400 with
#    an "assistant message with tool_calls must be followed by tool messages" error, so
#    the tool failure resurfaces as an unrelated-looking API error.
#
#    The turn budget adds a second way to strand an id, and it is easy to miss: routing
#    agent -> give_up skips tool_node altogether, so the pending calls are never executed
#    and never answered. give_up therefore has to emit a CANCELLED ToolMessage per id
#    before it can call the model. Flow 2 never exposes this, because give_up is reached
#    from route_after_tools there, with every id already answered -- so the bug only
#    appears on the flow-3 path.
#
# 2. give_up calls the bare llm, not llm.bind_tools(TOOLS). Merely *asking* a model in a
#    system prompt to stop calling the tool is a request, not a constraint; removing the
#    tools from the request makes a fourth call impossible. (llm.bind_tools(TOOLS,
#    tool_choice="none") is the equivalent if you want the schemas still visible to the
#    model for context.)
