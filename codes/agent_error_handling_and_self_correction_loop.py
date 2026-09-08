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
MAX_VALIDATIONS = 2  # how many times a *finished* answer may be sent back for missing the question

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
    validations: int
    # Doubles as the pass/fail flag for the validation router: "" means the last checked
    # answer was accepted, anything else is the critique the agent has to act on. Same
    # idiom as last_error -- the routers stay pure and read the verdict off state.
    validation_feedback: str

REQUIRED_FIELDS = (
    "messages", "retry_count", "iterations", "last_error", "validations", "validation_feedback",
)

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

class Verdict(TypedDict):
    """Schema the checker is forced into, so the verdict arrives as a bool the router can
    branch on rather than prose someone has to string-match for the word 'yes'."""
    answers_question: Annotated[bool, ..., "True only if the draft answers the question that was asked, in full."]
    critique: Annotated[str, ..., "If it does not, what is missing or wrong. Addressed to the agent that wrote it. Empty if it does."]

validator = llm.with_structured_output(Verdict)

def validate(state: State) -> State:
    """Second opinion on a finished answer. The agent decides when it is *done*; this node
    decides whether it is *responsive*, in a separate LLM call that sees only the original
    question and the draft. Withholding the tool transcript is the point: a checker that
    watched three successful tool calls tends to accept whatever came after them, so it
    would rubber-stamp exactly the answers worth catching."""
    check_state("validate", state)
    # The original question, not the latest human turn -- a rejection message from a
    # previous round is also a HumanMessage, and the answer must be judged against what
    # was actually asked, not against the last critique.
    question = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    draft = state["messages"][-1].content

    verdict = validator.invoke(
        [
            SystemMessage(
                content=(
                    "You check whether a draft answer answers the question that was asked. Judge only "
                    "that: is every part of the question addressed, with the concrete detail it asked "
                    "for? Do not rewrite the answer, do not grade its style, and do not reward effort "
                    "-- a draft that narrates the work it did without stating the result does not "
                    "answer the question."
                )
            ),
            HumanMessage(content=f"QUESTION:\n{question}\n\nDRAFT ANSWER:\n{draft}"),
        ]
    )

    if verdict["answers_question"]:
        print("[validate] PASS")
        # Clearing the feedback is what tells route_after_validation to end the run, and it
        # also matters on a later pass: a stale critique here would end the run at give_up.
        return {"validation_feedback": ""}

    attempt = state["validations"] + 1
    print(f"[validate] FAIL (attempt {attempt}/{MAX_VALIDATIONS}) -- {verdict['critique']}")
    # The critique goes back as a HumanMessage rather than a SystemMessage: it follows an
    # AIMessage, which is the shape a chat-completions request expects, and the agent reads
    # it as the user pushing back -- which is what it is.
    return {
        "messages": [
            HumanMessage(
                content=(
                    f"ANSWER REJECTED (check {attempt} of {MAX_VALIDATIONS}): {verdict['critique']}\n"
                    "Answer the original question directly this time. Call the tools again if you need to."
                )
            )
        ],
        "validations": attempt,
        "validation_feedback": verdict["critique"],
    }

def route_after_agent(state: State) -> Literal["tools", "give_up", "validate"]:
    last_message = state["messages"][-1]
    # Both halves guard the same thing: only an AIMessage that actually asked for a tool
    # should reach tool_node. `.tool_calls` is an AIMessage-only attribute, so on a
    # HumanMessage or ToolMessage the bare attribute access would raise AttributeError --
    # the isinstance check short-circuits before that happens. And an AIMessage that just
    # answered in prose has tool_calls == [], which is falsy, so the plain-answer turn
    # routes to validate. Without the second half every agent turn would go to tool_node,
    # which would then iterate an empty list and hand the model back nothing.
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Two independent budgets, deliberately checked in two different places.
        # retry_count only moves when a tool *raises*, so it cannot see a run that loops
        # on successful calls -- that one would spin until GraphRecursionError. The turn
        # budget belongs here rather than in route_after_tools because this is the edge
        # every agent turn crosses, failing or not.
        if state["iterations"] >= MAX_ITERATIONS:
            return "give_up"
        return "tools"
    # Plain prose answer -- the model thinks it is finished. It does not get the last word:
    # validate re-reads the answer against the original question before the run can end.
    # No budget check here; validate's own router owns the exit from that loop.
    return "validate"

def route_after_tools(state: State) -> Literal["agent", "give_up"]:
    # Budget spent -- stop handing the model another chance to call the tool.
    return "give_up" if state["retry_count"] >= MAX_RETRIES else "agent"

def route_after_validation(state: State) -> Literal["agent", "give_up", "__end__"]:
    if not state["validation_feedback"]:
        return END
    # A third budget, for a third failure mode: the answer is wrong rather than the tool.
    # The turn budget is re-checked here because route_after_agent only tests it on the
    # tool-call branch -- an agent that keeps producing rejected prose never crosses that
    # branch, so without this the agent/validate loop would answer to MAX_VALIDATIONS alone.
    if state["validations"] >= MAX_VALIDATIONS or state["iterations"] >= MAX_ITERATIONS:
        return "give_up"
    return "agent"

def give_up(state: State) -> State:
    """Shared exit for all three budgets. Answers with the plain llm -- no tools bound -- so
    the model physically cannot emit another tool call, which is what guarantees the
    # graph terminates rather than looping on a broken or unsatisfiable dependency.

    Its answer goes straight to END, deliberately unvalidated: this node exists because a
    budget is already spent, and sending its output back to the checker would just build a
    second loop out of the escape hatch from the first one."""
    check_state("give_up", state)
    if state["retry_count"] >= MAX_RETRIES:
        reason = f"The tool failed {MAX_RETRIES} times; the last error was: {state['last_error']}."
        instruction = (
            "Stop trying to use the tool. Tell the user plainly that you could not complete the "
            "request, and say what went wrong."
        )
        print(f"[give_up] retry budget spent ({state['retry_count']}/{MAX_RETRIES})")
    elif state["validation_feedback"]:
        reason = (
            f"{state['validations']} of your answers were rejected as not answering the question; "
            f"the last critique was: {state['validation_feedback']}."
        )
        instruction = (
            "Write the best answer you can now, and state plainly which part of the question you "
            "were unable to answer."
        )
        print(f"[give_up] validation budget spent ({state['validations']}/{MAX_VALIDATIONS})")
    else:
        reason = (
            f"You have used all {MAX_ITERATIONS} allowed turns without reaching an answer. "
            "The tool calls succeeded, but they were not getting you closer to one."
        )
        instruction = (
            "Stop trying to use the tool. Tell the user plainly that you could not complete the "
            "request, and say what went wrong."
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
            SystemMessage(content=f"{reason} {instruction}")
        ]
    )
    return {"messages": cancellations + [response]}

graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_node("give_up", give_up)
graph.add_node("validate", validate)
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "give_up": "give_up", "validate": "validate"}
)
graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "give_up": "give_up"})
graph.add_conditional_edges(
    "validate", route_after_validation, {"agent": "agent", "give_up": "give_up", END: END}
)
graph.add_edge("give_up", END)

app = graph.compile()
print(app.get_graph().draw_mermaid())


def run(label: str, question: str, nudge: str | None = None) -> None:
    print(f"\n=== {label} ===")
    # `nudge` rigs the *agent* the way the tools above are rigged -- it is only there so one
    # flow can fail validation on demand. Real runs pass a question and nothing else.
    seed = ([SystemMessage(content=nudge)] if nudge else []) + [HumanMessage(content=question)]
    result = app.invoke(
        {
            "messages": seed,
            "retry_count": 0,
            "iterations": 0,
            "last_error": "",
            "validations": 0,
            "validation_feedback": "",
        }
    )
    print(
        f"-- transcript (retry_count={result['retry_count']} iterations={result['iterations']} "
        f"validations={result['validations']}) --"
    )
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


# Flow 4: nothing fails and the agent converges -- it just does not answer the question.
# The tool call succeeds, the agent stops, and the graph would have ended there before the
# validation node existed. The nudge makes the first answer omit the number deterministically;
# the checker sees a draft that describes a conversion without stating one, rejects it, and
# the critique goes back to the agent as a new turn.
run(
    "flow 4: tool succeeds, agent stops, validator sends the answer back",
    "Convert 40 US dollars to euros. What is the amount in euros?",
    nudge=(
        "RIGGED FOR THIS DEMO: your first final answer must say only that you performed the "
        "conversion and that the tool returned a result. Do not reveal the converted amount in any "
        "form -- not as digits, not spelled out in words, not rounded or approximated. If you are "
        "told your answer was rejected, then answer properly, with the number."
    ),
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
# A THIRD BUDGET: THE ANSWER ITSELF
#
# Both budgets above watch the *machinery* -- how often a tool threw, how many turns were
# spent. Neither looks at what the agent finally said, so flow 4 slips past both: one clean
# tool call, one confident paragraph, retry_count 0, iterations 2. The only thing that
# catches it is reading the answer back against the question, which is what validate does.
#
# Three things make it a check rather than a formality:
#
# 1. A separate LLM call, not a second opinion from the same conversation. The agent has
#    just spent several turns convincing itself; asking it "did you answer?" in the same
#    thread mostly gets "yes". The checker starts cold.
# 2. It sees the question and the draft, and not the tool transcript. Successful-looking
#    tool calls are precisely the thing that makes an unresponsive answer read as fine.
# 3. Structured output. answers_question is a bool the router branches on. A prose verdict
#    would have to be string-matched, and "no, this is not quite complete" contains "yes"
#    about as often as it does not.
#
# The loop it opens needs its own bound for the same reason the others do: an agent that
# cannot answer the question will not start being able to on the fourth try, so
# MAX_VALIDATIONS caps the rejections and route_after_validation re-checks the turn budget
# on the way past -- route_after_agent only tests iterations on its tool-call branch, so an
# agent bouncing between agent and validate never crosses the edge that would stop it.
#
# The obvious upgrade, once this shape is clear: give the validator a rubric, or run it on
# tool *results* rather than the final answer, so a wrong intermediate value is caught while
# there are still turns left to fix it.
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
