"""
agent_loop.py

A manual "agent loop" implemented from scratch, in pure Python, with no
agent framework (no LangChain, no LlamaIndex, no SDK agent runner).

Loop shape:

    PERCEIVE -> THINK (call LLM) -> ACT (call tool) -> OBSERVE (read result) -> loop

Requirements:
    pip install openai ddgs

Setup:
    Paste your OpenAI API key into OPENAI_API_KEY below, OR set it as an
    environment variable before running:
        export OPENAI_API_KEY="sk-..."

Run:
    python agent_loop.py
"""

import os
import ast
import json
import operator
import time

from openai import OpenAI

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------

OPENAI_API_KEY = ""  # <-- paste your OpenAI API key here (or use the env var)
MODEL = "gpt-4o-mini"
MAX_STEPS = 6         # hard cap on think/act/observe iterations per task (loop-runaway guard)
MAX_LLM_RETRIES = 3   # retries for transient LLM API errors (rate limit, timeout, 5xx)

client = OpenAI(api_key=OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY"))

# ----------------------------------------------------------------------
# 2. TOOLS
#    These are the functions the "Act" step is allowed to call. Each tool
#    is plain Python -- the LLM never executes code, it only ever returns
#    a tool *name* and a JSON *arguments* string, which we parse and run.
# ----------------------------------------------------------------------

def web_search(query: str) -> str:
    """Real web search (DuckDuckGo, no API key required)."""
    from ddgs import DDGS
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as e:
        # Network/tool failure -> raise, caller turns this into an
        # observation ("ERROR: ...") instead of crashing the whole loop.
        raise RuntimeError(f"web_search failed: {e}")
    if not results:
        return "No results found."
    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)


# A small, *safe* arithmetic evaluator. We deliberately do NOT use eval(),
# since that would let a hallucinated or adversarial expression run
# arbitrary Python. Only numeric literals and +-*/** are allowed.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"disallowed expression node: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Safely evaluate a pure arithmetic expression, e.g. '68170000 / 84270000'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except Exception as e:
        raise RuntimeError(f"calculator failed on '{expression}': {e}")
    return str(result)


TOOL_IMPLS = {
    "web_search": web_search,
    "calculator": calculator,
}

# JSON-schema tool declarations, in the format the OpenAI API expects.
# This is the "menu" of actions we tell the LLM it is allowed to pick from.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current facts, such as population figures, dates, heights, or other statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a pure arithmetic expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '68170000 / 84270000'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are an agent that can call tools (web_search, calculator) to answer "
    "multi-step questions. Break the task into steps and call one tool at a "
    "time. Once you have enough information, respond with the final answer "
    "in plain text and do not call any more tools."
)

# ----------------------------------------------------------------------
# 3. THINK: call the LLM, with retry logic for transient failures
# ----------------------------------------------------------------------

def call_llm_with_retry(messages):
    """
    THINK step. Calls the LLM and returns its response.
    Retries with exponential backoff on transient errors (rate limits,
    timeouts, 5xx). Does NOT retry on things like an invalid API key --
    those raise immediately since retrying won't help.
    """
    last_err = None
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [retry] LLM call failed ({e}); retrying in {wait}s "
                  f"(attempt {attempt}/{MAX_LLM_RETRIES})...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {MAX_LLM_RETRIES} attempts: {last_err}")


# ----------------------------------------------------------------------
# 4. THE MANUAL AGENT LOOP
#    perceive -> think -> act -> observe -> loop
# ----------------------------------------------------------------------

def run_agent(user_message: str) -> str:
    # --- PERCEIVE: read the user message, seed the conversation ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    print(f"\n=== TASK: {user_message} ===")
    print(f"[perceive] {user_message}")

    for step in range(1, MAX_STEPS + 1):
        # --- THINK: ask the LLM what to do next ---
        response = call_llm_with_retry(messages)
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # No tool call requested -> the LLM is giving its final answer.
            # This is the loop's only normal exit point.
            print(f"[think]  final answer: {msg.content}")
            return msg.content

        # --- ACT: parse and execute every requested tool call ---
        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                # The LLM produced malformed JSON arguments. Don't crash --
                # feed the parse error back as the observation so the LLM
                # can see its mistake and try again next turn.
                result = f"ERROR: could not parse arguments ({e}): {raw_args!r}"
                print(f"[act]    {name}({raw_args!r}) -> {result}")
            else:
                print(f"[think]  decided to call: {name}({args})")
                fn = TOOL_IMPLS.get(name)
                if fn is None:
                    # The LLM asked for a tool that doesn't exist.
                    result = f"ERROR: unknown tool '{name}'"
                else:
                    try:
                        result = fn(**args)
                    except TypeError as e:
                        # Wrong/missing arguments for the tool's signature.
                        result = f"ERROR: bad arguments for {name}: {e}"
                    except Exception as e:
                        # Tool raised (network error, bad expression, etc).
                        # Caught here so ONE failing tool call can't kill
                        # the whole agent loop.
                        result = f"ERROR: {e}"
                print(f"[act]    {name}({args}) -> {result}")

            # --- OBSERVE: feed the tool result back into the conversation ---
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
            print("[observe] recorded result, looping back to THINK")

        # --- LOOP: go back to THINK with the updated message history ---

    # Safety valve: if we get here, the agent never produced a final
    # answer within MAX_STEPS turns (e.g. it kept calling tools forever).
    raise RuntimeError(f"Agent did not converge within {MAX_STEPS} steps")


# ----------------------------------------------------------------------
# 5. RUN THREE MULTI-STEP TASKS END TO END
# ----------------------------------------------------------------------

if __name__ == "__main__":
    tasks = [
        "What is the population of France divided by the population of Germany?",
        "What is the combined population of Japan and South Korea?",
        "How many meters taller is the Eiffel Tower than the Statue of Liberty (including its pedestal/base)?",
    ]

    for task in tasks:
        try:
            answer = run_agent(task)
            print(f"[FINAL] {answer}\n{'-' * 70}")
        except RuntimeError as e:
            # Top-level guard: one task's failure shouldn't stop the rest.
            print(f"[FAILED] {task} -> {e}\n{'-' * 70}")
