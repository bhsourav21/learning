import os
import ast
import operator
import json
import time

from openai import OpenAI

# Bare Ollama tag - no "ollama_chat/" prefix here. That prefix is a
# LiteLLM/DSPy routing hint; the OpenAI SDK passes this string straight
# through to Ollama, which 404s on anything but its own tag name.
#
# Must be a model with native tool-calling support. gemma3 has none and
# fails with "does not support tools" as soon as tools= is passed.
MODEL = "qwen3:4b"
MAX_STEPS = 6
MAX_LLM_RETRIES = 3

# Ollama exposes an OpenAI-compatible API at /v1, so the normal OpenAI
# client works unchanged - only base_url changes. api_key is unused by
# Ollama but the SDK rejects an empty one, so pass any placeholder.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
)

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

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "search the web for current facts(population, dates, statistics rtc.)",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
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
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            },
        },
    },
]

# No [ ] here. Adjacent string literals already join into ONE string;
# wrapping them in brackets makes a LIST, and a list as message content
# is what Ollama rejects with "invalid message format".
SYSTEM_PROMPT = (
    "You are an agent that can call tools (web_search, calculator) to answer "
    "multi-step questions. Break the tasks into steps and call one tool at a time. "
    "Once you have enough information, respond with the final answer "
    "in plain text and do not call any more tools."
)


def call_llm_with_retry(messages):
    "THINK step. Retries with backoff transient erros."
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
            # A 4xx (except 429) means WE sent something invalid - retrying
            # cannot help, so fail fast instead of sleeping 14 seconds.
            status = getattr(e, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise RuntimeError(f"LLM call rejected (not retryable): {e}") from e
            wait = 2 ** attempt
            print(f"  [retry] LLM call failed ({e}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {MAX_LLM_RETRIES} attempts: {last_err}")

def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    print(f"[perceive] {user_message}")

    for step in range(1, MAX_STEPS + 1):
        response = call_llm_with_retry(messages)
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            print(f"[think] Final answer: {msg.content}")
            return msg.content

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments
            print(f"[act] step {step}: {name}({raw_args})")
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                result = f"ERROR: could not parse arguments ({e}): {raw_args}"
            else:
                fn = TOOL_IMPLS.get(name)
                if fn is None:
                    result = f"ERROR: unknown tool '{name}'"
                else:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = f"ERROR: {e}"
            print(f"[observe] {str(result)[:300]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })

    raise RuntimeError(f"Agent did not converge within {MAX_STEPS} steps")

if __name__ == "__main__":
    tasks = [
        "What is the population of France divided by the population of Germany",
        "What is the compound population of Japan and South Korea?",
        "How many meters the Eifel Tower is taller than the Statue of Liberty?",
    ]

    for task in tasks:
        try:
            answer = run_agent(task)
            print(f"[FINAL] {answer}")
        except RuntimeError as e:
            print(f"[FAILED] {task} -> {e}")

