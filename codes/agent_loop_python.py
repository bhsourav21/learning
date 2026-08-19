import ast
import json
import operator
import time
from openai import OpenAI
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()

MODEL = "gpt-4o-mini"
MAX_STEPS = 6
MAX_LLM_RETRIES = 3

client = OpenAI()

def web_search(query: str) -> str:
    """Real web search (DuckDuckGo, no api key required)"""
    try:
        # results = list(DDGS().news(
        #     query,
        #     max_results=3,
        #     timelimit="w",
        # ))
        results = list(DDGS().text(query, max_results=3))
    except Exception as e:
        raise RuntimeError(f"web_search failed: {e}")
    if not results:
        return "No results found"
    
    # for r in results:
    #     print(f"- {r['title']}: {r['body']}")
    #     print("-" * 100)
    
    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
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
    """Safely evaluate a pure arithmetic expression (np eval() of raw python)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except Exception as e:
        raise RuntimeError(f"calculator failed on expression: {expression}: {e}")
    return str(result)

TOOL_IMPLS = {
    "calculator": calculator,
    "web_search": web_search,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current facts (population, dates, statistics).",
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
            "description": "Evaluate a pure arithmetic expression and return the numeric data",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are an agent that can call tools (web_search, calculator) to answer "
    "multi-step question. Break the task into steps and call one tool at a time. "
    "Once you have enough information, respond with the final answer "
    "in plain text and do not call any more tools."
)

# Think step with retry logic
def call_llm_with_retry(messages):
    """THINK step. Retries with backoff on transient errors."""
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
            wait = 2 ** attempt # exponential backoff
            print(f"  [retry] LLM call failed ({e}: retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {MAX_LLM_RETRIES} attempts: {last_err}")

def run_agent(user_message: str) -> str:
    # --- PERCEIVE: read the user message, seed the conversation ---
    messages = [
        {
            "role": "system", "content": SYSTEM_PROMPT
        },
        {
            "role": "user", "content": user_message
        },
    ]
    
    print(f"[perceive] {user_message}")

    for step in range(1, MAX_STEPS + 1):
        # --- THINK: ask the LLM what to do next ---
        response = call_llm_with_retry(messages)
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        print()
        print(f"msg:{msg}")
        print()

        if not msg.tool_calls:
            print(f" [think] final answer: {msg.content}")
            return msg.content #normal exit point
        
        # --- ACT: parse and execute every requested tool call ---
        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                result = f"ERROR: could not parse arguments ({e}): {raw_args!r}"
            else:
                fn = TOOL_IMPLS.get(name)
                if fn is None:
                    result = f"ERROR: unknown tool '{name}'"
                else:
                    try:
                        result = fn(**args)

                        print()
                        print(f"result:{result}")
                        print()

                    except Exception as e:
                        result = f"ERROR: {e}" # tool failure doesn't crash the loop
            
            # --- OBSERVE: feed the tool result back into the conversation ---
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })
        # --- LOOP: go back to THINK with the updated message history
    
    raise RuntimeError(f"Agent did not converge within {MAX_STEPS} steps")
        

if __name__ == "__main__":
    tasks = [
        "What is the population of France divided bythe population of Germany?",
        "What is the combined population of Japan and South Korea?",
        "How many meters taller is the Eiffel Tower than the Statue of Liberty (incluing its pedestal/base)?",
    ]
    for task in tasks:
        try:
            answer = run_agent(task)
            print(f"[final] {answer}")
        except RuntimeError as e:
            print(f"[FAILED] {task} -> {e}")