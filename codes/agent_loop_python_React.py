import re
import ast
import operator
from openai import OpenAI
from dotenv import load_dotenv
from ddgs import DDGS
import time

load_dotenv()

MODEL = "gpt-4o-mini"
MAX_STEPS = 6
MAX_LLM_RETRIES = 3

client = OpenAI()

def web_search(query: str) -> str:
    """Real web search (DuckDuckGo, no api key required)"""
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as e:
        raise RuntimeError(f"web_search failed: {e}")

    print(f"[web_search] live DuckDuckGo results for '{query}':")
    if not results:
        print("  (no results)")
        return "No results found"

    for r in results:
        print(f"  - {r['title']}: {r['body']}")

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
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Disallowed expression: {ast.dump(node)}")

def calculator(expr: str) -> str:
    """A restricted arithmetic evaluator (no eval() / no arbitrary code exec)."""
    try:
        tree = ast.parse(expr, mode="eval")
        return str(_safe_eval(tree.body if isinstance(tree, ast.Expression) else tree))
    except Exception as e:
        return f"Error evaluating '{expr}': {e}"

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

SYSTEM_PROMPT = """You are a reasoning agent that solves questions step by step.
At every turn you must output exactly two lines in this format:

Thought: <your reasoning about what you know so far and what to do next>
Action: <tool_name>[<tool_input>]

Available tools:
    web_search[query]           - look up a fact
    calculator[expr]            - evaluate an arithmetic expression
    finish[answer]              - give the final answer and stop

After you emit an action, the environment will run it and reply with:
Observation: <result>

Contnue the Thought/Action/Observation cycle until you call finish[]."""

ACT_ONLY_SYSTEM_PROMPT = """You are an agent that solves questions by calling tools.
At every turn you must output exactly one line in this format.

Action: <tool_name>[<tool_input>]

Do not output any reasoning or explanation, only the Action line.

Available tools:
    web_search[query]           - look up a fact
    calculator[expr]            - evaluate an arithmetic expression
    finish[answer]              - give the final answer and stop

After you emit an Action, the environment will run it and reply with:
Observation: <result>

Continue calling actions until you call finish[]."""

# ---------------------------------------------------------------------------
# 2. Regex parsing of the model's raw text output
# ---------------------------------------------------------------------------

REACT_STEP_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*\nAction:\s*(?P<tool>\w+)\[(?P<arg>.*?)\]\s*$",
    re.DOTALL,
)

ACT_ONLY_STEP_RE = re.compile(
    r"Action:\s*(?P<tool>\w+)\[(?P<arg>.*?)\]\s*$",
    re.DOTALL,
)

def parse_react_step(raw_text: str):
    m = REACT_STEP_RE.search(raw_text.strip())
    if not m:
        raise ValueError(f"Could not parse ReAct step from model output:\n{raw_text!r}")
    return m.group("thought").strip(), m.group("tool"), m.group("arg").strip()


def parse_act_only_step(raw_text: str):
    m = ACT_ONLY_STEP_RE.search(raw_text.strip())
    if not m:
        raise ValueError(f"Could not parse Action-only step from model output:\n{raw_text!r}")
    return m.group("tool"), m.group("arg").strip()

def real_llm_call(messages):
    """THOUGHT step. Retries with backoff on transient errors."""
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
            print(f" [retry] LLM call failed {e}: retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {MAX_LLM_RETRIES} attempts: {last_err}")

def run_react_agent(question: str, max_steps: int = 6):
    print(f"=== ReAct agent ===\nQuestion: {question}\n")
    history = []

    messages = [
        {
            "role": "system", "content": SYSTEM_PROMPT
        },
        {
            "role": "user", "content": question
        },         
    ]
    llm_response = real_llm_call(messages)
    llm_response_text = llm_response.choices[0].message.content

    for step_num in range(1, max_steps + 1):
        thought, tool, arg = parse_react_step(llm_response_text)

        print(f"Step {step_num}")
        print(f"  Thought: {thought}")
        print(f"  Action: {tool}[{arg}]")

        if tool == "finish":
            print(f". >>> Final answer: {arg}\n")
            history.append(llm_response_text)
            return arg

        if tool not in TOOL_IMPLS:
            observation = f"Error: unknown tool '{tool}'"
        else:
            observation = TOOL_IMPLS[tool](arg)

        print(f" Observation: {observation}\n")
        history.append(llm_response_text + f"\nObservation: {observation}")

        messages.append({"role": "assistant", "content": llm_response_text})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        llm_response = real_llm_call(messages)
        llm_response_text = llm_response.choices[0].message.content

    raise RuntimeError("ReAct agent did not finish with max_steps")

def run_act_only_agent(question: str, max_steps: int = 6):
    print(f"=== Act-only agent (direct tool calling, no though step) ===\nQuestion: {question}\n")
    history = []

    messages = [
        {
            "role": "system", "content": SYSTEM_PROMPT
        },
        {
            "role": "user", "content": question
        },         
    ]
    llm_response = real_llm_call(messages)
    llm_response_text = llm_response.choices[0].message.content

    for step_num in range(1, max_steps + 1):
        tool, arg = parse_act_only_step(llm_response_text)

        print(f"Step {step_num}")
        print(f"  Action: {tool}[{arg}]")

        if tool == "finish":
            print(f". >>> Final answer: {arg}\n")
            history.append(llm_response_text)
            return arg

        if tool not in TOOL_IMPLS:
            observation = f"Error: unknown tool '{tool}'"
        else:
            observation = TOOL_IMPLS[tool](arg)

        print(f" Observation: {observation}\n")
        history.append(llm_response_text + f"\nObservation: {observation}")

        messages.append({"role": "assistant", "content": llm_response_text})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        llm_response = real_llm_call(messages)
        llm_response_text = llm_response.choices[0].message.content

    raise RuntimeError("Act-only agent did not finish with max_steps")


if __name__ == "__main__":
    QUESTION = (
        "How many years before the iPhone's 2007 release was the founder "
        "of the company that built it born?"
    )
    GROUND_TRUTH = "52 years (2007 - 1955, Steve Jobs)"

    react_answer = run_react_agent(QUESTION)
    print("-" * 70 + "\n")
    act_only_answer = run_act_only_agent(QUESTION)

    print("-" * 70)
    print(f"Ground truth: {GROUND_TRUTH}")
    print(f"ReAct answer: {react_answer}")
    print(f"Act-only answer: {act_only_answer}")