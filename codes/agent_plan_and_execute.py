import json
import re
import operator
from ddgs import DDGS
from dotenv import load_dotenv
from openai import OpenAI
import ast

load_dotenv()

MODEL = "gpt-4o-mini"

client = OpenAI()

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

def web_search(query: str) -> str:
    """Real web search (DuckDuckGo, no api key required)"""
    try:
        results = list(DDGS().text(query, max_results=3))
    except Exception as e:
        raise RuntimeError(f"web_search failed: {e}")
    if not results:
        return "No results found"
    return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

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
        tree = ast.parse(expression.replace(",", ""), mode="eval")
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

def call_llm(messages: list[dict], tools: list[dict] | None = None):
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    return response.choices[0].message  # full message: gives both .content and .tool_calls

def parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered list like '1. Do X\n2. Do Y' into ['Do X', 'Do Y']."""
    steps = []
    for line in text.strip().splitlines():
        line = line.strip()
        match = re.match(r"^\d+[.)]\s*(.+)$", line)
        if match:
            steps.append(match.group(1).strip())
    return steps

def make_plan(question, tools):
    """STEP 1: call the LLM ONCE to produce an ordered list of steps"""
    prompt = (
        "Break the request into a short, ordered list of atomic steps. "
        "Each step should need at most one tool call. "
        f"Available tools: {[t['function']['name'] for t in tools]}.\n\n"
        f"Request: {question}\n\nReturn the plan as numbered list only."
    )
    plan_message = call_llm([{"role": "user", "content": prompt}])
    return parse_numbered_list(plan_message.content)  # e.g. ["Find Paris population", ...]

def execute_step(step, tools, prior_results):
    """STEP 2: tool-calling LLM executes ONE step, given the plan step + results so far"""
    messages = [
        {"role": "system", "content": "Execute exactly one step. Use a tool if needed. "},
        {"role": "user", "content": f"Step: {step}\nResults so far: {prior_results}"},
    ]
    response = call_llm(messages, tools=tools)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        return TOOL_IMPLS[tool_call.function.name](**args)

    return response.content

def plan_and_execute_agent(question, tools):
    plan = make_plan(question, tools)
    print("plan:")
    print("\n".join(f"- {item}" for item in plan))
    print()
    results = []
    for step in plan:
        result = execute_step(step, tools, results)
        results.append({"step": step, "result": result})

    synth_prompt = f"Steps and results: \n{results}\n\nWrite the final answer to: {question}"
    final = call_llm([{"role": "user", "content": synth_prompt}])
    return final.content

if __name__ == "__main__":
    question = "Compare Paris and Tokyo: find each city's population, compute the ratio between them, look up today's USD-to-JPY exchange rate, and write a 2-sentence summary."
    answer = plan_and_execute_agent(question, TOOL_SCHEMAS)
    print(f"Question:\n{question}\n\nAnswer:\n{answer}")
