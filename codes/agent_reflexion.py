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


def react_agent_core(messages, tools, max_steps=8):
    """ReAct loop over an already-seeded messages list; returns the final answer text."""
    for step in range(max_steps):
        response = call_llm(messages, tools=tools)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            observation = TOOL_IMPLS[name](**args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(observation)
                }
            )

    return "Gave up after max_steps without a final answer"


def reflexion_agent(question, tools, max_retries=2):
    memory = []

    for attempt in range(max_retries + 1):
        messages = [
            {
                "role": "system",
                "content": "Solve the task using tools."
            },
            {
                "role": "user",
                "content": question
            }
        ]

        if memory:
            messages.append(
                {
                    "role": "user",
                    "content": "Notes from your previous attempt:\n" + "\n".join(memory)
                }
            )


        answer = react_agent_core(messages, tools)

        verdict = call_llm([
            {
                "role": "user",
                "content": f"Question: {question}\nProposed answer: {answer}\n"
                "Is this correct and internally consistent? Reply CORRECT or explain the flaw."
            }
        ])

        print()
        print("question:")
        print(question)
        print()
        print("answer:")
        print(answer)
        print()
        print("verdict:")
        print(verdict.content)
        print()

        if verdict.content.strip().startswith("CORRECT"):
            return answer

        memory.append(verdict.content) # store the critique, try again

    return answer # best effort after running out of retries

if __name__ == "__main__":
    question = "Compare Paris and Tokyo: find each city's population (the scale should be similar), compute the ratio between them, look up today's USD-to-JPY exchange rate, and write a 2-sentence summary."
    answer = reflexion_agent(question, TOOL_SCHEMAS)
    print(f"Question:\n{question}\n\nAnswer:\n{answer}")