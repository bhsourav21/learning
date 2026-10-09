import os
import re
import ast
import operator

from ddgs import DDGS
from openai import OpenAI

MODEL = "qwen3:4b"
MAX_STEPS = 6

# qwen3 is a REASONING model. Ollama returns its private thinking in a
# separate `reasoning` field, and that thinking consumes the token budget
# BEFORE any visible text is produced. With a small budget `content` comes
# back as '' and the parse below fails with a confusing error. Measured:
# 300 tokens -> content ''; 2500 tokens -> clean two-line output.
# (Neither extra_body={"think": False} nor a /no_think suffix is honoured
# by Ollama's /v1 endpoint, so a generous budget is the working fix.)
MAX_TOKENS = 2500

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
)

# ----------------------------------------------------------------- tools
def web_search(query: str) -> str:
    """Real web search (DuckDuckGo, no API key required)."""
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
    """Safely evaluate a pure arithmetic expression, e.g. '1996 - 1973'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
    except Exception as e:
        raise RuntimeError(f"calculator failed on '{expression}': {e}")
    return str(result)

# "finish" is handled by the loop itself, not listed here.
TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
}

# ----------------------------------------------------------------- parsing
# Named groups must match what parse_react_step() reads below.
#   - DOTALL + lazy .*? so a Thought may span several lines
#   - (?P<arg>.*?) allows an EMPTY arg, so "finish[]" still parses
REACT_STEP_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*\n+\s*Action:\s*(?P<tool>\w+)\s*\[(?P<arg>.*?)\]",
    re.DOTALL,
)

def parse_react_step(raw_text):
    m = REACT_STEP_RE.search(raw_text.strip())
    if not m:
        raise ValueError(f"Could not parse ReAct step:\n{raw_text!r}")
    return m.group("thought").strip(), m.group("tool").strip(), m.group("arg").strip()

SYSTEM_PROMPT = """You are a reasoning agent that solves questions step by step.
At every turn you must output exactly two lines in this format:

Thought: <your reasoning about what you know so far and what to do next>
Action: <tool_name>[<tool_input>]

Available tools:
    web_search[query]     - look up a fact on the web
    calculator[expression] - evaluate arithmetic, e.g. calculator[1990 - 1980]
    finish[answer]        - give the final answer and stop

After you emit an action, the environment will run it and reply with:
Observation: <result>

Continue the Thought/Action/Observation cycle until you call finish[answer].
Do not write the Observation yourself - wait for it.
"""

# ----------------------------------------------------------------- the loop
def llm_step(question: str, history: list[str]) -> str:
    """THINK step. Sends the system prompt, the question, and the transcript
    so far. Without the question the agent has nothing to reason about."""
    transcript = "\n".join(history)
    user = f"Question: {question}"
    if transcript:
        user += f"\n\n{transcript}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=MAX_TOKENS,
        # Stop before the model invents its own Observation - the
        # environment supplies those, not the model.
        stop=["Observation:"],
    )
    return response.choices[0].message.content or ""

def run_react_agent(question: str, max_steps: int = MAX_STEPS) -> str:
    history: list[str] = []
    print(f"[question] {question}\n")

    for step_num in range(1, max_steps + 1):
        raw = llm_step(question, history)

        try:
            thought, tool, arg = parse_react_step(raw)
        except ValueError as e:
            # Feed the failure back as an observation so the model can
            # self-correct, rather than killing the whole run.
            print(f"[step {step_num}] parse failed: {e}")
            history.append(
                "Observation: ERROR your reply was not in the required format. "
                "Reply with exactly two lines: a Thought: line then an "
                "Action: tool[input] line."
            )
            continue

        print(f"[step {step_num}] Thought: {thought}")
        print(f"[step {step_num}] Action: {tool}[{arg}]")

        if tool == "finish":
            return arg

        fn = TOOLS.get(tool)
        if fn is None:
            observation = (
                f"ERROR: unknown tool '{tool}'. "
                f"Available: {', '.join(TOOLS)}, finish"
            )
        else:
            try:
                observation = fn(arg)
            except Exception as e:
                observation = f"ERROR: {e}"

        print(f"[step {step_num}] Observation: {str(observation)[:300]}\n")
        history.append(
            f"Thought: {thought}\nAction: {tool}[{arg}]\nObservation: {observation}"
        )

    raise RuntimeError(f"Agent did not call finish[] within {max_steps} steps")

if __name__ == "__main__":
    try:
        answer = run_react_agent("How old was the founder of Anthropic was when 9/11 happened?")
        print(f"\n[FINAL] {answer}")
    except RuntimeError as e:
        print(f"\n[FAILED] {e}")
