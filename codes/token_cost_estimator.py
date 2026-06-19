from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"

# gpt-4o-mini pricing (USD per 1M tokens)
INPUT_PRICE_PER_M  = 0.150
OUTPUT_PRICE_PER_M = 0.600


def count_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model(MODEL)
    return len(enc.encode(text))


def estimate_cost(prompt: str, expected_output_tokens: int) -> dict:
    """
    Estimate the USD cost of one gpt-4o-mini API call.

    Args:
        prompt:                 The input prompt text.
        expected_output_tokens: Your estimate of how many tokens the response will be.

    Returns:
        A dict with token counts, per-part costs, total cost, and a CoT explanation.
    """
    input_tokens = count_tokens(prompt)

    input_cost  = (input_tokens          / 1_000_000) * INPUT_PRICE_PER_M
    output_cost = (expected_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
    total_cost  = input_cost + output_cost

    # ── Chain of Thought: ask the model to explain the calculation ─────────────
    cot_prompt = f"""
You are a cost calculation expert for OpenAI API usage.
Walk through the cost estimation step by step for the following call:

Model         : {MODEL}
Input tokens  : {input_tokens:,}
Output tokens : {expected_output_tokens:,}
Input price   : ${INPUT_PRICE_PER_M} per 1M tokens
Output price  : ${OUTPUT_PRICE_PER_M} per 1M tokens

Think step by step:
1. Calculate the input cost.
2. Calculate the output cost.
3. Sum them for the total.
4. State the final estimated cost clearly.
""".strip()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": cot_prompt}],
        max_tokens=300,
        temperature=0,
    )
    explanation = response.choices[0].message.content

    return {
        "model":                 MODEL,
        "input_tokens":          input_tokens,
        "expected_output_tokens": expected_output_tokens,
        "input_cost_usd":        round(input_cost,  8),
        "output_cost_usd":       round(output_cost, 8),
        "total_cost_usd":        round(total_cost,  8),
        "cot_explanation":       explanation,
    }


# ── Test it ────────────────────────────────────────────────────────────────────
prompt = (
    "Explain the difference between supervised and unsupervised learning "
    "with two concrete examples for each."
)

result = estimate_cost(prompt, expected_output_tokens=300)

print(f"Model           : {result['model']}")
print(f"Input tokens    : {result['input_tokens']:,}")
print(f"Output tokens   : {result['expected_output_tokens']:,}")
print(f"Input cost      : ${result['input_cost_usd']:.8f}")
print(f"Output cost     : ${result['output_cost_usd']:.8f}")
print(f"Total cost      : ${result['total_cost_usd']:.8f}")
print(f"\n── Chain of Thought Explanation ──\n{result['cot_explanation']}")
