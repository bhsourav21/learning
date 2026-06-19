from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def advise_decision(scenario: str) -> str:
    """Analyse a business decision using Chain-of-Thought reasoning."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=700,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a startup financial advisor. "
                    "Break down the decision step by step: analyse cash flow, "
                    "ROI, risk, and timeline for each option before recommending."
                ),
            },
            {
                "role": "user",
                "content": f"{scenario}\n\nReason through each option carefully.",
            },
        ],
    )
    return response.choices[0].message.content


# Test it
scenario = """
A startup has $50,000 left in the bank. They have two options:
Option A: Spend $40,000 on a marketing campaign expected to bring
$80,000 in new revenue over 3 months.
Option B: Hire a developer for $35,000 to build a new feature
expected to increase retention by 20%, worth ~$60,000/year.
Which option should they choose?
"""
print(advise_decision(scenario))
