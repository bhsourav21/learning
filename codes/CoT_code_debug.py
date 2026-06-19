from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def debug_code(buggy_code: str) -> str:
    """Debug Python code using Chain-of-Thought reasoning."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=600,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Python debugging expert. "
                    "Walk through the code line by line, identify the bug, "
                    "explain why it fails, then provide the fixed code."
                ),
            },
            {
                "role": "user",
                "content": f"Debug this code:\n\n{buggy_code}",
            },
        ],
    )
    return response.choices[0].message.content


# Test it
buggy_code = """
def find_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)

print(find_average([]))  # This crashes!
"""
print(debug_code(buggy_code))
