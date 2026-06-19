from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def solve_math(problem: str) -> str:
    """Solve a maths problem using Zero-Shot CoT."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=600,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a maths tutor. Always show your working step by step. "
                    "End your answer with: Final Answer: <number><unit>"
                ),
            },
            {
                "role": "user",
                "content": f"{problem}\n\nThink step by step.",
            },
        ],
    )
    return response.choices[0].message.content


# Test it
problem = (
    "A factory produces 240 widgets per hour. It runs 8 hours on weekdays and 5 hours "
    "on weekends. How many widgets are produced in a week (5 weekdays + 2 weekend days)?"
)
print(solve_math(problem))
