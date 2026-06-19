import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def structured_cot(question: str) -> dict:
    """Returns both the reasoning and the final answer separately."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=700,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer questions by first reasoning inside <thinking> tags, "
                    "then giving only the final answer inside <answer> tags."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    text = response.choices[0].message.content
    thinking = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return {
        "reasoning": thinking.group(1).strip() if thinking else "",
        "answer": answer.group(1).strip() if answer else text,
    }


# Test it
result = structured_cot(
    "If you invest $1000 at 5% annual interest, compounded yearly, "
    "how much do you have after 3 years?"
)
print("Reasoning:", result["reasoning"])
print("Answer:", result["answer"])
