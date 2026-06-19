import re
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

system_prompt = (
    "You are a math expert. "
    "End your answer with: Final Answer: <number><unit>"
)

user_prompt = "Emma has twice as many stickers as Liam. Together they have 48 stickers. \
How many does Emma have? Think step by step."


def extract_final_answer(text: str) -> str:
    match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


client = OpenAI()

answers = []
for i in range(3):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1,
        max_tokens=512,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    reply = response.choices[0].message.content
    answers.append(reply)

for i, answer in enumerate(answers, start=1):
    print("-----------------------------------------")
    print(f"Run {i}: {answer}")

final_answers = [extract_final_answer(answer) for answer in answers]
vote_counts = Counter(final_answers)
final_result, max_votes = vote_counts.most_common(1)[0]

print("\n-----------------------------------------")
print("Vote counts:")
for answer, count in vote_counts.most_common():
    print(f"  {answer}: {count}")

print("\nFinal result (most common):", final_result)
print(f"Appeared {max_votes} out of {len(answers)} times")