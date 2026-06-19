from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Same system prompt and test query for both runs.
system_prompt = (
    "You are a support ticket classifier. "
    "Reply with exactly one word: Refund, Technical, or Billing."
)

# One example only — model sees Refund once and may over-use that label.
one_shot_examples = [
    ("I want a full refund for my broken laptop.", "Refund"),
]

# Three examples — model learns all three categories and their patterns.
three_shot_examples = [
    ("I want a full refund for my broken laptop.", "Refund"),
    ("The app crashes whenever I upload photos.", "Technical"),
    ("Why was I charged $19.99 twice on my statement?", "Billing"),
]

# Billing-style question (not a refund request).
test_query = "Why did my credit card get charged $9.99 twice this month?"
expected = "Billing"


def build_messages(examples: list[tuple[str, str]]) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for user_text, assistant_text in examples:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": test_query})
    return messages


def classify(client: OpenAI, examples: list[tuple[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=16,
        temperature=0,
        messages=build_messages(examples),
    )
    return response.choices[0].message.content.strip()


client = OpenAI()

one_shot_result = classify(client, one_shot_examples)
three_shot_result = classify(client, three_shot_examples)

print("=" * 50)
print("ONE-SHOT vs THREE-SHOT PROMPTING")
print("=" * 50)
print(f"\nTest query: {test_query}")
print(f"Expected:   {expected}\n")

print("-" * 50)
print("ONE-SHOT (1 example)")
for user_text, assistant_text in one_shot_examples:
    print(f"  User:      {user_text}")
    print(f"  Assistant: {assistant_text}")
print(f"\nPrediction: {one_shot_result}")
print(f"Correct:    {one_shot_result == expected}")

print("-" * 50)
print("THREE-SHOT (3 examples)")
for user_text, assistant_text in three_shot_examples:
    print(f"  User:      {user_text}")
    print(f"  Assistant: {assistant_text}")
print(f"\nPrediction: {three_shot_result}")
print(f"Correct:    {three_shot_result == expected}")

print("-" * 50)
print("Why three-shot works better here:")
print(
    "  One-shot only shows 'Refund', so the model may label "
    "payment issues as Refund too."
)
print(
    "  Three-shot also shows Technical and Billing examples, "
    "so the model learns the right category boundaries."
)
print("=" * 50)
