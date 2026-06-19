from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Few-shot prompting: system instructions + example input/output pairs,
# then the real user query (no examples needed for the new input).
system_prompt = (
    "You are a sentiment classifier. "
    "Reply with exactly one word: Positive, Negative, or Neutral."
)

# Example pairs shown to the model before the actual question.
few_shot_examples = [
    ("I love this product!", "Positive"),
    ("This is terrible and broken.", "Negative"),
    ("It's okay, nothing special.", "Neutral"),
]

user_prompt = "The food was amazing and the service was quick."

messages = [{"role": "system", "content": system_prompt}]
for example_input, example_output in few_shot_examples:
    messages.append({"role": "user", "content": example_input})
    messages.append({"role": "assistant", "content": example_output})
messages.append({"role": "user", "content": user_prompt})

print("messages: ")
print(messages)

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=64,
    messages=messages,
)

print("--------------------------------")
print("System prompt:")
print(system_prompt)
print("--------------------------------")
print("Few-shot examples:")
for i, (example_input, example_output) in enumerate(few_shot_examples, start=1):
    print(f"  Example {i}")
    print(f"    User:      {example_input}")
    print(f"    Assistant: {example_output}")
print("--------------------------------")
print("User prompt:")
print(user_prompt)
print("--------------------------------")
print("Response:")
print("--------------------------------")
print(response.choices[0].message.content)
print(f"\nfinish_reason: {response.choices[0].finish_reason}")
print("--------------------------------")
print("whole response:")
print("--------------------------------")
print(response)
print("--------------------------------")