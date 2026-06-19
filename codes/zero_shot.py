from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Zero-shot prompting: instructions only, no example input/output pairs.
system_prompt = (
    "You are a helpful cooking assistant. "
    "Suggest simple recipes that take 30 minutes or less."
)
user_prompt = "What can I make with eggs and cheese?"

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=512,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)

print("--------------------------------")
print("System prompt:")
print(system_prompt)
print("--------------------------------")
print("User prompt:")
print(user_prompt)
print("--------------------------------")
print("Response:")
print("--------------------------------")
print(response.choices[0].message.content)
print(f"\nfinish_reason: {response.choices[0].finish_reason}")
print("--------------------------------")
