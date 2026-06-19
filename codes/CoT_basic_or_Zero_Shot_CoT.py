from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

system_prompt = (
    "You are a math expert. "
    "End your answer with: Final Answer: <number><unit>"
)

user_prompt = "If a train travels 60 km/h for 2.5 hours, how far does it go? Think step by step."

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=1024,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)

print(response.choices[0].message.content)