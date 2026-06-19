from dotenv import load_dotenv
import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

load_dotenv()
client = OpenAI()

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(6)
)
def call_openai(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content

# Usage
answer = call_openai("Explain exponential backoff in simple words")
print(f"answer:{answer}")