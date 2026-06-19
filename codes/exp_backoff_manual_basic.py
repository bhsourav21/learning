from dotenv import load_dotenv
import openai
from openai import OpenAI
import random
import time

load_dotenv()
client = OpenAI()

def call_openai_manual(
        prompt: str,
        max_retries: int = 6,
        base_wait: float = 1.0,
        max_wait: float = 60.0
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model = "gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            if attempt == max_retries - 1 :
                raise "All retries exhausted"
            
            # Calculate wait: 1s, 2s, 4s, 8s ... capped at max_wait
            wait = min(base_wait * (2 ** attempt), max_wait)
            jitter = random.uniform(0,1) # add randomness
            total_wait  = wait + jitter

            print(f"""Rate limit hit. Attempt {attempt + 1}/{max_retries}"). 
                  Waiting {total_wait:.1f}s before retry...""")
            time.sleep(total_wait)

answer = call_openai_manual("Summerize the concept of exponential backoff in one sentence")
print(f"answer:{answer}")
