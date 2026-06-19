import openai
import time
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

load_dotenv()
client = OpenAI()

def stream_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Stream with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            full_text = ""
            with client.chat.completions.stream(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                timeout=30.0, # seconds — applies to the whole stream
                ) as stream:
                    for event in stream:
                        if event.type == "content.delta":
                            print(event.delta, end="", flush=True)
                            full_text += event.delta
            print()
            return full_text
        
        except openai.APIConnectionError as e:
            print(f"\nConnection error (attempt {attempt+1}): {e}")
        except openai.RateLimitError as e:
            wait = 2 ** attempt
            print(f"\nRate limited — retrying in {wait}s...")
            time.sleep(wait)
        except openai.APIStatusError as e:
            print(f"\nAPI error {e.status_code}: {e.message}")
            break # Don't retry on 4xx
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            break
    return ""

result = stream_with_retry("Explain Python generators in two sentences.")