from dotenv import load_dotenv
import anthropic
import time

load_dotenv()

client = anthropic.Anthropic()

load_dotenv()

def stream_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Stream with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            full_text = ""
            with client.messages.stream(
                model="claude-haiku-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                        full_text += text
            print()
            return full_text
        
        except anthropic.APIConnectionError as e:
            print(f"\nConnection error (attempt {attempt+1}): {e}")
        except anthropic.RateLimitError as e:
            wait = 2 ** attempt
            print(f"\nRate limited — retrying in {wait}s...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(2 ** attempt) # retry on 5xx (overloaded)
            else:
                break # don't retry on 4xx
    return ""

result = stream_with_retry("Explain Python generators in two sentences.")