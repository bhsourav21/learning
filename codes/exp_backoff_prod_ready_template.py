# This combines everything: tenacity retry, logging, max_tokens, and distinguishing retryable vs.
# non-retryable errors.

from dotenv import load_dotenv
import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

load_dotenv()
client = OpenAI()

# Set up logging so retry messages appear in your console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai_client")

def is_rate_limit(exception):
    return isinstance(exception, openai.RateLimitError)

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
    before_sleep = before_sleep_log(logger, logging.WARNING), # log each retry,
    reraise=True # re-raise the original error after all retries fail
)
def chat(prompt: str, max_tokens: int = 512) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )

        usage = response.usage
        logger.info(
            f"""Tokens used - prompt: {usage.prompt_tokens}, 
            completion: {usage.completion_tokens},
            total: {usage.total_tokens}"""
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        logger.error("Invalid API key. Check your credentials.")
    except openai.BadRequestError as e:
        logger.error(f"Bad request: {e}")
        raise "do not retry - fix the prompt"
    
if __name__ == "__main__":
    result = chat("Summamrize exponential backoff in three sentences")
    print(f"result:{result}")







