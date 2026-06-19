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
logger = logging.getLogger(__name__)

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep = before_sleep_log(logger, logging.WARNING) # log each retry
)
def call_openai_with_logging(prompt: str) -> str:
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
answer = call_openai_with_logging("What is token in NLP?")
print(f"answer:{answer}")