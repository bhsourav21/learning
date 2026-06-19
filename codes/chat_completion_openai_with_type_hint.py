from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

load_dotenv()

client = OpenAI()

def get_chat_completion(
    # note the type hint for each parameter
    messages: list[ChatCompletionMessageParam],
    model: str = "gpt-4o-mini",
    max_tokens: int = 500,
    temperature: float = 1.0,
) -> ChatCompletion:
    """Send a list of messages to OpenAI and return the full 
    chat completion."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def build_prefill(country: str):
    return f"The capital of  {country} is: "


# Test it
def build_message(country: str):
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": f"""You are a concise
        "assistant."""},
        {"role": "user",   "content": f"""What is the capital 
        of France?"""},
        {"role": "assistant",   "content": f"""The capital of 
        {country} is: """},
    ]

    return messages

completion: ChatCompletion = get_chat_completion(build_message("France"))
print(build_prefill("France") + completion.choices[0].message.content)
