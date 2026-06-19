from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

load_dotenv()

client = OpenAI()

# Track token usage — only available in the FINAL chunk
# when stream_options={'include_usage': True} is set
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    stream=True,
    max_tokens = 512,
    stream_options={"include_usage": True}, # <-- request usage stats
    )

usage = None
for chunk in stream:
    # The final usage chunk has choices=[] — guard before accessing
    if chunk.choices:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

    # Usage only appears on the very last chunk (which has empty choices)
    if chunk.usage is not None:
        usage = chunk.usage

print()
if usage:
    print(f"\nPrompt tokens : {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens : {usage.total_tokens}")