from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

load_dotenv()

client = OpenAI()

# Streaming: tokens arrive one chunk at a time
with client.chat.completions.stream(
    model="gpt-4o-mini",
    # model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Explain Async I/O in python"
        }
    ],
    stream_options={"include_usage": True},  # required to populate usage in final completion
) as stream:
    for event in stream:
        # stream() yields typed events, not raw chunks
        # content.delta events carry the text directly in event.delta
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)

# After the 'with' block, get the fully-typed final completion:
final = stream.get_final_completion()
print(f"\n\nfinal:{final}")
print(f"\n\nFinish reason : {final.choices[0].finish_reason}")
print(f"Prompt tokens : {final.usage.prompt_tokens}")
print(f"Output tokens : {final.usage.completion_tokens}")
