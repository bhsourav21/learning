from dotenv import load_dotenv
import anthropic
import time

load_dotenv()

client = anthropic.Anthropic()

# Streaming: tokens arrive one chunk at a time
with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Explain async I/O in Python."
        }
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# After the with-block, retrieve the fully-typed Message object:
final = stream.get_final_message()
print(f"\n\nfinal:{final}")
print(f"\n\nStop reason : {final.stop_reason}")
print(f"Input tokens : {final.usage.input_tokens}")
print(f"Output tokens : {final.usage.output_tokens}")
print(f"Model : {final.model}")