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
            "content": "Write a short poem on python"
        }
    ]
) as stream:
    print("stream.text_stream:")
    print(stream.text_stream)
    print()
    for text in stream.text_stream:
        # text_stream yields plain string content directly
        print(text, end="", flush=True)
        time.sleep(1)

print() # newline after streaming finishes