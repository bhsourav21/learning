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
    for event in stream:
        print("--------------------------")
        print("event")
        print(event)