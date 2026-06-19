from dotenv import load_dotenv
import anthropic
import time

load_dotenv()

client = anthropic.Anthropic()

full_text = ""
collected_chunks = []

# Streaming: tokens arrive one chunk at a time
with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "list three benefits of streaming APIs"
        }
    ]
) as stream:
    for text in stream.text_stream:
        full_text = full_text + text
        collected_chunks.append(text)
        print(text, end="", flush=True)

print() # newline after streaming finishes
print(f"\nTotal characters received:{len(full_text)}")
print(f"\nTotal chunks received:{len(collected_chunks)}")
print(collected_chunks)

# output:
# Total characters received:742

# Total chunks received:9
# ['#', ' Three Benefits of Streaming APIs\n\n1. **Real-Time Data Access**\n   - Receive continuous, '
# 'live updates', ' instantly rather than waiting for '
# 'periodic polling or batch processing\n   - Enables immediate', 
# ' responsiveness to changing conditions (e.g., live stock prices, '
# 'sensor data', ', user activity)\n\n2. **Reduced Latency and '
# 'Bandwidth**\n   - Data is', 
# ' pushed to clients only when it changes, '
# 'eliminating unnecessary repeated requests\n   - '
# 'Decreases network overhead and server load compared', 
# ' to constant polling\n\n3. **Better User Experience**\n   - Supports '
# 'dynamic, responsive applications with live updates (dash', 
# 'boards, notifications, collaborative tools)\n   - '
# 'Creates more engaging interfaces with minimal delays '
# 'between data changes', ' and user visibility']