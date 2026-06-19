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
            "content": "Explain recursion briefly."
        }
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# After the 'with' block the stream is closed; get the final typed message:
final = stream.get_final_message()
print(f"\n\nStop reason : {final.stop_reason}")

# Possible stop_reason values:
# 'end_turn' → model finished naturally
# 'max_tokens' → hit the max_tokens limit
# 'stop_sequence' → a custom stop sequence was matched
# 'tool_use' → model wants to call a tool