# Anthropic's Claude models support extended thinking: the model reasons through a problem
# step-by-step before giving its final answer. When streaming is enabled, this reasoning arrives as
# thinking_delta events — a unique capability not available in other APIs.

# Extended thinking — Claude reasons before answering
# Thinking tokens stream as 'thinking_delta' events
from dotenv import load_dotenv
import anthropic
import time

load_dotenv()

client = anthropic.Anthropic()


with client.messages.stream(
    model="claude-opus-4-5",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{
        "role": "user",
        "content": "What is the greatest common divisor of 1071 and 462?"
    }],
    ) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "thinking_delta":
                # Claude's internal reasoning — stream in grey/italic in your UI
                print(f"[thinking] {delta.thinking}", end="", flush=True)
            elif delta.type == "text_delta":
                # Claude's final answer
                print(delta.text, end="", flush=True)

final = stream.get_final_message()
print(f"\n\nStop reason : {final.stop_reason}")
print(f"Input tokens : {final.usage.input_tokens}")
print(f"Output tokens : {final.usage.output_tokens}")

# ■ Note: Thinking tokens count toward your max_tokens budget. Set budget_tokens high enough for
# complex problems (e.g. 10,000+). Thinking blocks cannot be partially recovered during error recovery
# — only text blocks can be resumed.