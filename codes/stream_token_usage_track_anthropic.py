# Token usage is included in the message_delta event (cumulative)
# Access it via get_final_message() after the stream ends
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-haiku-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is machine learning?"}],
    ) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Usage is always available — no extra flag needed (unlike OpenAI)
final = stream.get_final_message()
print()
print(f"\nInput tokens : {final.usage.input_tokens}")
print(f"Output tokens : {final.usage.output_tokens}")
print(f"Total tokens : {final.usage.input_tokens + final.usage.output_tokens}")

# Question:
# This code returns numbr of tokens after the steam ends. what is the way to get token usage during streaming?
# Answer: 
# The short answer: you can't get output tokens mid-stream — they don't exist until generation finishes. 
# But you can get input tokens at the start of the stream and output tokens at the end (still within the stream, before it closes), 
# by iterating raw events instead of text_stream.
