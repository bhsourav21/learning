# OpenAI introduced the Responses API as a successor to Chat Completions. It uses a typed event
# model instead of raw delta chunks, making it easier to handle different event types (text, tool calls,
# errors) in a structured way.

# The newer Responses API (2025+) — supports streaming natively
# Uses client.responses.create() instead of client.chat.completions.create()
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

with client.responses.stream(
    model="gpt-4o-mini",
    input="Explain streaming in one paragraph.",
    ) as stream:
    for event in stream:
        # The Responses API emits typed events
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "response.completed":
            print(f"\n\nTotal tokens: {event.response.usage.total_tokens}")

# Shorthand: iterate text only
with client.responses.stream(
    model="gpt-4o-mini",
    input="Name 5 Python libraries for AI.",
    ) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)