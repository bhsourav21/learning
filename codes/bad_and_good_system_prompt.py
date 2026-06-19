from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=512,
    system=( # <-- This is the SYSTEM PROMPT
    "You are a friendly cooking assistant. "
    "Always suggest simple recipes. "
    "Never recommend recipes that take more than 30 minutes."
    "Always suggest recipes that suites Indian taste."
    ), 
    messages=[
        {"role": "user", "content": "What can I make with eggs and cheese?"}
    ]
)

print("--------------------------------")
print("Response: ")
print("--------------------------------")
print(response.content[0].text)
print(f"\nstop_reason: {response.stop_reason}")

if response.stop_reason == "max_tokens":
    output_tokens = response.usage.output_tokens if response.usage else "unknown"
    print(
        f"[WARNING] Response was truncated (stop_reason=max_tokens). "
        f"Output used {output_tokens} tokens. "
        f"Increase max_tokens and retry.",
    )
print("--------------------------------")
