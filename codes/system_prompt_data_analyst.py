from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=2048,
    system=( # <-- This is the SYSTEM PROMPT
    "You are a data analyst. "
    "Always analyze the data and provide insights. "
    "Always provide recommendations on the data. "
    "Always provide a report on the data. "
    "Always provide a summary of the data. "
    "Always provide a visualization of the data. "
    ), 
    messages=[
        {
            "role": "user", 
            "content": "Write a code for bubble sort algorithm."
        },
    ]
)

print("--------------------------------")
print("Response: ")
print("--------------------------------")
print(response.content[0].text)
print("--------------------------------")
print(f"\nstop_reason: {response.stop_reason}")

if response.stop_reason == "max_tokens":
    output_tokens = response.usage.output_tokens if response.usage else "unknown"
    print(
        f"[WARNING] Response was truncated (stop_reason=max_tokens). "
        f"Output used {output_tokens} tokens. "
        f"Increase max_tokens and retry.",
    )
print("--------------------------------")