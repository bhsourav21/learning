from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=512,
    system=( # <-- This is the SYSTEM PROMPT
    "You are a strict code reviewer. "
    "While reviewing a code, always check for the following: "
    "1. The code is written in python. "
    "2. The code is written in a way that is easy to understand. "
    "3. The code is written in a way that is easy to maintain. "
    "4. The code is written in a way that is easy to test. "
    "5. The code is written in a way that is easy to deploy. "
    "6. The code is written in a way that is easy to scale. "
    "7. The code is written in a way that is easy to debug. "
    "Make sure the code as comment to explain the code. "
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
