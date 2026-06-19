from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

response = client.messages.create(
    model = "claude-haiku-4-5",
    max_tokens=14000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[
        {
            "role": "user",
            "content": "Is there an infinite number of prime numbers where n mod 4 == 3?"
        }
    ]
)

for block in response.content:
    if block.type == "thinking":
        print("=== CLAUDE'S THINKING ===")
        print(block.thinking)
    elif block.type == "text":
        print("\n=== FINAL ANSWER ===")
        print(block.text)