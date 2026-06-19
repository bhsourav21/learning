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
    ), 
    messages=[
        {"role": "user", "content": "What can I make with eggs and cheese?"}
    ]
)

print(response.content[0].text)