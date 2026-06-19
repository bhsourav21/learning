from dotenv import load_dotenv
import anthropic

load_dotenv()

# Prefill the *structure*, not just an intro line ending with ":".
# The model continues after this text — if you only prefill a greeting,
# it often adds another one ("Here are some quick options!").
prefill = "Here are a few egg and cheese recipes:\n\n1. **"

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=512,
    system=( # <-- This is the SYSTEM PROMPT
    "You are a cooking assistant. "
    "Always suggest simple recipes. "
    "Never recommend recipes that take more than 30 minutes. "
    "When the last message is a partial assistant reply, continue it exactly. "
    "Do not add another introduction, preamble, or phrases like "
    "'Here are', 'Sure!', or 'Great question'."
    ), 
    messages=[
        {
            "role": "user", 
            "content": "What can I make with eggs and cheese?"
        },
        {
            "role": "assistant", # <-- PRE-FILLED ASSISTANT PROMPT
            "content": prefill # AI continues from here
        },
    ]
)

print(prefill + response.content[0].text)