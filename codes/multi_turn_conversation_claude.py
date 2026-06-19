import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

system_prompt = f"""
    You are an expert tax consultant. Think step by step and respond.
"""

user_prompt = [
    "How to smartly invest $5000 to get maximum return in USA?",
    "What is your cofidence level on the investment plan?",
    "Justify why your confidence level is this",
    "Based on the confidence level you gave in turn 2 and your justification in turn 3, what is the single biggest risk I should watch out for?",
    "Given that risk, which of your original suggestions would you now rank as the safest option and why?"
]


messages = []  # grows with each turn: user → assistant → user → assistant ...

number_of_turns = 5

for i in range(number_of_turns):
    # Append the next user turn
    messages.append({"role": "user", "content": user_prompt[i]})

    response_anthropic = client.messages.create(
        model="claude-haiku-4-5",
        system=system_prompt,   # top-level param, not inside messages
        messages=messages,
        max_tokens=1024
    )

    assistant_text = response_anthropic.content[0].text

    # Append the assistant reply so the next turn has full history
    messages.append({"role": "assistant", "content": assistant_text})

    print("--------------------------------------------")
    print(f"Turn {i + 1}")
    print(f"User     : {user_prompt[i]}")
    print(f"Assistant: {assistant_text}")

print("--------------------------------------------")
print("Final response:")
print(f"\n\n{response_anthropic.content[0].text}")
