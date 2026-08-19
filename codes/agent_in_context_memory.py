"""
In-context memory: a single running `messages` list, seeded with a system
prompt, that grows every turn and is truncated (sliding-window) whenever it
exceeds a token budget. The full list is resent to the model on every call.
"""

import os
from dataclasses import dataclass

import tiktoken
from openai import OpenAI

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

MODEL = "gpt-4o-mini"
CONTEXT_LIMIT_TOKENS = 1600
SYSTEM_PROMPT = "You are a helpful assistant."

encoding = tiktoken.encoding_for_model(MODEL)


@dataclass
class ConversationTurn:
    user: str


def count_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += 4  # per-message overhead (role/name/separators)
        total += len(encoding.encode(message["content"]))
    return total + 2  # priming tokens for the reply


def call_llm(messages: list[dict]) -> str:
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def run_conversation(conversation_turns: list[ConversationTurn]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    loop_counter = 0
    for turn in conversation_turns:
        loop_counter = loop_counter + 1
        messages.append({"role": "user", "content": turn.user})

        # print("-"*100)
        # print(f"messages: loop{loop_counter}")
        # print(messages)
        # print("-"*100)
        # print()

        # Sliding-window truncation: evict oldest turns until back under budget
        while count_tokens(messages) > CONTEXT_LIMIT_TOKENS and len(messages) > 3:
            print('Context limit crossed')
            messages.pop(1)  # oldest surviving user turn
            if messages[1]["role"] == "assistant":
                messages.pop(1)  # its assistant reply

        response = call_llm(messages)  # <-- full array sent, every call
        messages.append({"role": "assistant", "content": response})
        print(f"User: {turn.user}\nAssistant: {response}\n")

    return messages

if __name__ == "__main__":
    conversation_turns = [
        ConversationTurn(user="Hi, my name is Sourav."),
        ConversationTurn(user="What's a good book on distributed systems?"),
        ConversationTurn(user="Why did you pick that one specifically?"),
        ConversationTurn(user="I mostly work in Python, does that matter?"),
        ConversationTurn(user="What's the difference between Paxos and Raft?"),
        ConversationTurn(user="Which one is easier to implement from scratch?"),
        ConversationTurn(user="Any good repos with a reference Raft implementation?"),
        ConversationTurn(user="Switching topics -- what's a CAP theorem in one sentence?"),
        ConversationTurn(user="Give me an example of a system that favors AP over CP."),
        ConversationTurn(user="And one that favors CP over AP?"),
        ConversationTurn(user="How does that tie back to Raft's leader election?"),
        ConversationTurn(user="What happens during a network partition in Raft?"),
        ConversationTurn(user="Can a split-brain scenario happen in Raft?"),
        ConversationTurn(user="What's the role of the term number in Raft?"),
        ConversationTurn(user="How is that different from Paxos's ballot numbers?"),
        ConversationTurn(user="Okay, let's go back to the book recommendation -- what was it called again?"),
        ConversationTurn(user="Can you remind me what my name is?"),
        ConversationTurn(user="What was the first topic we discussed today?"),
        ConversationTurn(user="Summarize our whole conversation in three bullet points."),
        ConversationTurn(user="Thanks, that's all for now."),
    ]
    run_conversation(conversation_turns)
