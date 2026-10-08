import os
from dataclasses import dataclass
import tiktoken
from openai import OpenAI

# Local Ollama, no OpenAI account involved. Ollama exposes an
# OpenAI-compatible API at /v1, so only base_url changes. Ollama
# ignores api_key, but the SDK refuses to construct without one,
# so pass any placeholder rather than setting OPENAI_API_KEY.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
MODEL = "qwen3:4b"

CONTEXT_LIMIT_TOKENS = 1600
SYSTEM_PROMPT = "You are a helpful assistant"

# tiktoken only knows OpenAI models - encoding_for_model("qwen3:4b")
# raises KeyError. cl100k_base is a reasonable stand-in for counting;
# it is NOT qwen3's real tokeniser, so treat the numbers as estimates
# (usually within ~10-20%) when trimming against CONTEXT_LIMIT_TOKENS.
encoding = tiktoken.get_encoding("cl100k_base")

@dataclass
class ConversationTurn:
    user_message: str

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
        messages.append({"role": "user", "content": turn.user_message})

        while count_tokens(messages) > CONTEXT_LIMIT_TOKENS and len(messages) > 3:
            print('Context limit crossed')
            messages.pop(1)
            if messages[1]["role"] == 'assistant':
                messages.pop(1)

        response = call_llm(messages)
        messages.append({"role":"assistant", "content":response})
        print(f"User:{turn.user_message}\nAssistant:{response}")
    return messages

if __name__ == "__main__":
    conversation_turns = [
        ConversationTurn(user_message="Hi, my name is Sourav."),
        ConversationTurn(user_message="What's a good book on distributed systems?"),
        ConversationTurn(user_message="Why did you pick that one specifically?"),
        ConversationTurn(user_message="I mostly work in Python, does that matter?"),
        ConversationTurn(user_message="What's the difference between Paxos and Raft?"),
        ConversationTurn(user_message="Which one is easier to implement from scratch?"),
        ConversationTurn(user_message="Any good repos with a reference Raft implementation?"),
        ConversationTurn(user_message="Switching topics -- what's a CAP theorem in one sentence?"),
        ConversationTurn(user_message="Give me an example of a system that favors AP over CP."),
        ConversationTurn(user_message="And one that favors CP over AP?"),
        ConversationTurn(user_message="How does that tie back to Raft's leader election?"),
        ConversationTurn(user_message="What happens during a network partition in Raft?"),
        ConversationTurn(user_message="Can a split-brain scenario happen in Raft?"),
        ConversationTurn(user_message="What's the role of the term number in Raft?"),
        ConversationTurn(user_message="How is that different from Paxos's ballot numbers?"),
        ConversationTurn(user_message="Okay, let's go back to the book recommendation -- what was it called again?"),
        ConversationTurn(user_message="Can you remind me what my name is?"),
        ConversationTurn(user_message="What was the first topic we discussed today?"),
        ConversationTurn(user_message="Summarize our whole conversation in three bullet points."),
        ConversationTurn(user_message="Thanks, that's all for now."),
    ]
    run_conversation(conversation_turns)

