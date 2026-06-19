# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# EXAMPLE 1: ConversationBufferMemory — 5-turn conversation
# The model remembers everything said from Turn 1 onwards.
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Install: pip install langchain langchain-openai

from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ■■ Step 1: Create the language model ■■■■■■■■■■■■■■■■■■■■■■■■■■■■
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

# ■■ Step 2: Create BufferMemory ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# memory_key="history" means the stored conversation will be
# injected into the prompt under the variable {history}.
# return_messages=True returns Message objects (not a plain string)
# — required when using ChatOpenAI.

memory = ConversationBufferMemory(
memory_key="history", # variable name in the prompt template
return_messages=True # needed for chat models
)

# ■■ Step 3: Create a ConversationChain ■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ConversationChain wraps: prompt template + memory + llm together.
# It automatically handles loading and saving history on every call.

chain = ConversationChain(
llm=llm,
memory=memory,
verbose=True # prints the full prompt sent to the model each turn
)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Run 5 turns — note how later turns reference earlier ones
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

print("=" * 60)
print("TURN 1")
response1 = chain.invoke(
    {"input": "Hi! My name is Alice and I am a data scientist."}
)
print(f"AI: {response1['response']}")

print("\n" + "=" * 60)
print("TURN 2")
response2 = chain.invoke({"input": "I love working with Python and pandas."})
print(f"AI: {response2['response']}")

print("\n" + "=" * 60)
print("TURN 3")
response3 = chain.invoke(
    {"input": "My favourite project was building a churn prediction model."}
)
print(f"AI: {response3['response']}")

print("\n" + "=" * 60)
print("TURN 4 — testing memory: asking about Turn 1 info")
response4 = chain.invoke({"input": "What is my name and what do I do?"})
print(f"AI: {response4['response']}")
# Expected: AI should recall 'Alice' and 'data scientist' from Turn 1

print("\n" + "=" * 60)
print("TURN 5 — testing memory: asking about Turn 2 and 3 info")
response5 = chain.invoke(
    {"input": "What programming language do I use and what was myfavourite project?"}
)
print(f"AI: {response5['response']}")
# Expected: AI should recall 'Python/pandas' and 'churn prediction model'

# Inspecting the buffer after 5 turns

# ■■ Inspect the stored memory after 5 turns ■■■■■■■■■■■■■■■■■■■■■■
# memory.chat_memory.messages returns the full list of Message objects

print("\n■■ Memory contents after 5 turns ■■")
for i, msg in enumerate(memory.chat_memory.messages):
# Each message has a 'type' ('human' or 'ai') and 'content'
    role = "Human" if msg.type == "human" else "AI"
    # Truncate long messages for display
    content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
    print(f"[{i+1}] {role}: {content_preview}")

# Count total messages stored
total_messages = len(memory.chat_memory.messages)
print(f"\nTotal messages in buffer: {total_messages}")

# Expected: 10 messages (5 human + 5 AI responses)
# Load memory as a dict — this is what gets injected into the prompt

loaded = memory.load_memory_variables({})
print(f"Memory key: {list(loaded.keys())}") # ['history']
print(f"Type of history: {type(loaded['history'])}") # list of Message objects