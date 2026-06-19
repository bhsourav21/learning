# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# EXAMPLE 3: ConversationSummaryMemory — 15-turn conversation
# Watch how the summary evolves and keeps token cost bounded.
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Install: pip install langchain langchain-openai

from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ■■ Step 1: Create the language model ■■■■■■■■■■■■■■■■■■■■■■■■■■■■
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

# ■■ Step 2: Create SummaryMemory ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# SummaryMemory needs an LLM to generate the summaries.
# The same llm can be reused, or a cheaper/faster one can be passed.

memory = ConversationSummaryMemory(
    llm=llm, # used internally to create summaries
    memory_key="history", # injected into the prompt as {history}
    return_messages=True # return as Message objects for chat models
)

# ■■ Step 3: Create the chain ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False # set True to see the full prompt each turn
)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# 15-turn conversation about planning a road trip
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
turns = [
    "Hi! I'm planning a road trip from New York to Los Angeles.",
    "I want to take Route 66 and stop at interesting places.",
    "My budget is around $3000 for the whole trip.",
    "I have 14 days to complete the journey.",
    "I'll be travelling alone in my Honda Civic.",
    "I love history and quirky roadside attractions.",
    "I prefer staying at motels rather than hotels.",
    "I'm vegetarian, so I'll need food options along the way.",
    "What are the must-see stops in Missouri?",
    "What about Oklahoma — any hidden gems there?",
    "I'm a bit nervous about driving through the Texas panhandle.",
    "What's the best way to handle car maintenance for a long trip?",
    "How much should I budget per day for food and fuel?",
    "What apps would you recommend for a solo road tripper?",
    "Summarise everything we've discussed so far about my trip." # Turn 15: test recall
    ]

print("Running 15-turn conversation with SummaryMemory...\n")
responses = []
for i, user_input in enumerate(turns, start=1):
    response = chain.invoke({"input": user_input})
    responses.append(response["response"])
    print(f"Turn {i:2d} | User: {user_input[:60]}...")
    print(f" | AI : {response['response'][:80]}...")
    print()

# Inspecting memory state after 15 turns

# ■■ After all 15 turns: inspect what the memory looks like ■■■■■■■■
print("\n" + "=" * 60)
print("MEMORY INSPECTION AFTER 15 TURNS")
print("=" * 60)

# Load the current memory state
loaded_memory = memory.load_memory_variables({})
print("loaded_memory['history']:")
print(loaded_memory['history'])

# SummaryMemory stores a single summary string (not individual messages)
# The 'history' key contains a SystemMessage with the running summary
print("\nMemory type:", type(memory))
print("\nCurrent summary stored in memory:")
print("-" * 40)
# The summary is in memory.buffer (moving_summary_buffer belongs to SummaryBufferMemory)
print(memory.buffer)
print("-" * 40)
# Note: the summary should mention New York, LA, Route 66,
# $3000 budget, 14 days, Honda Civic, history lover,
# vegetarian, motels — all compressed into a paragraph!
# Count messages in the summary buffer
messages = memory.chat_memory.messages
print(f"\nMessages in raw chat_memory: {len(messages)}")
print("(Only very recent messages are kept — older ones are summarised)")
# Show the final Turn 15 response
print("\nTurn 15 AI response (should recall all key trip details):")
print(responses[14])