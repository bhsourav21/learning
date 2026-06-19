# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# EXAMPLE 5: Measuring token usage — BufferMemory vs SummaryMemory
# Run both side by side for 15 turns on the SAME conversation.
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv

load_dotenv()

# ■■ Step 1: Create the language model ■■■■■■■■■■■■■■■■■■■■■■■■■■■■
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

# ■■ Same conversation for both tests ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
TURNS = [
    "Hi! I'm planning a road trip from New York to Los Angeles.",
    "I want to take Route 66 and stop at interesting places.",
    "My budget is around $3000 for the whole trip.",
    "I have 14 days to complete the journey.",
    "I'll be travelling alone in my Honda Civic.",
    "I love history and quirky roadside attractions.",
    "I prefer staying at motels rather than hotels.",
    "I'm vegetarian, so I need food options along the way.",
    "What are must-see stops in Missouri?",
    "What about Oklahoma?",
    "Any tips for driving through the Texas panhandle?",
    "How should I handle car maintenance for a long trip?",
    "How much to budget per day for food and fuel?",
    "What apps would you recommend?",
    "Summarise everything we've discussed so far."
]

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# TEST 1: BufferMemory — track cumulative tokens
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

buffer_memory = ConversationBufferMemory(memory_key="history", return_messages=True)
buffer_chain = ConversationChain(llm=llm, memory=buffer_memory, verbose=False)
buffer_tokens_per_turn = [] # track tokens used at each turn
buffer_total = 0

print("Testing BufferMemory for 15 turns...")
with get_openai_callback() as cb:
    for turn_num, user_input in enumerate(TURNS, 1):
        prev_total = cb.total_tokens
        buffer_chain.invoke({"input": user_input})
        tokens_this_turn = cb.total_tokens - prev_total
        buffer_tokens_per_turn.append(tokens_this_turn)
        print(f" Turn {turn_num:2d}: {tokens_this_turn:5d} tokens "
        f"(cumulative: {cb.total_tokens:6d})")
        buffer_grand_total = cb.total_tokens

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# TEST 2: SummaryMemory — track cumulative tokens
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

summary_memory = ConversationSummaryMemory(llm=llm, memory_key="history", return_messages=True)
summary_chain = ConversationChain(llm=llm, memory=summary_memory, verbose=False)
summary_tokens_per_turn = []
summary_total = 0
print("\nTesting SummaryMemory for 15 turns...")
with get_openai_callback() as cb:
    for turn_num, user_input in enumerate(TURNS, 1):
        prev_total = cb.total_tokens
        summary_chain.invoke({"input": user_input})
        tokens_this_turn = cb.total_tokens - prev_total
        summary_tokens_per_turn.append(tokens_this_turn)
        print(f" Turn {turn_num:2d}: {tokens_this_turn:5d} tokens "
        f"(cumulative: {cb.total_tokens:6d})")
        summary_grand_total = cb.total_tokens

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Print comparison report
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
print("\n" + "=" * 55)
print(f"{'COMPARISON AFTER 15 TURNS':^55}")
print("=" * 55)
print(f"{'Turn':<6} {'BufferMemory':>14} {'SummaryMemory':>14} {'Difference':>12}")
print("-" * 55)
for i, (b, sm) in enumerate(zip(buffer_tokens_per_turn, summary_tokens_per_turn), 1):
    diff = b - sm
    sign = "+" if diff > 0 else ""
    print(f" {i:<4} {b:>14,} {sm:>14,} {sign}{diff:>11,}")
    print("-" * 55)

print(f" {'TOTAL':<4} {buffer_grand_total:>14,} {summary_grand_total:>14,} "
f"{buffer_grand_total - summary_grand_total:>+12,}")
savings_pct = (buffer_grand_total - summary_grand_total) / buffer_grand_total * 100
print(f"\n SummaryMemory saved {savings_pct:.1f}% of total tokens over 15 turns")