import dspy
from typing import Literal

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

classify = dspy.Predict("email -> category: Literal['billing', 'technical', 'account', 'other']")
emails = [
"Dear Sir, Your electricity bill is $50 for the month of Sep, 2026",
"Dear Sir, My cell phone has stopped working from today morning. Its not charging.",
"Dear Sir, My saving account balance os showing $100 less than what is should be.",
"Dear Sir, I am going to eat rice and fish in lunch today.",
]
runner = dspy.Parallel(num_threads=16)
batch = [(classify, {"email": e}) for e in emails]
results = runner(batch)

for r in results:
    print(r.category)