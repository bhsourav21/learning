import dspy

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

solve = dspy.ChainOfThought("word_problem -> answer: float")
r = solve(word_problem="A train leaves at 09:40 and arrives at 13:15."
          "It stopped for 25 minutes. How many hours was it moving?")
print(f"reasoning:{r.reasoning}")
print(f"reasoning:{r.answer}")
