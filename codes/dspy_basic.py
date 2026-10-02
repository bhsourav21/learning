import dspy

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

qa = dspy.Predict("question -> answer")

result = qa(question="Why is the sky blue at noon but red at sunlight?")

print()
dspy.inspect_history(n=1)
print()

print(f"answer:{result.answer}")