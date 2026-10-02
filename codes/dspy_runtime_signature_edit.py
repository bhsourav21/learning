import dspy

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

sig = dspy.Signature("email -> category")
sig2 = sig.with_instructions("Classify the email. Prefer 'billing' whenever money is mentioned")
sig3 = sig2.append("confidence", dspy.OutputField(desc="0.0-1.0"), type_=float)

clf = dspy.Predict(sig3)

body = "I bought a television today. It costs $300."
out = clf(email=body)
print(f"category: {out.category}")
print(f"confidence: {out.confidence}")