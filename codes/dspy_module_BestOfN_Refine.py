import dspy

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

def non_too_long(args, pred) -> float:
    """Reward short taglines: 1.0 if under 60 charcters, else 0.0"""
    return 1.0 if len(pred.tagline) < 60 else 0.0

base = dspy.ChainOfThought("product_description -> tagline")
best_of_n = dspy.BestOfN(module=base, N=5, reward_fn=non_too_long, threshold=1.0)
refine = dspy.Refine(module=base, N=5, reward_fn=non_too_long, threshold=1.0)

desc = "The cricket bat is made of English Willow. It weighs 2.5 pounds. Its a SS TON Bat. It has 14 grains. Its well seasoned and match ready."

best_of_n_tagline = best_of_n(product_description=desc).tagline
refine_tagline = refine(product_description=desc).tagline

print(f"BetOfN tagline: {best_of_n_tagline}")
print(f"Refine tagline: {refine_tagline}")
