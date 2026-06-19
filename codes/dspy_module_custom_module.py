# What this demonstrates:

# Custom module — subclass dspy.Module, declare sub-modules in __init__, 
# wire them in forward().

# Chaining steps — the output of step 1 (outline) feeds directly into 
# step 2 as an input field.

# Intermediate results — wrapping both in dspy.Prediction lets you inspect 
# each step's output, which is useful for debugging.

# ChainOfThought — both steps reason before producing output, which improves 
# quality for structured tasks.

from dotenv import load_dotenv
import dspy

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# ── Signatures (the "what") ────────────────────────────────────────────────

class GenerateOutline(dspy.Signature):
    """Given a topic, produce a structured outline with key points to cover."""
    topic: str = dspy.InputField()
    outline: list[str] = dspy.OutputField(desc="3-5 key points as a list")

class ExpandOutline(dspy.Signature):
    """Turn an outline into a short, readable article introduction."""
    topic: str = dspy.InputField()
    outline: list[str] = dspy.InputField()
    article_intro: str = dspy.OutputField(desc="2-3 paragraph introduction")


# ── Custom Module (the "how") ──────────────────────────────────────────────

class ArticlePlanner(dspy.Module):
    """
    A two-step pipeline:
      Step 1 → draft an outline from the topic
      Step 2 → expand that outline into an article introduction
    """

    def __init__(self):
        # Each step is its own named sub-module
        self.outline_step = dspy.ChainOfThought(GenerateOutline)
        self.expand_step  = dspy.ChainOfThought(ExpandOutline)

    def forward(self, topic: str):
        # Step 1: generate the outline
        outline_result = self.outline_step(topic=topic)

        # Step 2: use the outline to write the intro
        article_result = self.expand_step(
            topic=topic,
            outline=outline_result.outline,
        )

        # Return both so callers can inspect intermediate results
        return dspy.Prediction(
            outline=outline_result.outline,
            article_intro=article_result.article_intro,
        )


# ── Run it ─────────────────────────────────────────────────────────────────

planner = ArticlePlanner()

#the following two statements are same
result  = planner(topic="Why sleep is critical for learning")
# result  = planner.forward(topic="Why sleep is critical for learning")

print("=== OUTLINE ===")
for i, point in enumerate(result.outline, 1):
    print(f"  {i}. {point}")

print("\n=== ARTICLE INTRO ===")
print(result.article_intro)