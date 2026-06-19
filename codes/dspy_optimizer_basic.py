# What each piece does, in plain English:

# Training data  →  "Here are examples of correct input/output pairs"
#      ↓
# Metric         →  "Here is how you score a prediction as good or bad"
#      ↓
# Optimizer      →  tries different few-shot examples from your training data,
#                   scores each combination using your metric,
#                   keeps the best one
#      ↓
# Optimized program  →  same code, but the prompt now includes
#                       the best few-shot examples automatically
# The key insight: you never touch the prompt yourself. The optimizer figures 
# out which examples to include in the prompt to maximise your metric score. 
# That's the whole idea.

# The three things you always need to run any optimizer:
# Thing             What it is              In this example
#---------------------------------------------------------------
# Program           your Module             SentimentClassifier
# Training data     labeled examples        trainset
# Metric            scoring function        sentiment_correct

import dspy
from dspy.teleprompt import BootstrapFewShot

# ── 1. Setup ───────────────────────────────────────────────────────────────
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# ── 2. Define the task (Signature) ────────────────────────────────────────
class ClassifySentiment(dspy.Signature):
    """Classify the sentiment of a movie review."""
    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="either 'positive' or 'negative'")

# ── 3. Define the program (Module) ────────────────────────────────────────
class SentimentClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifySentiment)

    def forward(self, review):
        return self.classify(review=review)

# ── 4. Training data — examples the optimizer learns from ─────────────────
#    dspy.Example = one input/output pair
#    .with_inputs("review") tells DSPy which fields are inputs (rest are labels)

trainset = [
    dspy.Example(review="This movie was absolutely amazing!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Total waste of time, very boring.",  sentiment="negative").with_inputs("review"),
    dspy.Example(review="I loved every minute of it!",        sentiment="positive").with_inputs("review"),
    dspy.Example(review="Terrible acting and awful plot.",     sentiment="negative").with_inputs("review"),
    dspy.Example(review="A masterpiece, highly recommended.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Couldn't even finish watching it.",  sentiment="negative").with_inputs("review"),
]

# ── 5. Metric — how do we score a prediction? ─────────────────────────────
#    DSPy calls this function for every example during optimization.
#    It must return True/False or a number (higher = better).

def sentiment_correct(example, prediction, trace=None):
    return prediction.sentiment.lower() == example.sentiment.lower()

# ── 6. Run the optimizer ───────────────────────────────────────────────────
#    BootstrapFewShot tries different few-shot examples from trainset
#    and picks the combination that scores highest on your metric.

optimizer = BootstrapFewShot(
    metric=sentiment_correct,
    max_bootstrapped_demos=2,   # max few-shot examples to add to the prompt
)

program         = SentimentClassifier()
optimized       = optimizer.compile(program, trainset=trainset)

# ── 7. Compare before vs after ────────────────────────────────────────────
# test_review = "The film had stunning visuals but a weak story."
test_review = "Although the film had a weak story, it had stunning visuals."

before = program(review=test_review)
after  = optimized(review=test_review)

print(f"Before optimization: {before.sentiment}")
print(f"After  optimization: {after.sentiment}")

# Save so you never need to re-run the optimizer
optimized.save("sentiment_optimized.json")