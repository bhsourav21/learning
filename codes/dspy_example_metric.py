import dspy

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

# ====================================================================
# 1. THE PROGRAM - the thing we want to measure
# ====================================================================
# The docstring is part of the prompt. We ask for short answers so that
# comparing against our labels is meaningful.

class ShortAnswer(dspy.Signature):
    """Answer the question with the bare fact only.
    No sentence, no explanation. Just the answer itself.
    If its a number, return is as int/float versus writing in a a word.
    """
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa = dspy.Predict(ShortAnswer)

# ====================================================================
# 2. THE TRAINSET - a list of dspy.Example
# ====================================================================
# A dspy.Example is just a labelled data point. You give it the same
# fields your signature uses: the input (question) and the correct
# output (answer).
#
# .with_inputs("question") is the part people miss. It tells DSPy
# "question is an INPUT; everything else here is a LABEL". Without it
# DSPy would not know which fields to feed the program and which to
# hold back as the expected answer.

trainset = [
    dspy.Example(question="What is the capital of France?",
                 answer="Paris").with_inputs("question"),
    dspy.Example(question="What is the capital of Japan?",
                 answer="Tokyo").with_inputs("question"),
    dspy.Example(question="What is 2 + 2?",
                 answer="4").with_inputs("question"),
    dspy.Example(question="Which planet is the largest in our solar system?",
                 answer="Jupiter").with_inputs("question"),
    dspy.Example(question="How many continents are there on Earth?",
                 answer="7").with_inputs("question"),
]

# An Example behaves like an object with attributes, and it can show you
# which half is which:
first = trainset[0]
print("--- what one Example looks like ---")
print(f"question (input): {first.question}")
print(f"answer   (label): {first.answer}")
print(f".inputs()        : {first.inputs()}")
print(f".labels()        : {first.labels()}")
print()

# ====================================================================
# 3. THE METRIC - a plain function that scores one prediction
# ====================================================================
# A DSPy metric is always called with (example, prediction, trace=None):
#   example    - one dspy.Example from your set, with the label
#   prediction - what your program actually returned
#   trace      - only used by optimizers; ignore it while learning
#
# Return True/False (or a number). Here: case-insensitive exact match.

def normalise(text: str) -> str:
    """Lowercase and drop surrounding spaces and any trailing period."""
    return text.strip().rstrip(".").lower()

def exact_match(example, prediction, trace=None):
    return normalise(example.answer) == normalise(prediction.answer)

# ====================================================================
# 4. SCORING BY HAND - so you can see what Evaluate does for you
# ====================================================================
print("--- scoring each example by hand ---")
score = 0
for example in trainset:
    prediction = qa(**example.inputs())   # feed only the inputs
    correct = exact_match(example, prediction)
    score += correct
    mark = "PASS" if correct else "FAIL"
    print(f"[{mark}] {example.question}")
    print(f"       expected: {example.answer!r}  got: {prediction.answer!r}")

print(f"\nscore: {score}/{len(trainset)} = {score / len(trainset):.0%}\n")

# ====================================================================
# 5. THE SAME THING WITH dspy.Evaluate
# ====================================================================
# Identical idea, but it handles looping, threading and a results table.
# This is what you pass to an optimizer later on.

print("--- the same score via dspy.Evaluate ---")
evaluate = dspy.Evaluate(
    devset=trainset,
    metric=exact_match,
    num_threads=1,         # keep it 1 so the output stays readable
    display_progress=True,
)
print(evaluate(qa))
