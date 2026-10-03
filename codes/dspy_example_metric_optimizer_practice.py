import dspy
from dspy.teleprompt import BootstrapFewShot

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

class CountingQuestions(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

program = dspy.Predict(CountingQuestions)

trainset = [
    dspy.Example(question="How many days are in a week?",
                 answer="seven").with_inputs("question"),
    dspy.Example(question="How many weeks are there in a year?",
                answer="fifty-two").with_inputs("question"),
    dspy.Example(question="How many legs does a spider have?",
                 answer="eight").with_inputs("question"),
    dspy.Example(question="How many sides does a triangle have?",
                 answer="three").with_inputs("question"),
    dspy.Example(question="How many hours are in a day?",
                 answer="twenty-four").with_inputs("question"),
    dspy.Example(question="How many months are in a year?",
                 answer="twelve").with_inputs("question"),
    dspy.Example(question="How many colours are in a rainbow?",
                 answer="seven").with_inputs("question"),
    dspy.Example(question="How many continents are on Earth?",
                 answer="seven").with_inputs("question"),
    dspy.Example(question="How many letters are in the English alphabet?",
                 answer="twenty-six").with_inputs("question"),
]

devset = [
    dspy.Example(question="How many minutes are in an hour?",
                 answer="sixty").with_inputs("question"),
    dspy.Example(question="How many sides does a square have?",
                 answer="four").with_inputs("question"),
    dspy.Example(question="How many strings does a standard guitar have?",
                 answer="six").with_inputs("question"),
    dspy.Example(question="How many wheels does a bicycle have?",
                 answer="two").with_inputs("question"),
    dspy.Example(question="How many degrees are in a right angle?",
                 answer="ninety").with_inputs("question"),
    dspy.Example(question="How many players from one soccer team are on the pitch?",
                 answer="eleven").with_inputs("question"),
]

def normalise(text: str) -> str:
    return text.strip().rstrip(".").lower()

def exact_match(example, prediction, trace=None):
    return normalise(example.answer) == normalise(prediction.answer)

evaluate = dspy.Evaluate(
    devset = devset,
    metric = exact_match,
    num_threads =1 ,
    display_progress = False
)

print("=== BEFORE OPTIMIZING ===")
for example in devset:
    prediction = program(**example.inputs())
    mark = "PASS" if exact_match(example, prediction) else "FAIL"
    print(f"[{mark}] {example.question}")
    print(f"       expected: {example.answer!r}  got: {prediction.answer!r}")

before = evaluate(program)
print(f"before:{before}")
print(f"\nbaseline score: {before.score:.0f}%\n")

print("=== optimizing (this makes real LM calls, give it a moment) ===")
optimizer = BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=3,
    max_labeled_demos=6,
)
optimized = optimizer.compile(program, trainset=trainset)
print("done\n")

# ====================================================================
# 6. SCORE AGAIN - same devset, same metric, same LM
# ====================================================================
print("=== AFTER optimizing ===")
for example in devset:
    prediction = optimized(**example.inputs())
    mark = "PASS" if exact_match(example, prediction) else "FAIL"
    print(f"[{mark}] {example.question}")
    print(f"       expected: {example.answer!r}  got: {prediction.answer!r}")

after = evaluate(optimized)
print(f"\nbaseline  score: {before.score:.0f}%")
print(f"optimized score: {after.score:.0f}%\n")


demos = optimized.predictors()[0].demos
print(f"=== the optimizer attached {len(demos)} demos to the prompt ===")
for n, demo in enumerate(demos, start=1):
    print(f"  {n}. {demo.question} -> {demo.answer!r}")

print("\n=== the prompt that was actually sent on the last call ===")
dspy.inspect_history(n=1)