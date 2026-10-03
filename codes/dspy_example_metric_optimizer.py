# ====================================================================
# WHAT THIS FILE DEMONSTRATES
# ====================================================================
# Running it took the devset score from 0% to 83% - without changing the
# LM, the metric, the devset, or any of the code below. Only the PROMPT
# changed. That is the whole point of a DSPy optimizer.
#
# THE SETUP (deliberately rigged to show the gap):
# The labels spell numbers out as words ("seven"), but the signature
# docstring is vague on purpose - just "Answer the question." So the
# program has no way to know what format you want. The baseline said:
#
#     expected: 'sixty'  got: 'There are 60 minutes in an hour.'
#     expected: 'four'   got: 'A square has four sides.'
#
# The model is not wrong about the world. It is wrong about the FORMAT,
# and nothing in the prompt told it otherwise. After optimizing:
#
#     expected: 'four'   got: 'four'
#     expected: 'six'    got: 'six'
#
# THE KEY IDEA - NO WEIGHTS ARE TRAINED:
# gemma3:4b is byte-for-byte identical before and after. BootstrapFewShot
# only ran the program over the trainset, scored each attempt with the
# metric, kept the passing ones, and wrote them into the prompt as solved
# examples. Three worked examples taught the format better than any prose
# instruction would have. "Optimizing" in DSPy means choosing what goes
# in the prompt, and the result lands in .demos on the predictor.
#
# Read the sections in order. Section 7 prints the demos and the real
# prompt, so you can see the change rather than take it on faith.
# ====================================================================

import dspy
from dspy.teleprompt import BootstrapFewShot

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

# ====================================================================
# 1. THE PROGRAM - note what is NOT in the docstring
# ====================================================================
# Last time we told the model exactly how to format its answer. Here we
# deliberately do not. The docstring is vague on purpose, because that
# is the gap the optimizer is going to fill for us.
#
# Our labels spell numbers out as lowercase words ("seven"), but asked
# "how many ...?" a model almost always replies with a digit ("7").
# Nothing in the prompt tells it otherwise, so it cannot know.

class CountingQuestion(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

program = dspy.Predict(CountingQuestion)

# ====================================================================
# 2. TWO SETS OF EXAMPLES, NOT ONE
# ====================================================================
# trainset - the optimizer is allowed to look at these
# devset   - held back, used only to score. This is how you avoid
#            fooling yourself: a program can look great on the examples
#            it was tuned on and still be useless on new questions.

trainset = [
    dspy.Example(question="How many days are in a week?",
                 answer="seven").with_inputs("question"),
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

# ====================================================================
# 3. THE METRIC - unchanged from the previous file
# ====================================================================
# Remember the signature: (example, prediction, trace=None).
#
# The trace argument finally matters here. dspy.Evaluate calls the metric
# with trace=None, meaning "just score this". BootstrapFewShot calls it
# WITH a trace, meaning "was this attempt good enough to keep as an
# example?". Same function, two jobs. We can ignore the difference
# because a plain True/False answers both questions.

def normalise(text: str) -> str:
    return text.strip().rstrip(".").lower()

def exact_match(example, prediction, trace=None):
    return normalise(example.answer) == normalise(prediction.answer)

evaluate = dspy.Evaluate(
    devset=devset,
    metric=exact_match,
    num_threads=1,
    display_progress=False,
)

# ====================================================================
# 4. BASELINE - score before touching anything
# ====================================================================
# Always do this first. Without a baseline you cannot tell whether the
# optimizer helped, did nothing, or made things worse.

print("=== BEFORE optimizing ===")
for example in devset:
    prediction = program(**example.inputs())
    mark = "PASS" if exact_match(example, prediction) else "FAIL"
    print(f"[{mark}] {example.question}")
    print(f"       expected: {example.answer!r}  got: {prediction.answer!r}")

before = evaluate(program)
print(f"\nbaseline score: {before.score:.0f}%\n")

# ====================================================================
# 5. THE OPTIMIZER
# ====================================================================
# WHAT IT DOES, in three steps:
#
#   1. Runs the current program on each trainset question.
#   2. Scores each attempt with our metric. Attempts that PASS are kept;
#      a kept attempt is called a "bootstrapped demo" - the model
#      successfully generated its own worked example.
#   3. Writes those demos into the prompt as solved examples, so at
#      inference time the model sees the pattern before answering.
#
# WHY IT IS NEEDED:
# No weights are trained here. Nothing about gemma3 changes. The only
# thing being optimized is the PROMPT - and specifically which examples
# go in it. That is enough, because showing a model three solved
# examples teaches a format far more reliably than describing it.
#
# The two knobs:
#   max_bootstrapped_demos - how many self-generated demos to include
#   max_labeled_demos      - how many raw trainset examples to drop in
#                            directly. These are the safety net: if the
#                            program fails every trainset question,
#                            there is nothing to bootstrap, and these
#                            labelled examples carry the format instead.

print("=== optimizing (this makes real LM calls, give it a moment) ===")
optimizer = BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=3,
    max_labeled_demos=3,
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

# ====================================================================
# 7. WHAT ACTUALLY CHANGED
# ====================================================================
# This is the payoff. The optimizer did not touch your code - it filled
# in the .demos list on the predictor, and DSPy renders those demos into
# the prompt on every call.

# When this was run, the three demos it chose were:
#
#     1. How many days are in a week?            -> 'seven'
#     2. How many legs does a spider have?       -> 'eight'
#     3. How many sides does a triangle have?    -> 'three'
#
demos = optimized.predictors()[0].demos
print(f"=== the optimizer attached {len(demos)} demos to the prompt ===")
for n, demo in enumerate(demos, start=1):
    print(f"  {n}. {demo.question} -> {demo.answer!r}")

print("\n=== the prompt that was actually sent on the last call ===")
dspy.inspect_history(n=1)

# ====================================================================
# 8. THE MOST IMPORTANT LESSON - LOOK AT WHAT STILL FAILED
# ====================================================================
# The optimized run scored 83%, not 100%. The two survivors were:
#
#     expected: 'sixty'   got: '60'
#     expected: 'ninety'  got: '90'
#
# Now look at the demos the optimizer picked: seven, eight, three. Every
# one is a SMALL number. The model generalised the spelling-out pattern
# for small numbers and fell straight back to digits for 60 and 90,
# because it never saw a large number written out.
#
# So the demos were not representative of the devset. That is a DATA
# problem, not a code problem - and it is the normal way few-shot
# optimization fails in practice. Two things caused it:
#
#   - max_bootstrapped_demos=3 is a tight budget.
#   - Trainset ORDER decided which three it got; the optimizer works
#     through the trainset and stops once the budget is full.
#
# TRY THIS to see it for yourself:
#   a) raise max_bootstrapped_demos to 6, or
#   b) move the 'twenty-four' and 'twenty-six' examples to the front of
#      the trainset so a large number gets into the prompt.
# Either should flip those two cases. The program never changes - only
# which examples the model gets shown.
#
# ====================================================================
# WHY trainset AND devset ARE SEPARATE
# ====================================================================
# The optimizer is allowed to look at the trainset; the devset is held
# back purely for scoring. If you score on the same examples you tuned
# against, you are measuring your own demos rather than real
# generalisation, and the number will flatter you.
#
# ====================================================================
# A NOTE ON RE-RUNNING
# ====================================================================
# Caching is off (line with configure_cache), and gemma3:4b is not
# deterministic, so the exact scores and which examples pass will shift
# a little between runs. The 0% -> 83% jump is the durable part; treat
# single percentages as noisy.
