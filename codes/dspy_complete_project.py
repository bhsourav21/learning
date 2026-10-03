import dspy
from typing import Literal
import json, random

Category = Literal["billing", "technical", "account", "other"]

# signature
class TriageEmail(dspy.Signature):
    """Sort an inbound customer support email so it reaches the right team.

    Judge urgency by business impact rather than the customer's tone.
    """
    email: str = dspy.InputField(desc="Raw body text of the customer's email")
    account_tier: Literal["free", "pro", "enterprise"] = dspy.InputField()

    category: Category = dspy.OutputField()
    urgency: int = dspy.OutputField(desc="1 = can wait a week, 5 = page someone now")
    summary: str = dspy.OutputField(desc="One sentence, at most 20 words")

class DraftReply(dspy.Signature):
    """Write a short first reply to a customer support email.

    Acknowledge the specific problem, state the next concrete step, and give
    no time estimate you cannot keep. Never promise a refund or a fix date.
    """
    email: str = dspy.InputField()
    category: str = dspy.InputField()
    summary: str = dspy.InputField()

    draft_reply: str = dspy.OutputField(desc="Under 120 words, plain and warm")

# program
class TriageProgram(dspy.Module):
    def __init__(self, escalate_at: int = 4):
        super().__init__()
        self.escalate_at = escalate_at
        self.triage = dspy.ChainOfThought(TriageEmail)
        self.reply = dspy.ChainOfThought(DraftReply)

    def forward(self, email: str, account_tier: str) -> dspy.Prediction:
        t = self.triage(email=email, account_tier=account_tier)
        urgency = min(5, t.urgency + 1) if account_tier == "enterprise" else t.urgency

        if urgency >= self.escalate_at:
            draft = ""
        else:
            draft = self.reply(email=email, category=t.category, summary=t.summary).draft_reply

        return dspy.Prediction(
            category=t.category,
            urgency=urgency,
            summary=t.summary,
            draft_reply=draft,
            escalated=urgency >= self.escalate_at
        )

# data and metric
def load(path="tickets.jsonl", seed=0):
    rows = [json.loads(l) for l in open(path)]
    data = []
    for r in rows:
        data.append(
            dspy.Example(
                email=r["email"],
                account_tier=r["tier"],
                category=r["category"],
                urgency=r["urgency"]
            ).with_inputs("email", "account_tier")
        )
    random.Random(seed).shuffle(data)
    n = len(data)
    return data[: int(.5*n)], data[int(.5*n): int(.75*n)], data[int(.75*n):]

trainset, valset, testset = load()
print(len(trainset), len(valset), len(testset))

def triage_quality(example, prediction, trace=None) -> float:
    """0.0-1.0. Category matters most, then urgency, then reply brevity."""
    cat = float(prediction.category == example.category)
    urg = float(abs(prediction.urgency - example.urgency) <= 1)
    brief = 1.0 if (prediction.escalated or len(prediction.draft_reply.split()) <= 120) else 0.0

    score = 0.6 * cat + 0.3 * urg + 0.1 * brief
    
    if trace is not None:
        return score >= 0.9
    return score

# baseline, optimize, compare
lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

program = TriageProgram()
evaluate = dspy.Evaluate(
    devset=testset,
    metric=triage_quality,
    num_threads=16,
    display_progress=True
)

baseline = evaluate(program)
print(f"baseline: {baseline.score:.1f}%")

optimizer = dspy.MIPROv2(metric=triage_quality, auto="light", num_threads=16)
optimized = optimizer.compile(
    program,
    trainset=trainset,
    valset=valset,
    max_bootstrapped_demos=2,
    max_labeled_demos=2
)

tuned = evaluate(optimized)
print(f"optimized: {tuned.score:.1f}% (delta {tuned.score - baseline.score:+.1f})")

optimized.save("triage_v1.json")

program = TriageProgram()
program.load("triage_v1.json")

out = program(
    email="Hi - you charged my card twice this morning for the same invoice "
          "(#4471). I've attached both receipts. Can you sort this out?",
    account_tier="pro",
)

print(f"category:{out.category}")
print(f"urgency:{out.urgency}")
print(f"escalated:{out.escalated}")
print(f"summary:{out.summary}")