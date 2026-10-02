import dspy
from typing import Literal

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

class TriageEmail(dspy.Signature):
    email: str = dspy.InputField(desc="Raw body text of the customer's email")
    account_tier: Literal["free", "pro", "enterprise"] = dspy.InputField()

    category: Literal["billing", "technical", "account", "other"] = dspy.OutputField()
    urgency: int = dspy.OutputField(desc="1 = can wait for a week, 5 = wake someone up")
    one_line_summary: str = dspy.OutputField()

triage = dspy.Predict(TriageEmail)

body = "The stabilizer in my refrigerator is not working from yesterday"
r = triage(email=body, account_tier="pro")

print(f"category:{r.category}")
print(f"urgency:{r.urgency}")
print(f"one_line_summary:{r.one_line_summary}")
