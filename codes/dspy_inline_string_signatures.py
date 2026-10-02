import dspy
from typing import Literal

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

qa = dspy.Predict("question -> answer")
summarize = dspy.Predict("document, audience -> summary, key_terms")
rate = dspy.Predict("review -> sentiment: float, is_spam: bool")
tag = dspy.Predict("article -> topics: list[str], tone: Literal['format', 'casual']")

print(rate(review="Battery died in two days. Avoid.").sentiment)
text = "Sachin Tendular is a great batsman."
print(tag(article=text).tone)