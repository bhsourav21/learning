from dotenv import load_dotenv
from openai import OpenAI
import dspy

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class ExtractKeywords(dspy.Signature):
    text: str = dspy.InputField()
    keywords: list[str] = dspy.OutputField(desc="Top 3 keywords")

extract = dspy.Predict(ExtractKeywords)
result = extract(text="Machine learning models learn patterns from data...")

print("result:")
print(result.keywords)
