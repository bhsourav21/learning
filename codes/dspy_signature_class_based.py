from dotenv import load_dotenv
from typing import Literal
import dspy

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class SentimentAnalysis(dspy.Signature):
    """Classify the sentiment of a product review"""

    review: str = dspy.InputField(
        desc="The product review text to classify"
    )
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField(
        desc="The overall sentiment of the review"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 to 1.0"
    )
    reasoning: str = dspy.OutputField(
        desc="Breif explanation of the classification"
    )

classify = dspy.Predict(SentimentAnalysis)
result = classify(review="The laptop is incredibly fast and the charge lasts for a day")

print(f"sentiment:{result.sentiment}")
print(f"confidence:{result.confidence}")
print(f"reasoning:{result.reasoning}")

print("prompt:")
dspy.inspect_history(n=1)
