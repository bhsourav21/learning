from dotenv import load_dotenv
from openai import OpenAI
import dspy

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Format: "input1, input2 -> output1, output2"

# Single input, single output
# translate = dspy.Predict("document -> summary")

# Multiple inputs and outputs
translate = dspy.Predict("text, source_language, target_language -> translation")
result = translate(
    text = "My name is Lion",
    source_language = "English",
    target_language = "Bengali"
)

print("result")
print(result.translation)