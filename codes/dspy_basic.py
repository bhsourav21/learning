from dotenv import load_dotenv
from openai import OpenAI
import dspy

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")

dspy.configure(lm=lm)

# A one-liner program: topic -> fun_fact
get_fact = dspy.Predict("topic -> fun_fact")
result = get_fact(topic="the Moon")

print("result:")
print(result)

print("prompt:")
dspy.inspect_history(n=1)
