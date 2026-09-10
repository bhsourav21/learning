from dotenv import load_dotenv
from openai import OpenAI
import dspy

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

qa = dspy.Predict("question -> Answer")
result = qa(question="Why is the sky blue at noon but red at sunset?")

# print(result)
dspy.inspect_history(n=1) 
