from dotenv import load_dotenv
from openai import OpenAI
import dspy

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class SolveMathProblem(dspy.Signature):
    "Solve the match problem step by step"
    problem: str = dspy.InputField()
    answer: str = dspy.OutputField()

solver = dspy.ChainOfThought(SolveMathProblem)
result = solver(problem="A train travels 120 km in 1.5 hours. What is its speed in km/h?")

print("reasoning:")
print(result.reasoning)
print()
print("result:")
print(result)
print()
print("answer:")
print(result.answer)