from dotenv import load_dotenv
from openai import OpenAI
import dspy

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

def calculate(expression: str) -> float:
    """Evaluate a simple arithmetic expression. Supports +, -, *, /"""
    return eval(expression)

class ResearchQuestion(dspy.Signature):
    """Answer a question by searching for information and reasoning."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Detailed, sourced answer")

agent = dspy.ReAct(ResearchQuestion, tools=[calculate])
result = agent(question="A train travels at 120 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. What is the total distance covered?")

print("result:")
print(result)
print()
print("answer")
print(result.answer)

    
