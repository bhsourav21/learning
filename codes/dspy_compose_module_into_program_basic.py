import dspy

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step_one = dspy.ChainOfThought("a -> b")
        self.step_two = dspy.Predict("a, b -> c")

    def forward(self, a: str):
        b = self.step_one(a=a).b
        out = self.step_two(a=a, b=b).c
        print("in forward")
        return out

program = MyProgram()
print(f"result:{program(a='I have 100 dollars. I spent 7 dollars. What is the remaining amount I have?')}")
