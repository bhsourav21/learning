import dspy

ex = dspy.Example(
    question="What is the capital of Australia",
    answer="Canberra",
).with_inputs("question")

print(ex.inputs())
print(ex.labels())
print(ex.question, ex.answer)