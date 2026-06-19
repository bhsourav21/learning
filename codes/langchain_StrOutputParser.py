from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

prompt = ChatPromptTemplate.from_messages([
# System: set the AI's tone and role
("system", "You are a {tone} assistant who specialises in {domain}."),
# Human: the actual request
("human", "Give me a {length} explanation of {concept}.")
])

llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

result = chain.invoke({
"tone": "friendly", # fills {tone}
"domain": "machine learning",# fills {domain}
"length": "one-paragraph", # fills {length}
"concept": "overfitting" # fills {concept}
})

print("type(result):")
print(type(result))
print("\n\nresult:")
print(result)
