from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {persona}. Respond in {language}. Always format your output as {format}."),
        ("human", "Please help me with the following: {task}")
    ]
)

chain = prompt | llm

response = chain.invoke(
    {
        "persona": "python expert",
        "language": "english",
        "format": "json",
        "task": "explain recursion"
    }
)

print("response")
print(response.content)
