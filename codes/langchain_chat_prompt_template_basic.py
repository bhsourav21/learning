from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {language} programmer"),
    ("user",  "Explain the concept of {topic} in {language}"),
])

# formatted_message = chat_prompt.format_messages(
#     language = "python",
#     topic = "decorators"
# )

# display the chat prompt
# for msg in formatted_message:
#     print(f'"{msg.type.upper()}": "{msg.content}"')

# invoke using chain
chain = chat_prompt | llm
response = chain.invoke(
    {
        "language": "python",
        "topic": "decorator"
    }
)

print(f"response:")
print(response.content)
print(f"type(response):{type(response)}")

