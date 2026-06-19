from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI(temperature=0.6)
name = llm.invoke("I want to open a restaurant for Indian food. " \
"Suggest a fancy name for this.")
print(f"name:{name}")

