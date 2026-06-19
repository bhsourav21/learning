from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
load_dotenv()

prompt_template = PromptTemplate(
    input_variables = ["cuisine"],
    template = """I want to open a restaurant for {cuisine} food.
    Suggest a fancy name for this."""
)

llm = OpenAI(temperature=0.6)

# LCEL: replaces the deprecated LLMChain
chain = prompt_template | llm

result = chain.invoke({"cuisine": "Indian"})
print(result)


