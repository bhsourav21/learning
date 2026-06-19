from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
load_dotenv()

llm = OpenAI(temperature=0.6)

name_template = PromptTemplate(
    input_variables = ["cuisine"],
    template = """I want to open a restaurant for {cuisine} food.
    Suggest a fancy name for this."""
)

# LCEL: replaces the deprecated LLMChain
name_chain = name_template | llm

menu_template = PromptTemplate(
    input_variables = ["restaurant_name"],
    template = """Suggest me non-veg 2 starters, 2 non-veg main course 
    and 2 desert items for {restaurant_name}.
    First write the restaurant name and then put the menu."""
)

menu_chain = menu_template | llm

chain = (
    name_chain
    | RunnableLambda(lambda name: {"restaurant_name": name})
    | menu_chain
)

final_response = chain.invoke({"cuisine": "Indian"})
print(final_response)
