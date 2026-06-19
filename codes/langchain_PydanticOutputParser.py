# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# EXAMPLE 12: PydanticOutputParser — Basic usage
# Ask the LLM for a book recommendation, get back a typed object
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field # Pydantic v2
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

# ■■ Step 1: Define your schema as a Pydantic model ■■■■■■■■■■■■■■■■
# Each field has a type annotation and a Field(description=...).
# The description helps the LLM understand what to put in each field.
class BookRecommendation(BaseModel):
    title: str = Field(description="Title of the recommended book")
    author: str = Field(description="Full name of the author")
    genre: str = Field(description="Genre of the book, e.g. 'science fiction'")
    year: int = Field(description="Year the book was published")
    summary: str = Field(description="One-sentence summary of the book")
    why_read_it: str = Field(description="One sentence explaining why this book suits the user")

# ■■ Step 2: Create the parser ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Pass your Pydantic model class — NOT an instance — to the parser.
parser = PydanticOutputParser(pydantic_object=BookRecommendation)

# ■■ Step 3: Build the prompt — inject format instructions ■■■■■■■■■
# parser.get_format_instructions() returns a string that tells the
# LLM exactly what JSON structure to produce.
# We inject it using a {format_instructions} variable in the prompt.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledgeable librarian who recommends books.\n"
        "{format_instructions}"
    ), # <-- injects JSON schema instructions
    (
        "human",
        "Recommend a {genre} book suitable for a {reader_level} reader."
    )
])

# ■■ Step 5: Build the full chain ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# prompt → llm → parser
# The parser automatically validates and converts JSON → BookRecommendation
chain = prompt | llm | parser

# ■■ Step 6: Invoke the chain ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Note: format_instructions is passed as a variable alongside your own vars
result = chain.invoke({
    "format_instructions": parser.get_format_instructions(), # schema hint
    "genre": "classic science fiction", # your variable
    "reader_level": "adult" # your variable
})

# ■■ Step 7: Use the typed result ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# result is a BookRecommendation object — not a string!
print(type(result))

# Access fields by name with dot notation
print(result.title) # e.g. "Dune"
print(result.author) # e.g. "Frank Herbert"
print(result.genre) # e.g. "science fiction"
print(result.year) # e.g. 1965 (an int, not a string!)
print(result.summary) # e.g. "A saga set on a desert planet..."
print(result.why_read_it) # e.g. "A perfect entry into epic sci-fi..."
# Convert to dict if needed
print(result.model_dump()) # {'title': 'Dune', 'author': 'Frank Herbert', ...}
# Convert to JSON string if needed
print(result.model_dump_json()) # '{"title": "Dune", "author": ...}'
