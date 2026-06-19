# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# EXAMPLE 13: More complex schema — Optional fields and a list
# Parse a product review into structured fields
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field # Pydantic v2
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

# ■■ Schema: structured product review ■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(description="Rating from 1 to 5", ge=1, le=5)
    pros: List[str] = Field(description="List of positive aspects, max 3 items")
    cons: List[str] = Field(description="List of negative aspects, max 3 items")
    verdict: str = Field(description="One-word verdict: 'Recommended', 'Neutral', or 'Avoid'")
    best_for: Optional[str] = Field(
        default=None,
        description="Who this product is best suited for, or null"
    )

# ■■ Parser and prompt ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
parser = PydanticOutputParser(pydantic_object=ProductReview)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional product reviewer. Be balanced and fair.{format_instructions}"),
    ("human", "Write a structured review of the {product} based on its {use_case} use case.")
    ])

chain = prompt | llm | parser

# ■■ Invoke ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
review = chain.invoke({
    "format_instructions": parser.get_format_instructions(),
    "product": "Sony WH-1000XM5 headphones",
    "use_case": "remote work and focus"
    })

# ■■ Access typed fields ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
print(f"Product : {review.product_name}")
print(f"Rating : {review.rating}/5") # guaranteed int
print(f"Pros : {review.pros}") # guaranteed list
print(f"Cons : {review.cons}") # guaranteed list
print(f"Verdict : {review.verdict}")
print(f"Best for: {review.best_for}") # Optional — may be None