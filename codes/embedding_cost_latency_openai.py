from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()

client = OpenAI()

texts = [
    "Sachin scored 108 not out vs England while Chansing mor ethan 350 in Chennai",
    "Dominoz Pizza is the Bast",
    "Pizza Hut is another famous pizza chain",
    "My favourite subject is mathematics",
    "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour",
    "Sourav Ganguly is the best Indian cricket captain of all time",
    "I enjoy cooking non-veg Indian food",
    "Reading story books is an excellent habit",
    "Chetan Bhat's novels attact me a lot",
    "Mangoe is the best fruit",
    "Machine learning is transforming the way we build software",
    "The Eiffel Tower is located in Paris, France",
    "Python is one of the most popular programming languages in the world",
    "Regular exercise improves both physical and mental health",
    "Electric vehicles are becoming increasingly affordable",
    "The Amazon rainforest produces 20 percent of the world's oxygen",
    "Virat Kohli is one of the greatest batsmen in modern cricket",
    "Artificial intelligence will reshape the job market in the next decade",
    "A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals",
    "The James Webb Space Telescope has captured the deepest images of the universe"
]

start = time.time()

response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

latency = time.time() - start

vectors = [item.embedding for item in response.data]

print(f"\nTotal embeddings received: {len(vectors)}\n")

for i, (text, vector) in enumerate(zip(texts, vectors)):
    print(f"Input {i+1}: \"{text}\"")
    print(f"Vector dimension: {len(vector)}")
    
    print()

#Calculate cost
# The price $0.02 per 1M tokens is OpenAI's current rate for text-embedding-3-small. 
tokens_used = response.usage.total_tokens
cost = (tokens_used / 1_000_000) * 0.02  # $0.02 per 1M tokens for text-embedding-3-small

print(f"Tokens used : {tokens_used}")
print(f"Cost        : ${cost:.6f}")
print(f"Latency     : {latency:.3f}s")