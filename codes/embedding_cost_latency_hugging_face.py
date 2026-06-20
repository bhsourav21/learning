from dotenv import load_dotenv
from openai import OpenAI
import time
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

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

for text in texts:
    vec = model.encode("text")
    print(f"text:{text}")
    print(f"vec:{vec[:10]}")
    print(f"Vector dimension:{len(vec)}")
    print()

latency = time.time() - start

tokenizer = model.tokenizer
tokens_per_text = [len(tokenizer.encode(text)) for text in texts]
total_tokens = sum(tokens_per_text)

print(f"Tokens used : {total_tokens}")
print(f"Cost        : $0.00 (local model, no API charge)")
print(f"Latency     : {latency:.3f}s")
