import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI()

query = [
    "Give me some stats on sports",
    "Tell me about some good food options",
    "Can you please share some information on AI",
    "Give me some info about vehicles",
    "I need some suggestions on diet"
]

texts = [
    "Sachin scored 108 not out vs England while Chansing mor ethan 350 in Chennai",
    "Dominos Pizza is the best",
    "Pizza Hut is another famous pizza chain",
    "My favourite subject is mathematics",
    "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour",
    "Sourav Ganguly is the best Indian cricket captain of all time",
    "I enjoy cooking non-veg Indian food",
    "Reading story books is an excellent habit",
    "Chetan Bhagat's novels attact me a lot",
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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

response_query = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
)

response_texts = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

vectors_query = np.array([item.embedding for item in response_query.data])
vectors_texts = np.array([item.embedding for item in response_texts.data])

print(f"\n[OpenAI text-embedding-3-small] — Top 3 results per query\n")

for q_idx, q_text in enumerate(query):
    scores = [(cosine_similarity(vectors_query[q_idx], vectors_texts[i]), text) for i, text in enumerate(texts)]
    scores.sort(key=lambda x: x[0], reverse=True)

    print(f"Query: \"{q_text}\"")
    print(f"  {'#':<5} {'Similarity':<12} Text")
    print("  " + "-" * 80)
    for rank, (score, text) in enumerate(scores[:3], start=1):
        print(f"  {rank:<5} {score:<12.4f} \"{text}\"")
    print()