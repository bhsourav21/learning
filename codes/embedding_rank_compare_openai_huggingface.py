import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI()

query = "Name some of great batters in cricket"

texts = [
    "Sachin scored 108 not out vs England while Chansing mor ethan 350 in Chennai",
    "Dominoz Pizza is the Bast",
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

response_query_openai = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
)

response_texts_openai = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

vectors_query_openai = np.array(response_query_openai.data[0].embedding)
vectors_texts_openai = np.array([item.embedding for item in response_texts_openai.data])

print(f"\nTotal embeddings received: {len(vectors_texts_openai)}\n")
print(f"Query: \"{query}\"\n")
print(f"{'#':<5} {'Similarity':<12} Text")
print("-" * 100)

scores = [(cosine_similarity(vectors_query_openai, vectors_texts_openai[i]), text) for i, text in enumerate(texts)]
scores.sort(key=lambda x: x[0], reverse=True)

for rank, (score, text) in enumerate(scores, start=1):
    print(f"{rank:<5} {score:<12.4f} \"{text}\"")


model = SentenceTransformer("all-MiniLM-L6-v2")

vector_query_minilm = model.encode(query)
vectors_texts_minilm = np.array(model.encode(texts))

print(f"\n\nTotal embeddings received: {len(vectors_texts_minilm)}\n")
print(f"Query: \"{query}\"\n")
print(f"{'#':<5} {'Similarity':<12} Text")
print("-" * 100)

scores_minilm = [(cosine_similarity(vector_query_minilm, vectors_texts_minilm[i]), text) for i, text in enumerate(texts)]
scores_minilm.sort(key=lambda x: x[0], reverse=True)

for rank, (score, text) in enumerate(scores_minilm, start=1):
    print(f"{rank:<5} {score:<12.4f} \"{text}\"")
