from dotenv import load_dotenv
from openai import OpenAI

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
    "Readng story books is an excellent habit",
    "Chetan Bhat's novels attact me a lot",
    "Mangoe is the best fruit"
]

response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

vectors = [item.embedding for item in response.data]

print(f"\nTotal embeddings received: {len(vectors)}\n")

for i, (text, vector) in enumerate(zip(texts, vectors)):
    print(f"Input {i+1}: \"{text}\"")
    print(f"  Vector size : {len(vector)}")
    print(f"  First 10 dims: {vector[:10]}")
    print()