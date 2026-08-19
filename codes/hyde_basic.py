import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

corpus = [
    "It usually takes between 30 minutes and two hours to "
    "remove a wisdom tooth.",
    "The COVID-19 pandemic significantly impacted mental "
    "health, increasing depression and anxiety.",
    "Humans have used fire for approximately 800,000 years.",
    "Milvus is a cloud-based database for vector storage.",
]

def embed(texts, model="text-embedding-3-small"):
    resp = client.embeddings.create(input=texts, model=model)
    return np.array([d.embedding for d in resp.data])

def generate_hypothetical_docs(query, n=5):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Write a short passage that answers "
                "the question."
            },
            {
                "role": "user", "content": query
            },
        ],
        n=n,
        temperature=0.7,
        max_tokens=150,
    )
    return [choice.message.content for choice in resp.choices]

def hyde_vector(query, n=5):
    docs = generate_hypothetical_docs(query, n=n)
    doc_vecs= embed(docs)
    return doc_vecs.mean(axis=0)

def hyde_search(query, corpus_vecs, top_k=3):
    q_vec = hyde_vector(query)
    sims = corpus_vecs @ q_vec / (
        np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(q_vec)
    )
    ranked = np.argsort(-sims)[:top_k]
    return [(corpus[i], float(sims[i])) for i in ranked]

corpus_vecs = embed(corpus)
results = hyde_search("What is Milvus?", corpus_vecs)
for text, score in results:
    print(f"{score:.3f} {text}")