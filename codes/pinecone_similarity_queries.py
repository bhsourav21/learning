from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "bhsourav17-pinecone-index"
TOP_K = 3

queries = [
    "What are the layers of the AI ecosystem?",
    "How do large language models work?",
    "What is the role of vector databases in AI applications?",
    "What tools and frameworks are used for AI infrastructure?",
    "How is model training different from model inference?",
]

openai_client = OpenAI()
pc = Pinecone()
index = pc.Index(INDEX_NAME)

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding

for query in queries:
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)

    print(f"\nQuery: \"{query}\"")
    print("-" * 80)
    for rank, match in enumerate(results.matches, start=1):
        print(f"  Rank {rank} | Score: {match.score:.4f} | Page: {match.metadata.get('page', '?')}")
        print(f"  {match.metadata.get('text', '').strip()}")
        print()
