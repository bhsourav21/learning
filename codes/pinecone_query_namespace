from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

INDEX_NAME   = "bhsourav17-pinecone-index"
TOP_K        = 5
NS_ECOSYSTEM = "AI_Ecosystem_4Layer_Stack.pdf"
NS_ROLES     = "AI_ML_DS_Roles_Comparison.pdf"

openai_client = OpenAI()
pc    = Pinecone()
index = pc.Index(INDEX_NAME)


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def query(text: str, namespace: str, top_k: int = TOP_K) -> None:
    embedding = get_embedding(text)
    results = index.query(
        vector=embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    print(f"\nQuery    : {text}")
    print(f"Namespace: {namespace}\n")
    for rank, m in enumerate(results.matches, start=1):
        meta = m.metadata
        snippet = meta.get("text", "").strip().replace("\n", " ")[:260]
        print(f"[{rank}] score={m.score:.4f}  pg={meta.get('page_number')}")
        print(f"    {snippet}\n")


if __name__ == "__main__":
    query(
        text="What are the four layers of the AI ecosystem stack?",
        namespace=NS_ECOSYSTEM,
    )

    query(
        text="How does the role of a data scientist differ from an ML engineer?",
        namespace=NS_ROLES,
    )
