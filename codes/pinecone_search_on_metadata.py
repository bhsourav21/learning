import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "bhsourav17-pinecone-index"
TOP_K = 5

openai_client = OpenAI()
pc = Pinecone()
index = pc.Index(INDEX_NAME)


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def search(query: str, source_filename: str, top_k: int = TOP_K) -> list[dict]:
    """Semantic search restricted to chunks from a specific source file."""
    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter={"source_filename": {"$eq": source_filename}},
        include_metadata=True,
    )

    return results.matches


def display_results(matches: list, query: str, source_filename: str) -> None:
    print(f"\nQuery  : {query}")
    print(f"Filter : source_filename = {source_filename}")
    print(f"Hits   : {len(matches)}\n")
    print("─" * 70)
    for rank, match in enumerate(matches, start=1):
        meta = match.metadata
        print(f"[{rank}] Score: {match.score:.4f}  |  Page: {meta.get('page_number')}  |  Start index: {meta.get('start_index')}")
        print(f"    {meta.get('text', '').strip()[:300]}")
        print()


if __name__ == "__main__":
    # ── Example 1: search within the AI Ecosystem PDF ──────────────────────
    query_1 = "What are the layers of the AI ecosystem stack?"
    file_1 = "AI_Ecosystem_4Layer_Stack.pdf"
    matches_1 = search(query_1, file_1)
    display_results(matches_1, query_1, file_1)

    # ── Example 2: search within the Roles Comparison PDF ──────────────────
    query_2 = "What is the difference between a data scientist and an ML engineer?"
    file_2 = "AI_ML_DS_Roles_Comparison.pdf"
    matches_2 = search(query_2, file_2)
    display_results(matches_2, query_2, file_2)


# The key design choices:

# How the combined query works

# Pinecone's query() call does two things simultaneously:

# Semantic search — ranks all vectors by cosine similarity to the query embedding
# Metadata pre-filter — filter={"source_filename": {"$eq": source_filename}} tells Pinecone to only score vectors whose source_filename matches before doing ANN search
# This is more efficient than post-filtering; Pinecone applies the filter at the index level.

# Why source_filename instead of source

# The upsert code stores both, but source is a full absolute path (e.g. /Users/.../docs/AI_Ecosystem_4Layer_Stack.pdf) which is machine-specific. source_filename is just the basename, making filters portable.

# Usage pattern


# matches = search("your question here", "AI_Ecosystem_4Layer_Stack.pdf", top_k=5)
# The search() function returns raw Pinecone Match objects so you can use .score, .id, and .metadata freely.

