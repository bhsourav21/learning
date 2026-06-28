import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "bhsourav17-pinecone-index"
TOP_K = 5

# Source files used during upsert
FILE_ECOSYSTEM = "AI_Ecosystem_4Layer_Stack.pdf"
FILE_ROLES     = "AI_ML_DS_Roles_Comparison.pdf"

openai_client = OpenAI()
pc = Pinecone()
index = pc.Index(INDEX_NAME)


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


def query_unfiltered(query_vector: list[float], top_k: int = TOP_K) -> list:
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )
    return results.matches


def query_filtered(query_vector: list[float], source_filename: str, top_k: int = TOP_K) -> list:
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        filter={"source_filename": {"$eq": source_filename}},
        include_metadata=True,
    )
    return results.matches


def _truncate(text: str, limit: int = 260) -> str:
    text = text.strip().replace("\n", " ")
    return text[:limit] + "…" if len(text) > limit else text


def _print_matches(matches: list, label: str) -> None:
    print(f"\n  ── {label} ──")
    if not matches:
        print("  (no results)")
        return
    for rank, m in enumerate(matches, start=1):
        meta = m.metadata
        print(
            f"  [{rank}] score={m.score:.4f}  file={meta.get('source_filename')}  "
            f"pg={meta.get('page_number')}"
        )
        print(f"       {_truncate(meta.get('text', ''))}")


def _precision_at_k(matches: list, expected_file: str) -> float:
    """Fraction of top-k results that come from the expected file."""
    if not matches:
        return 0.0
    hits = sum(1 for m in matches if m.metadata.get("source_filename") == expected_file)
    return hits / len(matches)


def compare(query: str, target_file: str, top_k: int = TOP_K) -> None:
    """Run the same query with and without a source_filename filter, then compare."""
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  QUERY  : {query}")
    print(f"  FILTER : source_filename = {target_file}")
    print(sep)

    embedding = get_embedding(query)

    unfiltered = query_unfiltered(embedding, top_k)
    filtered   = query_filtered(embedding, target_file, top_k)

    _print_matches(unfiltered, f"WITHOUT filter  (global top-{top_k})")
    _print_matches(filtered,   f"WITH filter     (top-{top_k} from {target_file})")

    p_unfiltered = _precision_at_k(unfiltered, target_file)
    p_filtered   = _precision_at_k(filtered,   target_file)

    print(f"\n  Precision@{top_k} (fraction of results from target file)")
    print(f"    Without filter : {p_unfiltered:.0%}  ({int(p_unfiltered * top_k)}/{top_k} from {target_file})")
    print(f"    With filter    : {p_filtered:.0%}  ({int(p_filtered * top_k)}/{top_k} from {target_file})")

    delta = p_filtered - p_unfiltered
    verdict = (
        f"Filtering improved precision by {delta:.0%}."
        if delta > 0
        else "Precision was already 100% without filtering — no gain needed."
        if delta == 0
        else f"Unexpected: precision dropped by {abs(delta):.0%} after filtering."
    )
    print(f"    → {verdict}")
    print()


if __name__ == "__main__":
    # ── Test 1: query that clearly targets the ecosystem file ──────────────
    compare(
        query="What are the four layers of the AI ecosystem stack?",
        target_file=FILE_ECOSYSTEM,
    )

    # ── Test 2: query that clearly targets the roles file ──────────────────
    compare(
        query="How does the role of a data scientist differ from an ML engineer?",
        target_file=FILE_ROLES,
    )

    # ── Test 3: ambiguous query — could match either file ──────────────────
    compare(
        query="What skills are important for working with AI models?",
        target_file=FILE_ROLES,
    )
