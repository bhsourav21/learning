import numpy as np
from openai import OpenAI


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embeddings(
    texts: list[str] | str,
    client: OpenAI,
    openai_model: str = "text-embedding-3-small",
) -> np.ndarray:
    """Return a 2-D numpy array of shape (len(texts), embedding_dim)."""
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(input=texts, model=openai_model)
    return np.array([item.embedding for item in response.data])


def semantic_search(
    query: list[str] | str,
    texts: list[str],
    vectors_texts: np.ndarray,
    client: OpenAI,
    top_k: int = 3,
    openai_model: str = "text-embedding-3-small",
) -> dict[str, list[tuple[int, float, str]]]:
    """Return the top-k most similar texts for each query.

    Accepts a single query string or a list of query strings.
    Returns a dict mapping each query -> list of (rank, score, text) tuples.
    Corpus embeddings (vectors_texts) are pre-computed by the caller to avoid
    recomputing them across multiple calls.
    """
    single = isinstance(query, str)
    queries = [query] if single else query

    vectors_queries = get_embeddings(queries, client=client, openai_model=openai_model)

    results: dict[str, list[tuple[int, float, str]]] = {}
    for q_idx, q_text in enumerate(queries):
        scores = [
            (cosine_similarity(vectors_queries[q_idx], vectors_texts[i]), text)
            for i, text in enumerate(texts)
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        results[q_text] = [
            (rank, score, text)
            for rank, (score, text) in enumerate(scores[:top_k], start=1)
        ]

    return results
