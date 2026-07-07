import os
import re
import numpy as np
from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "bhsourav17-pinecone-index"
EMBED_MODEL = "text-embedding-3-small"

openai_client = OpenAI()
pc = Pinecone()
index = pc.Index(INDEX_NAME)

# ■■ 1. Load PDFs (same as your pinecone_upsert_custom_metadata.py) ■■■■■■■■■■
PDF_DIR = "/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/docs"
pdf_files = [
    "AI_Ecosystem_4Layer_Stack.pdf",
    "AI_ML_DS_Roles_Comparison.pdf",
]

pages = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(f"{PDF_DIR}/{pdf_file}")
    pages.extend(loader.load())

# ■■ 2. Chunk (same settings as your upsert pipeline) ■■■■■■■■■■■■■■■■■■■■■■■■
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    add_start_index=True,
    )
chunks_docs = splitter.split_documents(pages)

# ■■ 3. Extract text + metadata into parallel lists ■■■■■■■■■■■■■■■■■■■■■■■■■■■
chunk_texts = [doc.page_content for doc in chunks_docs]
chunk_metas = [
    {
        "source_filename": os.path.basename(doc.metadata.get("source", "")),
        "page_number": doc.metadata.get("page", 0) + 1,
        "start_index": doc.metadata.get("start_index", 0),
    }
    for doc in chunks_docs
]

# ■■ BM25 index (built once at startup) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Assume chunk_texts and chunk_metas are loaded (see Section 9)
STOPWORDS = {"a","an","the","is","in","it","of","and","or","to"}
def tokenise(text):
    t = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return [w for w in t.split() if w not in STOPWORDS and len(w)>1]

bm25 = BM25Okapi([tokenise(t) for t in chunk_texts])

# ■■ Dense retrieval (Pinecone) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def dense_retrieve(query: str, top_k: int = 10) -> list[dict]:
    emb = openai_client.embeddings.create(input=[query], model=EMBED_MODEL)
    vector = emb.data[0].embedding
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
    {"id": m.id, "score": m.score,
    "text": m.metadata.get("text",""), "source": "dense"}
    for m in results.matches
    ]

# ■■ Sparse retrieval (BM25) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def sparse_retrieve(query: str, top_k: int = 10) -> list[dict]:
    scores = bm25.get_scores(tokenise(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
    {"id": f"chunk-{idx}", "score": float(scores[idx]),
    "text": chunk_texts[idx], "source": "bm25"}
    for idx in top_idx if scores[idx] > 0
    ]

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "id",
    k: int = 60,
    ) -> list[dict]:
    """
    Merge multiple ranked lists using RRF.
    ranked_lists: each is a list of dicts sorted by score descending.
    Returns a single merged list sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    doc_store: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc[id_key]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            doc_store[doc_id] = doc # keep the doc for reference
    
    # Sort by RRF score descending
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
    {**doc_store[doc_id], "rrf_score": score}
    for doc_id, score in merged
    ]

# ■■ Full hybrid retrieval ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def hybrid_retrieve(query: str, top_k: int = 5) -> list[dict]:
    dense_results = dense_retrieve(query, top_k=10)
    sparse_results = sparse_retrieve(query, top_k=10)
    # Fuse using RRF
    merged = reciprocal_rank_fusion([dense_results, sparse_results])
    return merged[:top_k]

# ■■ Test ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# query = "What is the role of a data scientist versus an ML engineer?"
# results = hybrid_retrieve(query, top_k=5)

# print(f"\nHybrid results for: '{query}'")

# for i, r in enumerate(results, 1):
#     print(f" [{i}] RRF={r['rrf_score']:.4f} | src={r['source']} | {r['text'][:100]}")

queries = [
    "Which vector database has built-in hybrid search?",
    # "What does DSPy stand for / do?",
    # "What is Gemini 1.5 Pro's context window size?",
    # "Which tool is Anthropic's model with a 200K context window?",
    # "What primary tool stack does a Data Scientist use, per the comparison chart?"
]

for query in queries:
    results = hybrid_retrieve(query, top_k=5)
    print(f"\nHybrid results for: '{query}'")
    for i, r in enumerate(results, 1):
        # print(f" [{i}] RRF={r['rrf_score']:.4f} | src={r['source']} | {r['text'][:100]}")
        print(f" [{i}] RRF={r['rrf_score']:.4f} | src={r['source']} | {r['text']}")
