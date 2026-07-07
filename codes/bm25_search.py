import os
import re
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

load_dotenv()

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

# ■■ 4. Tokenise ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
STOPWORDS = {"a","an","the","is","in","it","of","and","or","to","for","with","on"}

def tokenise(text: str) -> list[str]:
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]

tokenised_corpus = [tokenise(t) for t in chunk_texts]

# ■■ 5. Build BM25 index ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
bm25 = BM25Okapi(tokenised_corpus, k1=1.5, b=0.75)
print(f"BM25 index built over {len(chunk_texts)} chunks")
print(f"Vocabulary size: {len(bm25.idf)} unique terms")

# ■■ 6. Query ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def bm25_retrieve(query: str, top_k: int = 5) -> list[dict]:
    scores = bm25.get_scores(tokenise(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    for i, idx in enumerate(top_idx):
        if scores[idx] > 0:
            return [
                {
                    "rank": i+1, "score": float(scores[idx]),
                    "text": chunk_texts[idx], "meta": chunk_metas[idx]
                }
            ]

# ■■ 7. Test it ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
results = bm25_retrieve("What are the four layers of the AI ecosystem?", top_k=5)
print(f"\nTop {len(results)} BM25 results:")
for r in results:
    print(f" [{r['rank']}] score={r['score']:.4f} | "
          f"file={r['meta']['source_filename']} | "
          f"pg={r['meta']['page_number']}")
    # print(f" {r['text'][:2000]}...")
    print(f" {r['text']}...")
