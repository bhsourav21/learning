# RAG Pipeline — Multi-Company Pinecone Ingest Plan

## Context
The existing `rag_pipeline.py` is a partial/broken script: `pdf_files` is never defined (causes `NameError`), namespace is set per-filename rather than per-company, and `category_label` is hardcoded to `"ai-ecosystem"`. The goal is to rewrite it to walk 5 company directories, derive namespace and category_label automatically from directory/filename structure, and upsert all chunks into Pinecone in batches.

## Directory Structure (discovered)
```
rag_input/
├── acme/       → namespace: "acme"
├── globex/     → namespace: "globex"
├── hooli/      → namespace: "hooli"
├── initech/    → namespace: "initech"
└── umbrella/   → namespace: "umbrella"
```
Each company has 4 PDFs: `[company]_business_model.pdf`, `[company]_finance.pdf`, `[company]_hr_policy.pdf`, `[company]_it_infrastructure.pdf`.

## Key Design Decisions

### Namespace = company directory name
One namespace per company (5 total). The outer loop iterates over directories, and the entire company batch is upserted to that company's namespace — no per-filename grouping needed.

### category_label derived from filename
Strip the company prefix from the stem: `acme_hr_policy.pdf` → `hr_policy`.  
Valid values: `business_model`, `finance`, `hr_policy`, `it_infrastructure`.  
Helper function: check which of the 4 keys the filename stem ends with.

### Chunk IDs
Format: `{company}-chunk-{i}` (i = sequential index across all chunks for that company). This keeps IDs unique per namespace.

## Implementation Plan

**File to modify**: `AI_study/learning/codes/rag_pipeline.py` (full rewrite)

### Structure of new script

```
1. Imports + load_dotenv  (same as existing)
2. Constants: RAG_INPUT_PATH, INDEX_NAME, BATCH_SIZE = 100
3. Helper: get_category_label(filename) → str
4. Init: OpenAI client, Pinecone client, index handle  (same as existing)
5. Splitter: RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
6. Helper: get_embeddings(texts) → list[list[float]]  (same as existing)
7. Outer loop: for each company directory
   a. Load all PDFs → pages list
   b. Split pages → chunks list
   c. Build vectors list with metadata (all 7 fields)
   d. Batch loop: embed + upsert into company namespace
   e. Print progress per batch and total
8. Final print: all companies processed
```

### get_category_label logic
```python
CATEGORIES = ["business_model", "finance", "hr_policy", "it_infrastructure"]

def get_category_label(filename: str) -> str:
    stem = os.path.splitext(filename)[0]   # e.g. "acme_hr_policy"
    for cat in CATEGORIES:
        if stem.endswith(f"_{cat}"):
            return cat
    return "unknown"
```

### Metadata per chunk (all 7 required fields)
```python
{
    "text":             chunk.page_content,
    "source":           chunk.metadata.get("source", ""),
    "page":             chunk.metadata.get("page", 0),
    "start_index":      chunk.metadata.get("start_index", 0),
    "source_filename":  os.path.basename(chunk.metadata.get("source", "")),
    "page_number":      chunk.metadata.get("page", 0) + 1,   # 0-indexed → 1-indexed
    "category_label":   get_category_label(source_filename),
}
```

### Upsert (simplified vs existing — no per-filename grouping needed)
```python
for batch_start in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[batch_start : batch_start + BATCH_SIZE]
    texts = [chunks[batch_start + j].page_content for j in range(len(batch))]
    embeddings = get_embeddings(texts)
    for vec, emb in zip(batch, embeddings):
        vec["values"] = emb
    index.upsert(vectors=batch, namespace=company)   # company = namespace
    print(f"[{company}] Upserted chunks {batch_start+1}–{batch_start+len(batch)}")
```

## What Changes vs Existing Code

| Aspect | Existing | New |
|---|---|---|
| PDF discovery | `pdf_files` (undefined, NameError) | `os.listdir` over company dirs |
| Namespace | `source_filename` (per-file) | `company` directory name (per-company) |
| category_label | hardcoded `"ai-ecosystem"` | derived from filename |
| Outer loop | flat single-company | nested: company → PDFs |
| Chunk IDs | `chunk-{i}` | `{company}-chunk-{i}` |

## Verification
1. Run the script: `python rag_pipeline.py`
2. Confirm 5 namespaces created in Pinecone console (acme, globex, hooli, initech, umbrella)
3. Spot-check a random vector: verify all 7 metadata fields are present and `category_label` matches the source PDF category
4. Verify chunk counts are reasonable (~20 PDFs × expected chunks per doc)
