"""
Local reproduction harness for the RAGPipeline defined in rag_pipeline.py.

WHY THIS FILE EXISTS
--------------------
This analysis was run inside a network-sandboxed shell that cannot reach
api.openai.com or api.pinecone.io (outbound egress is allow-listed and those
hosts are not on it: OpenAI calls fail with `httpx.ProxyError: 403 Forbidden`).
The real pipeline (rag_pipeline.py) is unmodified and is what you should run
against Pinecone + OpenAI in your own environment.

To still produce genuine, reproducible evidence for the failure-mode writeup
instead of inventing numbers, this script:
  1. Loads the *real* PDFs from rag_input and chunks them with the exact same
     RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100) config
     used in RAGPipeline.load_and_chunk.
  2. Replaces the OpenAI `text-embedding-3-small` call with a per-namespace
     TF-IDF vectorizer + cosine similarity (scikit-learn, fully offline). This
     is a stand-in retriever, not the same embedding space as production, but
     it is computed on the real chunk text, so retrieval-order effects driven
     by real document structure (near-duplicate templates, table fragmentation,
     ambiguous vocabulary) are genuine and reproducible, not fabricated.
  3. Logs every query in the same structured JSONL schema recommended for the
     production pipeline (see rag_pipeline_logging_patch.py).
  4. For the "final answer" field: since no LLM endpoint is reachable, answers
     for the fault-injection scenarios were authored by hand against the exact
     retrieved context, following the pipeline's real prompt contract ("answer
     only from context, else say I don't know"), to illustrate the two
     realistic behaviors (loose prompt -> hallucinates, strict prompt ->
     refuses). This is called out explicitly wherever it applies.
"""
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAG_INPUT = "/sessions/modest-keen-cray/mnt/rag_input"
LOG_PATH = "/sessions/modest-keen-cray/mnt/outputs/rag_query_log.jsonl"
CATEGORIES = ["business_model", "finance", "hr_policy", "it_infrastructure"]


def get_category_label(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    for cat in CATEGORIES:
        if stem.endswith(f"_{cat}"):
            return cat
    return "unknown"


def strip_boilerplate(page_text: str) -> str:
    """Drop the running header/footer that every page of every doc repeats:
        '<Company> | <Industry>'
        'Company ID: <id> | <Doc Type>'
        'CONFIDENTIAL — <Company> | Page N | For internal use only'
    These three lines carry near-zero information (they're identical across
    dozens of chunks in a namespace) but inject the company name + generic
    words into every chunk's TF-IDF vector, diluting the terms that actually
    distinguish one chunk from another.
    """
    lines = page_text.split("\n")
    out = []
    for line in lines:
        if "Company ID:" in line:
            # the title line right before it is the other half of the same
            # boilerplate block ("<Company> | <Industry>") -> drop it too
            if out and "|" in out[-1] and len(out[-1]) < 100:
                out.pop()
            continue
        if "CONFIDENTIAL" in line and "Page" in line:
            continue
        out.append(line)
    return "\n".join(out)


def load_and_chunk_company(company_path: str, company: str, clean: bool = False):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=100, add_start_index=True
    )
    records = []  # each: {"text", "source_filename", "page_number", "category_label"}
    for pdf_file in sorted(os.listdir(company_path)):
        if not pdf_file.endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(company_path, pdf_file))
        category = get_category_label(pdf_file)
        for page_no, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if clean:
                page_text = strip_boilerplate(page_text)
            for chunk_text in splitter.split_text(page_text):
                records.append({
                    "text": chunk_text,
                    "source_filename": pdf_file,
                    "page_number": page_no + 1,
                    "category_label": category,
                })
    return records


class LocalNamespace:
    """TF-IDF stand-in for one company's Pinecone namespace."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([r["text"] for r in records])

    def query(self, text: str, top_k: int = 4, category_filter: str | None = None,
              score_threshold: float = 0.0):
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        order = np.argsort(-sims)
        matches = []
        for idx in order:
            rec = self.records[idx]
            score = float(sims[idx])
            if category_filter and rec["category_label"] != category_filter:
                continue
            if score < score_threshold:
                continue
            matches.append({**rec, "score": score, "char_len": len(rec["text"])})
            if len(matches) >= top_k:
                break
        return matches


KEYWORD_ROUTER = {
    "hr_policy": ["leave", "pto", "vacation", "sick", "hiring", "onboarding",
                  "hr ", "human resources", "benefit", "performance review",
                  "code of conduct", "remote work", "hybrid", "working hours"],
    "finance": ["ebitda", "gross margin", "net profit", "balance sheet",
                "expenditure", "reimbursement", "purchase order", "audit",
                "total revenue", "income statement"],
    "it_infrastructure": ["mfa", "vpn", "laptops", "laptop", "byod", "cloud",
                          "aws", "azure", "encryption", "disaster recovery",
                          "rto", "rpo", "incident response", "security"],
    "business_model": ["arr", "tpv", "value proposition", "target market",
                       "strategic", "competitive", "pipeline value",
                       "business model", "revenue model"],
}


def infer_category(question: str) -> str | None:
    q = question.lower()
    hits = {
        cat: sum(1 for kw in kws if re.search(r"\b" + re.escape(kw.strip()) + r"\b", q))
        for cat, kws in KEYWORD_ROUTER.items()
    }
    hits = {c: n for c, n in hits.items() if n > 0}
    if not hits:
        return None
    # ambiguous if 2+ categories tie for the top score -> caller should widen, not narrow
    top = max(hits.values())
    winners = [c for c, n in hits.items() if n == top]
    return winners[0] if len(winners) == 1 else None


def log_query(query, namespace, top_k, matches, answer, note=""):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "namespace": namespace,
        "top_k_requested": top_k,
        "retrieved_chunks": [
            {
                "rank": i + 1,
                "score": round(m["score"], 4),
                "category_label": m["category_label"],
                "source_filename": m["source_filename"],
                "page_number": m["page_number"],
                "char_len": m["char_len"],
                "text_preview": m["text"][:160].replace("\n", " "),
            }
            for i, m in enumerate(matches)
        ],
        "final_answer": answer,
        "note": note,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def main():
    if os.path.exists(LOG_PATH):
        open(LOG_PATH, "w").close()

    namespaces = {}
    for company in sorted(os.listdir(RAG_INPUT)):
        company_path = os.path.join(RAG_INPUT, company)
        if not os.path.isdir(company_path):
            continue
        records = load_and_chunk_company(company_path, company)
        namespaces[company] = LocalNamespace(records)
        print(f"[{company}] {len(records)} chunks "
              f"(by category: "
              f"{ {c: sum(1 for r in records if r['category_label']==c) for c in CATEGORIES} })")

    print("\n" + "=" * 70)
    print("BASELINE QUERIES (5 normal, diverse queries -> forms the log review set)")
    print("=" * 70)
    baseline = [
        ("What are the working hours at Acme?", "acme"),
        ("What is the annual leave policy at Acme?", "acme"),
        ("What is the primary business model of Globex?", "globex"),
        ("What laptops are issued to Hooli employees?", "hooli"),
        ("What is Initech's revenue model?", "initech"),
    ]
    for q, ns in baseline:
        matches = namespaces[ns].query(q, top_k=4)
        # No LLM reachable offline; log retrieval evidence with answer left for
        # manual review focus (retrieval pattern), consistent with note field.
        log_query(q, ns, 4, matches, answer=None,
                  note="baseline retrieval-only run (no LLM call in this sandbox)")
        top = matches[0]
        print(f"\nQ: {q}  [{ns}]")
        for m in matches:
            print(f"  score={m['score']:.4f} cat={m['category_label']:<18} "
                  f"{m['source_filename']} pg{m['page_number']} len={m['char_len']}")

    print("\n" + "=" * 70)
    print("FAILURE MODE 1 -- WRONG CHUNK RETRIEVED")
    print("=" * 70)
    q1 = "What is Acme Corp's revenue?"
    matches1 = namespaces["acme"].query(q1, top_k=4)
    print(f"\nQ: {q1}")
    for m in matches1:
        print(f"  score={m['score']:.4f} cat={m['category_label']:<18} "
              f"{m['source_filename']} pg{m['page_number']} "
              f"preview={m['text'][:90]!r}")
    answer1 = (
        "Acme Corp's revenue was $98.4M in FY2024."
    )
    log_query(q1, "acme", 4, matches1, answer=answer1,
              note="Hand-authored answer against retrieved context, illustrating "
                   "the failure: it reports the business_model doc's ARR figure "
                   "($98.4M) as 'revenue', while the finance doc's actual Total "
                   "Revenue line for FY2024 is $142.6M -- a different, larger "
                   "number for a genuinely different metric. The question is "
                   "lexically ambiguous ('revenue') across two categories.")

    print("\n" + "=" * 70)
    print("FAILURE MODE 2 -- HALLUCINATION BEYOND CONTEXT")
    print("=" * 70)
    q2 = "Who is Acme Corp's Chief Financial Officer, and what is the company's customer churn rate?"
    matches2 = namespaces["acme"].query(q2, top_k=4)
    print(f"\nQ: {q2}")
    for m in matches2:
        print(f"  score={m['score']:.4f} cat={m['category_label']:<18} "
              f"{m['source_filename']} pg{m['page_number']}")
    answer2_loose = (
        "Acme Corp's CFO is Sarah Mitchell, and the company's annual customer "
        "churn rate is approximately 4.2%."
    )
    log_query(q2, "acme", 4, matches2, answer=answer2_loose,
              note="Hand-authored to illustrate the loose-prompt failure: neither "
                   "a CFO name nor a churn rate appears anywhere in the retrieved "
                   "context (or in any acme document) -- both facts are fabricated.")

    print("\n" + "=" * 70)
    print("FAILURE MODE 3 -- CONTEXT TOO LONG DEGRADES THE ANSWER")
    print("=" * 70)
    q3 = "How many days of sick leave do Acme employees get per year?"
    matches3_k4 = namespaces["acme"].query(q3, top_k=4)
    matches3_k20 = namespaces["acme"].query(q3, top_k=20)
    print(f"\nQ: {q3}  (top_k=4)")
    for m in matches3_k4:
        print(f"  score={m['score']:.4f} cat={m['category_label']:<18} len={m['char_len']}")
    print(f"\nQ: {q3}  (top_k=20, {len(matches3_k20)} chunks returned -- "
          f"most of the acme corpus)")
    for m in matches3_k20:
        print(f"  score={m['score']:.4f} cat={m['category_label']:<18} len={m['char_len']}")

    answer3_k4 = "Acme employees get 10 days of sick/medical leave per year (not permitted to carry forward, no encashment)."
    log_query(q3, "acme", 4, matches3_k4, answer=answer3_k4,
              note="Clean context (4 relevant chunks) -> correct, precise answer.")

    answer3_k20 = (
        "Acme employees receive sick leave as part of their benefits package; "
        "based on the leave policy this appears to be around 8-10 days, though "
        "the documentation also references casual leave (8 days), compensatory "
        "off, and other benefit allowances, so the exact sick leave figure is "
        "not fully clear from the combined material."
    )
    log_query(q3, "acme", 20, matches3_k20, answer=answer3_k20,
              note="Degraded: with top_k=20 the context includes finance, IT, "
                   "and business_model chunks unrelated to leave policy, plus "
                   "low-similarity HR fragments. The correct figure (10 days) "
                   "gets hedged/blurred against casual leave (8 days) and other "
                   "noise -- hand-authored to illustrate typical dilution "
                   "behavior when irrelevant chunks flood the prompt.")

    print(f"\nLog written to {LOG_PATH}")

    # ------------------------------------------------------------------
    # FIX VERIFICATION: rebuild namespaces with boilerplate stripped,
    # re-run the same 5 baseline queries + the wrong-chunk trigger query,
    # and compare top-4 category purity before vs. after.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FIX APPLIED: strip repeated header/footer boilerplate before chunking")
    print("=" * 70)
    namespaces_clean = {}
    for company in sorted(os.listdir(RAG_INPUT)):
        company_path = os.path.join(RAG_INPUT, company)
        if not os.path.isdir(company_path):
            continue
        records = load_and_chunk_company(company_path, company, clean=True)
        namespaces_clean[company] = LocalNamespace(records)

    def expected_category(question: str) -> str | None:
        return infer_category(question)

    compare_queries = baseline + [(q1, "acme")]
    before_after_log = []
    print(f"\n{'query':<55} {'top1 before':<12} {'top1 after':<12} "
          f"{'purity before':<15} {'purity after'}")
    for q, ns in compare_queries:
        exp_cat = expected_category(q)
        m_before = namespaces[ns].query(q, top_k=4)
        m_after = namespaces_clean[ns].query(q, top_k=4)
        purity_before = sum(1 for m in m_before if m["category_label"] == exp_cat) if exp_cat else None
        purity_after = sum(1 for m in m_after if m["category_label"] == exp_cat) if exp_cat else None
        print(f"{q[:53]:<55} {m_before[0]['score']:<12.4f} {m_after[0]['score']:<12.4f} "
              f"{str(purity_before)+'/4':<15} {str(purity_after)+'/4'}")
        before_after_log.append({
            "query": q, "namespace": ns, "expected_category": exp_cat,
            "top1_score_before": round(m_before[0]["score"], 4),
            "top1_score_after": round(m_after[0]["score"], 4),
            "category_purity_top4_before": purity_before,
            "category_purity_top4_after": purity_after,
        })

    with open("/sessions/modest-keen-cray/mnt/outputs/fix_comparison.json", "w") as f:
        json.dump(before_after_log, f, indent=2)
    print("\nComparison written to fix_comparison.json")


if __name__ == "__main__":
    main()
