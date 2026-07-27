"""
RAGPipeline v2 -- adds structured logging + the 3 fixes from the failure-mode
analysis (see ragas_eval_failure_modes.pdf) on top of rag_pipeline.py:

  1. Structured JSONL logging of every query: retrieved chunk ids, similarity
     scores, category labels, and the final answer.
  2. Category-aware retrieval: a lightweight keyword router infers which
     document category (business_model / finance / hr_policy /
     it_infrastructure) a question targets and applies a Pinecone metadata
     filter, instead of relying on raw embedding similarity across all
     categories at once. Ambiguous questions (matching 2+ categories, e.g.
     "revenue") are routed to the union of categories rather than guessed at.
  3. Similarity-score threshold + hard cap on context size, so requesting a
     larger top_k for recall doesn't automatically flood the prompt with
     low-relevance chunks.
  4. A stricter, hallucination-resistant prompt (temperature=0 + explicit
     "don't infer/estimate" instruction + source attribution requirement).

This file is a drop-in replacement for RAGPipeline's query/generate_answer
path. Ingestion (load_and_chunk / upsert_company / run) is unchanged from
rag_pipeline.py and is imported, not duplicated.
"""
import json
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from rag_pipeline import RAGPipeline

load_dotenv()

LOG_PATH = os.path.join(os.path.dirname(__file__), "rag_query_log.jsonl")

# Maps a category to keywords/phrases whose presence in a question is a
# reasonably strong signal the answer lives in that category's chunks.
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
    # "revenue" alone is deliberately left out of every list above: it is
    # genuinely ambiguous between business_model (ARR) and finance (Total
    # Revenue) in this corpus, and is handled explicitly in infer_categories().
}


def infer_categories(question: str) -> list[str]:
    """Returns the category (or categories, if genuinely ambiguous) a question
    most likely targets. Returns [] if no keyword signal is found, in which
    case the caller should fall back to an unfiltered search."""
    q = question.lower()
    hits = {
        cat: sum(1 for kw in kws if re.search(r"\b" + re.escape(kw.strip()) + r"\b", q))
        for cat, kws in KEYWORD_ROUTER.items()
    }
    hits = {c: n for c, n in hits.items() if n > 0}

    winners: set[str] = set()
    if hits:
        top = max(hits.values())
        winners = {c for c, n in hits.items() if n == top}

    if re.search(r"\brevenue\b", q):
        # "revenue" alone is ambiguous between business_model's ARR and
        # finance's Total Revenue -> widen instead of silently guessing
        winners |= {"business_model", "finance"}

    return sorted(winners)


STRICT_PROMPT_TEMPLATE = """\
You are a helpful assistant answering questions about {company}'s internal \
documentation using only the context below.

Rules:
- Answer only using facts explicitly stated in the context. Do not infer, \
estimate, guess, or draw on outside/general knowledge, even if it seems \
plausible.
- If the specific fact requested is not explicitly present in the context, \
respond exactly: "I don't know based on the available documents."
- If two chunks appear to give different numbers for what might be the same \
thing (e.g. ARR vs. Total Revenue), do not silently pick one -- name which \
metric/line-item each number is and where it's from.

Context:
{context}

Question: {question}"""


class RAGPipelineV2(RAGPipeline):

    def _log_query(self, query, namespace, top_k_requested, matches, answer,
                    category_filter=None):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "namespace": namespace,
            "top_k_requested": top_k_requested,
            "category_filter": category_filter,
            "retrieved_chunks": [
                {
                    "rank": i + 1,
                    "id": m.id,
                    "score": round(m.score, 4),
                    "category_label": m.metadata.get("category_label"),
                    "source_filename": m.metadata.get("source_filename"),
                    "page_number": m.metadata.get("page_number"),
                    "char_len": len(m.metadata.get("text", "")),
                    "text_preview": m.metadata.get("text", "")[:160].replace("\n", " "),
                }
                for i, m in enumerate(matches)
            ],
            "final_answer": answer,
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    # Fix 2: category-aware retrieval. Falls back to an unfiltered search if
    # the router has no opinion, so behavior is unchanged for out-of-corpus
    # or exploratory questions.
    def query_routed(self, text: str, namespace: str, top_k: int = 4):
        categories = infer_categories(text)
        if not categories:
            results = self.index.query(
                vector=self.get_embeddings([text])[0],
                top_k=top_k, namespace=namespace, include_metadata=True,
            )
            cat_filter = None
        else:
            pine_filter = (
                {"category_label": {"$eq": categories[0]}} if len(categories) == 1
                else {"category_label": {"$in": categories}}
            )
            results = self.index.query(
                vector=self.get_embeddings([text])[0],
                top_k=top_k, namespace=namespace, include_metadata=True,
                filter=pine_filter,
            )
            cat_filter = categories
        return results, cat_filter

    # Fix 3: score-threshold cap so a larger top_k (recall) can't flood the
    # prompt with low-relevance chunks (context-too-long failure mode).
    @staticmethod
    def _cap_context(results, score_threshold: float = 0.15, max_chunks: int = 6):
        kept = [m for m in results.matches if m.score >= score_threshold]
        return kept[:max_chunks] if kept else results.matches[:1]  # never return zero chunks

    # Fix 4: stricter, hallucination-resistant prompt + temperature=0.
    def generate_answer_v2(self, question: str, company: str, matches) -> str:
        context = "\n\n".join(
            f"[Chunk {rank} | {m.metadata.get('category_label')} | "
            f"{m.metadata.get('source_filename')} pg{m.metadata.get('page_number')}] "
            f"{m.metadata.get('text', '').strip()}"
            for rank, m in enumerate(matches, start=1)
        )
        prompt = STRICT_PROMPT_TEMPLATE.format(
            company=company, context=context, question=question
        )
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    # End-to-end: route -> retrieve -> cap -> generate -> log
    def ask(self, question: str, namespace: str, top_k: int = 4):
        results, cat_filter = self.query_routed(question, namespace, top_k=top_k)
        capped_matches = self._cap_context(results)
        answer = self.generate_answer_v2(question, namespace, capped_matches)
        self._log_query(question, namespace, top_k, capped_matches, answer,
                        category_filter=cat_filter)
        return answer


if __name__ == "__main__":
    pipeline = RAGPipelineV2(
        rag_input_path="/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/rag_input",
        index_name="genai-learning-index",
    )

    demo_questions = [
        ("What are the working hours at Acme?", "acme"),
        ("What is the annual leave policy at Acme?", "acme"),
        ("What is the primary business model of Globex?", "globex"),
        ("What laptops are issued to Hooli employees?", "hooli"),
        ("What is Initech's revenue model?", "initech"),
        ("What is Acme Corp's revenue?", "acme"),
        ("Who is Acme Corp's Chief Financial Officer, and what is the company's customer churn rate?", "acme"),
    ]
    for q, ns in demo_questions:
        answer = pipeline.ask(q, namespace=ns, top_k=4)
        print(f"\nQ ({ns}): {q}\nA: {answer}\n{'-'*60}")

    print(f"\nStructured logs written to {LOG_PATH}")
