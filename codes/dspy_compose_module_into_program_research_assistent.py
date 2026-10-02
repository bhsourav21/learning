import dspy
from typing import Literal

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

class PlanSearches(dspy.Signature):
    """Turn a research question into 2-4 focused search queries.
    Each query should target a different facet of the question so the
    combined results cover it fully, rather than repeating one angle.
    """
    question: str = dspy.InputField()
    queries: list[str] = dspy.OutputField()

class AnswerWithSources(dspy.Signature):
    """Answer the question using only the supplied context passages.
    If the context does not contain the answer, say so plainly rather
    than guessing. Cite passages by their number, like [2].
    """
    question: str = dspy.InputField()
    context: list[str] = dspy.InputField()
    answer: str = dspy.OutputField()
    confidence: Literal["high", "medium", "low"] = dspy.OutputField()

# ---------------------------------------------------------------- retrieval
# Stand-in for a real document store. In a production program this corpus
# would come from a database, a crawl, or a file loader.
CORPUS = [
    "DSPy signatures declare the input and output fields of an LM call "
    "without pinning down the prompt wording itself.",
    "dspy.ChainOfThought wraps a signature and adds a reasoning field, so "
    "the model is asked to think step by step before answering.",
    "A dspy.Module composes other modules. Submodules assigned in __init__ "
    "are tracked as parameters, which is what lets optimizers tune them.",
    "dspy.Embedder wraps an embedding model. Given a string identifier it "
    "routes through LiteLLM; given a callable it uses that function directly.",
    "dspy.retrievers.Embeddings builds an in-memory vector index over a "
    "corpus and returns the k nearest passages for a query.",
    "Optimizers such as MIPROv2 and BootstrapFewShot search over prompts "
    "and few-shot demonstrations to improve a program against a metric.",
    "Ollama serves local models over an OpenAI-compatible HTTP API on port "
    "11434, so LiteLLM can reach it with an ollama/ or ollama_chat/ prefix.",
    "Retrieval-augmented generation grounds an answer in retrieved passages "
    "so the model cites evidence instead of relying on parametric memory.",
]

# dspy.Embedder is the embedding *model*; it speaks LiteLLM, so a local
# Ollama embedding model is addressed with the "ollama/" prefix.
embedder = dspy.Embedder(
    "ollama/nomic-embed-text",
    api_base="http://localhost:11434",
    api_key="",
    batch_size=32,
)

# dspy.Embeddings is the *retriever* built on top of that embedder. It
# embeds CORPUS once, here, and then answers queries from memory.
retriever = dspy.Embeddings(corpus=CORPUS, embedder=embedder, k=3)

def my_vector_search(query: str) -> list[str]:
    """Return the passages from CORPUS most similar to `query`."""
    return retriever(query).passages

# ---------------------------------------------------------------- program
class ResearchAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.ChainOfThought(PlanSearches)
        self.answer = dspy.ChainOfThought(AnswerWithSources)
        # self.plan = dspy.Predict(PlanSearches)
        # self.answer = dspy.Predict(AnswerWithSources)

    def forward(self, question: str):
        queries = self.plan(question=question).queries

        # for query in queries:
        #     print(f"query:{query}")
        # print("****************************")

        # Fan out over the planned queries, keeping passage order stable
        # and dropping duplicates that several queries both retrieve.
        seen: set[str] = set()
        context: list[str] = []
        for q in queries:
            for passage in my_vector_search(q):
                print()
                print(f"query:{q}")
                print(f"passage:{passage}")
                print()
                if passage not in seen:
                    seen.add(passage)
                    context.append(passage)
            print("------------------------")

        out = self.answer(question=question, context=context)
        return dspy.Prediction(
            queries=queries,
            context=context,
            answer=out.answer,
            confidence=out.confidence,
        )

if __name__ == "__main__":
    program = ResearchAssistant()
    result = program(question="How does DSPy let you compose and then optimize a program?")

    print("queries:")
    for q in result.queries:
        print(f"  - {q}")

    print(f"\nretrieved {len(result.context)} passages:")
    for n, passage in enumerate(result.context, start=1):
        print(f"  [{n}] {passage[:80]}...")

    print(f"\nanswer:\n{result.answer}")
    print(f"\nconfidence:{result.confidence}")
