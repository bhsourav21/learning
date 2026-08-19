# from time import sleep
from time import perf_counter
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# model_bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
model_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

load_dotenv()

PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say 'I don't know'.

Context:
{context}

Question: {question}"""


class RAGChain:

    def __init__(self, index_name: str, top_k: int = 4):
        self.top_k = top_k

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.pinecone_index = Pinecone().Index(index_name)

    def _retrieve_with_bi_encoder(self, question: str, namespace: str) -> list[str]:
        query_vector = self.embeddings.embed_query(question)
        results = self.pinecone_index.query(
            vector=query_vector,
            top_k=self.top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return [match["metadata"]["text"] for match in results["matches"]]

    @staticmethod
    def _rerank_with_cross_encoder(question: str, chunks: list[str]) -> list[tuple[float, str]]:
        pairs = [(question, chunk) for chunk in chunks]
        scores = model_cross_encoder.predict(pairs)
        return sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    @staticmethod
    def _format_chunks(chunks: list[str]) -> str:
        return "\n\n".join(
            f"[Chunk {i + 1}] {chunk.strip()}"
            for i, chunk in enumerate(chunks)
        )

    def query(self, question: str, namespace: str) -> str:
        chunks = self._retrieve_with_bi_encoder(question, namespace)

        t0 = perf_counter()
        reranked = self._rerank_with_cross_encoder(question, chunks)
        rerank_latency = perf_counter() - t0

        original_rank = {chunk: i + 1 for i, chunk in enumerate(chunks)}

        col_w = 70
        sep   = f"+{'─'*14}+{'─'*14}+{'─'*8}+{'─'*(col_w+2)}+"
        header = f"| {'After Rerank':^12} | {'Before Rerank':^12} | {'Score':^6} | {'Chunk Preview':<{col_w}} |"

        print(f"\nQuery          : {question}")
        print(f"Namespace      : {namespace}")
        print(f"Rerank latency : {rerank_latency:.4f}s\n")
        print(sep)
        print(header)
        print(sep)
        for new_rank, (score, chunk) in enumerate(reranked, 1):
            old_rank = original_rank[chunk]
            preview  = chunk.strip().replace("\n", " ")
            for line_start in range(0, max(1, len(preview)), col_w):
                segment = preview[line_start:line_start + col_w]
                if line_start == 0:
                    print(f"| {new_rank:^12} | {old_rank:^12} | {score:^6.3f} | {segment:<{col_w}} |")
                else:
                    print(f"| {'':^12} | {'':^12} | {'':^6} | {segment:<{col_w}} |")
            print(sep)


if __name__ == "__main__":
    rag = RAGChain(index_name="genai-learning-index", top_k=5)
    rag.query(
        question="What is the working hour in acme?",
        namespace="acme",
    )