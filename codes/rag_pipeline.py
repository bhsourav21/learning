import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()


class RAGPipeline:
    CATEGORIES = ["business_model", "finance", "hr_policy", "it_infrastructure"]

    def __init__(
        self,
        rag_input_path: str,
        index_name: str,
        batch_size: int = 100,
    ):
        self.rag_input_path = rag_input_path
        self.batch_size = batch_size

        self.openai_client = OpenAI()
        self.index = Pinecone().Index(index_name)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            add_start_index=True,
        )

    # Derives the document category from the filename stem (e.g. "acme_hr_policy" → "hr_policy")
    @staticmethod
    def get_category_label(filename: str) -> str:
        stem = os.path.splitext(filename)[0]
        for cat in RAGPipeline.CATEGORIES:
            if stem.endswith(f"_{cat}"):
                return cat
        return "unknown"

    # Calls the OpenAI embeddings API and returns one vector per input text
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self.openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
        )
        return [item.embedding for item in response.data]

    # Loads all PDFs in a company directory and splits them into chunks
    def load_and_chunk(self, company_path: str):
        pages = []
        for pdf_file in sorted(os.listdir(company_path)):
            if not pdf_file.endswith(".pdf"):
                continue
            loader = PyPDFLoader(os.path.join(company_path, pdf_file))
            pages.extend(loader.load())
        return self.splitter.split_documents(pages)

    # Constructs the vector dicts (id + metadata) for every chunk; values filled later during upsert
    def build_vectors(self, company: str, chunks: list) -> list[dict]:
        vectors = []
        for i, chunk in enumerate(chunks):
            source_filename = os.path.basename(chunk.metadata.get("source", ""))
            vectors.append({
                "id": f"{company}-chunk-{i}",
                "values": None,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", 0),
                    "start_index": chunk.metadata.get("start_index", 0),
                    "source_filename": source_filename,
                    "page_number": chunk.metadata.get("page", 0) + 1,
                    "category_label": self.get_category_label(source_filename),
                },
            })
        return vectors

    # Embeds chunks in batches and upserts them into the company's Pinecone namespace
    def upsert_company(self, company: str, chunks: list, vectors: list[dict]):
        for batch_start in range(0, len(vectors), self.batch_size):
            batch = vectors[batch_start : batch_start + self.batch_size]
            texts = [chunks[batch_start + j].page_content for j in range(len(batch))]
            embeddings = self.get_embeddings(texts)
            for vec, emb in zip(batch, embeddings):
                vec["values"] = emb
            self.index.upsert(vectors=batch, namespace=company)
            print(f"[{company}] Upserted chunks {batch_start + 1}–{batch_start + len(batch)}")

    # Iterates over all company directories and runs the full ingest pipeline for each
    def run(self):
        for company in sorted(os.listdir(self.rag_input_path)):
            company_path = os.path.join(self.rag_input_path, company)
            if not os.path.isdir(company_path):
                continue

            chunks = self.load_and_chunk(company_path)
            vectors = self.build_vectors(company, chunks)
            self.upsert_company(company, chunks, vectors)
            print(f"[{company}] Done. Total chunks: {len(vectors)}\n")

        print("All companies processed.")

    # Builds a prompt from retrieved chunks and calls GPT to produce a grounded answer
    def generate_answer(self, question: str, results) -> str:
        context = "\n\n".join(
            f"[Chunk {rank}] {m.metadata.get('text', '').strip()}"
            for rank, m in enumerate(results.matches, start=1)
        )
        prompt = (
            "You are a helpful assistant. Answer the question using only the context below.\n"
            "If the answer is not in the context, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content.strip()
        print(f"\nAnswer:\n{answer}\n")
        return answer

    # Embeds the query text, searches the given namespace, and prints ranked results
    def query(self, text: str, namespace: str, top_k: int = 4):
        embedding = self.get_embeddings([text])[0]
        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        print(f"\nQuery    : {text}")
        print(f"Namespace: {namespace}\n")
        for rank, m in enumerate(results.matches, start=1):
            meta = m.metadata
            snippet = meta.get("text", "").strip().replace("\n", " ")[:260]
            print(f"[{rank}] score={m.score:.4f}  pg={meta.get('page_number')}")
            print(f"    {snippet}\n")
        return results


if __name__ == "__main__":
    pipeline = RAGPipeline(
        rag_input_path="/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/rag_input",
        index_name="genai-learning-index",
        batch_size=100,
    )
    pipeline.run()

    question = "What is the working hour in acme?"
    results = pipeline.query(text=question, namespace="acme", top_k=4)
    pipeline.generate_answer(question=question, results=results)