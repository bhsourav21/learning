from time import sleep
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

    # Returns a vector store retriever scoped to a specific company namespace
    def _get_retriever(self, namespace: str):
        vector_store = PineconeVectorStore(
            index=self.pinecone_index,
            embedding=self.embeddings,
            namespace=namespace,
            text_key="text",
        )
        return vector_store.as_retriever(search_kwargs={"k": self.top_k})

    # Joins retrieved Document objects into a numbered context block
    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(
            f"[Chunk {i + 1}] {doc.page_content.strip()}"
            for i, doc in enumerate(docs)
        )

    # Builds the LCEL chain: retrieve → format → prompt → LLM → parse
    def _build_chain(self, namespace: str):
        retriever = self._get_retriever(namespace)
        return (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # Runs the full RAG chain and streams the answer token by token
    def query(self, question: str, namespace: str) -> str:
        chain = self._build_chain(namespace)
        print(f"\nQuery    : {question}")
        print(f"Namespace: {namespace}")
        print(f"\nAnswer:")
        # chain.invoke returns the full answer at once
        # answer = chain.invoke(question)
        # chain.stream yields tokens incrementally — print each without newline as it arrives
        tokens = []
        for token in chain.stream(question):
            print(token, end="", flush=True)
            sleep(0.10)
            tokens.append(token)
        print("\n")
        return "".join(tokens)


if __name__ == "__main__":
    rag = RAGChain(index_name="genai-learning-index", top_k=4)
    rag.query(
        question="What is the working hour in acme?",
        namespace="acme",
    )