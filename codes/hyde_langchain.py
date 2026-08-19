from langchain_classic.chains import HypotheticalDocumentEmbedder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

load_dotenv()

client = OpenAI()

base_embeddings= OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.7)

corpus = [
    "It usually takes between 30 minutes and two hours to "
    "remove a wisdom tooth.",
    "The COVID-19 pandemic significantly impacted mental "
    "health, increasing depression and anxiety.",
    "Humans have used fire for approximately 800,000 years.",
    "Milvus is a cloud-based database for vector storage.",
]

prompt= PromptTemplate(
    input_variables=["question"],
    template="Write a short passage that answers the "
    "question.\nQuestion: {question}\nPassage:",
)

llm_chain= llm_chain = prompt | llm | StrOutputParser()

hyde_embeddings= HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=base_embeddings,
)

# Produces one averaged embedding, ready to hand to any
# vector store's similarity_search_by_vector() method.
def hyde_vec(question: str):
    return hyde_embeddings.embed_query("What is Milvus?")

def embed(texts, model="text-embedding-3-small"):
    resp = client.embeddings.create(input=texts, model=model)
    return np.array([d.embedding for d in resp.data])

def hyde_search(query, corpus_vecs, top_k=3):
    q_vec = hyde_vec(query)
    sims = corpus_vecs @ q_vec / (
        np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(q_vec)
    )
    ranked = np.argsort(-sims)[:top_k]
    return [(corpus[i], float(sims[i])) for i in ranked]

corpus_vecs = embed(corpus)
results = hyde_search("What is Milvus?", corpus_vecs)
for text, score in results:
    print(f"{score:.3f} {text}")