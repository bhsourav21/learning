from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ■■ Step 1: Load the PDF ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
path = '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/docs'
loader = PyPDFLoader(f"{path}/AI_Ecosystem_4Layer_Stack.pdf")
pages = loader.load()

# ■■ Step 2: Chunk ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
splitter_200 = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    add_start_index=True,
)
chunks_200 = splitter_200.split_documents(pages)

# ■■ Step 3: Upsert into Pinecone ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
INDEX_NAME = "bhsourav17-pinecone-index"
BATCH_SIZE = 100

openai_client = OpenAI()
pc = Pinecone()                  # reads PINECONE_API_KEY from env automatically
index = pc.Index(INDEX_NAME)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]

vectors = []
for i, chunk in enumerate(chunks_200):
    vectors.append({
        "id": f"chunk-{i}",
        "values": None,          # filled in batch below
        "metadata": {
            "text": chunk.page_content,
            "source": chunk.metadata.get("source", ""),
            "page": chunk.metadata.get("page", 0),
            "start_index": chunk.metadata.get("start_index", 0),
        }
    })

# Embed and upsert in batches
for batch_start in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[batch_start : batch_start + BATCH_SIZE]
    texts = [chunks_200[batch_start + j].page_content for j in range(len(batch))]
    embeddings = get_embeddings(texts)
    for vec, emb in zip(batch, embeddings):
        vec["values"] = emb
    index.upsert(vectors=batch)
    print(f"Upserted chunks {batch_start + 1} – {batch_start + len(batch)}")

print(f"\nDone. Total chunks upserted: {len(vectors)}")
