from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ■■ Step 1: Load the PDF ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
path = '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/docs'
loader = PyPDFLoader(f"{path}/AI_Ecosystem_4Layer_Stack.pdf")
pages = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile", # recommended default
    breakpoint_threshold_amount=95, # top 5% drops = boundaries
)

chunks_list = []
for page in pages:
    chunks = chunker.split_text(page.page_content)
    for chunk in chunks:
        chunks_list.append(chunk)

print(f"Number of Chunks: {len(chunks_list)}")

for i, chunk in enumerate(chunks_list, start=1):
    print(f"Chunk {i} ({len(chunk)} chars): {chunk[:80]}...")
    print()