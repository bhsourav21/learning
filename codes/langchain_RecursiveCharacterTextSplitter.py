from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load
path = '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/docs'
loader = PyPDFLoader(f"{path}/AI_Ecosystem_4Layer_Stack.pdf")
pages = loader.load()

output_path = '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/codes/output/RecursiveCharacterTextSplitter_output.txt'

def write_chunks(f, chunks, chunk_size, chunk_overlap):
    f.write(f"chunk size {chunk_size}, chunk overlap {chunk_overlap}\n")
    f.write("====================================\n")
    for i, chunk in enumerate(chunks, start=1):
        f.write(f"chunk#{i}\n")
        f.write("---------------\n")
        f.write(str(chunk) + "\n")
        f.write("\n")

# 2. Chunk
splitter_500 = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)
chunks_500 = splitter_500.split_documents(pages)

splitter_200 = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    add_start_index=True,
)
chunks_200 = splitter_200.split_documents(pages)

splitter_1000 = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    add_start_index=True,
)
chunks_1000 = splitter_1000.split_documents(pages)

with open(output_path, 'w', encoding='utf-8') as f:
    write_chunks(f, chunks_500, 500, 50)
    write_chunks(f, chunks_200, 200, 50)
    write_chunks(f, chunks_1000, 1000, 50)

print(f"Output written to {output_path}")
