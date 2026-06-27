from langchain_community.document_loaders import UnstructuredMarkdownLoader

path = '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/docs'

# single mode
loader = UnstructuredMarkdownLoader(f"{path}/rag_notes.md", mode='single')
docs = loader.load()
print("single mode")
print("-------------")
print(f'Documents: {len(docs)}') # 1
print()

print("page content:")
print(docs[0].page_content[:200])
print()

print("metadata:")
print(docs[0].metadata)
# {'source': 'rag_notes.md'}

# elements mode (recommended for RAG)
loader = UnstructuredMarkdownLoader(f"{path}/rag_notes.md", mode='elements')
docs = loader.load()
print("element mode")
print("--------------")
print(f'Elements: {len(docs)}') # e.g. 8-12
print()

# First element is always the Title
print("page content:")
for doc in docs:
    print(doc.page_content) # RAG Pipeline Notes
    print()

print("metadata:")
print(docs[0].metadata)
print()