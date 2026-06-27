import re
from langchain_community.document_loaders import PyPDFLoader

# ■■ Step 1: Load the PDF ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
path = '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/docs'
loader = PyPDFLoader(f"{path}/AI_Ecosystem_4Layer_Stack.pdf")
pages = loader.load()

def split_by_paragraph(text: str, min_len: int = 50) -> list[str]:
    """Split text on blank lines; drop chunks shorter than min_len."""
    chunks = re.split(r'\n\n+', text)
    return [c.strip() for c in chunks if len(c.strip()) >= min_len]

chunks = [chunk for page in pages for chunk in split_by_paragraph(page.page_content)]
print(f"Paragraphs found: {len(chunks)}")
print()

for i, ch in enumerate(chunks, start=1):
    print(f"\nChunk {i} ({len(ch)} chars):\n{ch[:120]}...")