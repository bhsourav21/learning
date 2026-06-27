import re
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

def hybrid_chunk(
    text: str,
    section_pattern: str = r"\n(?=#{1,6}\s)|\n\n+",
    min_chars: int = 100,
    max_chars: int = 2000,
    sem_threshold_type: str = "percentile",
    sem_threshold_amt: float = 95,
    ) -> list[dict]:

    """
    Phase 1: regex structural split.
    Phase 2: semantic sub-split for oversized sections.
    Returns list of {'text', 'chars', 'method'} dicts.
    """

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    sem_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=sem_threshold_type,
        breakpoint_threshold_amount=sem_threshold_amt,
    )

    # Phase 1
    raw_sections_final = []
    final_chunks = []

    raw_sections = [s.strip() for s in re.split(section_pattern, text) if s.strip()]
    for raw_section in raw_sections:
        raw_sections_final.append(raw_sections)
    
    for section in raw_sections:
        if len(section) < min_chars:
            continue # too short, skip
        elif len(section) <= max_chars:
            final_chunks.append({ # just right
                "text": section, "chars": len(section), "method": "regex"
            })
        else:
            # Phase 2: semantic sub-split
            sub = sem_chunker.split_text(section)
            for s in sub:
                if len(s.strip()) >= min_chars:
                    final_chunks.append({
                        "text": s.strip(),
                        "chars": len(s.strip()),
                        "method": "regex+semantic",
                    })

    return final_chunks


chunks_final = []
for page in pages:
    chunks = hybrid_chunk(page.page_content)
    for chunk in chunks:
        chunks_final.append(chunk)

print(chunks_final)

for i, ch in enumerate(chunks_final):
    print(f"[{ch['method']:15s}] Chunk {i+1}: {ch['chars']} chars")

# The most powerful production strategy combines both techniques: use regex to create coarse structural chunks
# (sections), then apply semantic splitting within each section to further refine at topic-shift boundaries. This
# keeps cost low while ensuring semantically coherent output.

# Step 1 — Regex pre-split
# Split the document into sections using heading or paragraph regex.
# Each section becomes an independent unit.

# Step 2 — Size filter 
# Short sections (< min_chars) are kept as-is. Sections within the target
# range are used directly. Only long sections (> max_chars) proceed to
# Step 3.

# Step 3 — Semantic sub-split
# Apply SemanticChunker (or the from-scratch splitter) to each oversized
# section, breaking it further at topic-shift boundaries.

# Step 4 — Merge tiny chunks
# Any chunk under min_chars remaining after semantic splitting is merged
# with its neighbour to avoid orphan chunks that hurt retrieval.

# Step 5 — Deduplicate & index
# Remove near-duplicate chunks (cosine sim > 0.98) and attach metadata
# (source, section title, chunk index) before embedding.