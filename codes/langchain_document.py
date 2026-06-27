from langchain_core.documents import Document

# A Document has exactly two fields
doc = Document(
    page_content = 'The actual text extracted from the source.',
    metadata = {
        'source': '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/docs/AI_Ecosystem_4Layer_Stack.pdf', # always present
        'page' : 0, # loader-specific
        # ... additional loader-specific keys
    }
)
# Access fields
print(doc.page_content[:100])
print(doc.metadata) # first 100 chars of text
# {'source': ..., 'page': 0}