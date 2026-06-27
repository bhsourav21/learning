from langchain_community.document_loaders import PyPDFLoader

# ■■ Step 1: Load the PDF ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
path = '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/docs'
loader = PyPDFLoader(f"{path}/AI_Ecosystem_4Layer_Stack.pdf")
pages = loader.load()

# ■■ Step 2: Print number of pages ■■■■■■■■■■■■■■■■■■■■■■■■■■■■
print(f'Total pages: {len(pages)}')
print()

# ■■ Step 3: First 500 characters of page 2 (index 1) ■■■■■■■■■
page2 = pages[1]
print('--- Page 2 content (first 500 chars) ---')
print(page2.page_content[:500])
print()

# ■■ Step 4: Print metadata ■■■■■■■■■■■■■■■■■■■■■■■■■■■■
i = 0
for page in pages:
    i = i + 1
    print(f"metadata of page# {i}:")
    print(page.metadata)
    print()