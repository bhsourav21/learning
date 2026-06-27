from langchain_community.document_loaders import WebBaseLoader

url = 'https://en.wikipedia.org/wiki/Vector_database'

loader = WebBaseLoader(
    web_paths=[url],
    requests_kwargs={
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    },
)

docs = loader.load()

# ■■ Inspect the Document object ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
print(f'Documents loaded : {len(docs)}') # 1
print(f'Content length : {len(docs[0].page_content)} chars')
print()

print('\npage_content (first 300 chars):')
print(docs[0].page_content[:300])
print()

print('\nmetadata:')
print(docs[0].metadata)
print()