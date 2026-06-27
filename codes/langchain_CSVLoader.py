from langchain_community.document_loaders.csv_loader import CSVLoader

path = '/Users/souravbhattacharya/Documents/Code_Projects/AI_study/learning/docs'

loader = CSVLoader(file_path=f"{path}/test.csv")
docs = loader.load()

print("docs:")
for doc in docs:
    print(doc)
    print()

print(f'Documents loaded: {len(docs)}') # 4 (one per data row)