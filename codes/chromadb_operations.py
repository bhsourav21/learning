import chromadb
# Data is saved in ./chroma_store on your disk

client = chromadb.PersistentClient(path="./chroma_store")

# Verify it's working
# print(client.heartbeat()) # returns a timestamp

# get_or_create avoids errors if the collection already exists
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"} # use cosine distance (recommended)
)

# finance (docs 1-5), space (6-10), health (11-15), technology (16-20)
documents = [
    "Inflation rates rose sharply in Q3 due to energy prices.",
    "The Federal Reserve raised interest rates by 75 basis points.",
    "Stock markets fell amid recession fears and dollar strength.",
    "Gold is traditionally seen as a hedge against inflation.",
    "Venture capital funding in AI startups doubled this year.",
    "NASA launched a new Mars rover to search for microbial life.",
    "The James Webb telescope revealed galaxies from 13 billion years ago.",
    "SpaceX successfully landed its reusable Starship booster.",
    "Astronomers detected gravitational waves from merging black holes.",
    "A new exoplanet in the habitable zone was found 40 light-years away.",
    "Researchers developed a vaccine using mRNA technology for cancer.",
    "A clinical trial showed 90% efficacy for the new diabetes drug.",
    "CRISPR gene editing corrected a hereditary blindness condition.",
    "A study linked gut microbiome diversity to reduced anxiety levels.",
    "AI models are now detecting tumours earlier than human radiologists.",
    "The new Python 3.12 release brings faster startup and better errors.",
    "Rust is gaining popularity for systems programming due to safety.",
    "Kubernetes 1.28 improved node resource management significantly.",
    "WebAssembly is enabling high-performance apps in the browser.",
    "Open-source LLMs like Llama 3 are closing the gap with GPT-4.",
]

metadatas = [
    {"category": "finance", "source": "Reuters", "date": "2024-09-10"},
    {"category": "finance", "source": "Bloomberg", "date": "2024-09-15"},
    {"category": "finance", "source": "CNBC", "date": "2024-09-20"},
    {"category": "finance", "source": "FT", "date": "2024-09-22"},
    {"category": "finance", "source": "TechCrunch", "date": "2024-09-25"},
    {"category": "space", "source": "NASA", "date": "2024-08-01"},
    {"category": "space", "source": "ESA", "date": "2024-08-05"},
    {"category": "space", "source": "SpaceX", "date": "2024-08-10"},
    {"category": "space", "source": "Nature", "date": "2024-08-15"},
    {"category": "space", "source": "NASA", "date": "2024-08-20"},
    {"category": "health", "source": "NIH", "date": "2024-07-01"},
    {"category": "health", "source": "Lancet", "date": "2024-07-10"},
    {"category": "health", "source": "NEJM", "date": "2024-07-15"},
    {"category": "health", "source": "Nature", "date": "2024-07-20"},
    {"category": "health", "source": "Science", "date": "2024-07-25"},
    {"category": "technology", "source": "GitHub", "date": "2024-06-01"},
    {"category": "technology", "source": "Mozilla", "date": "2024-06-10"},
    {"category": "technology", "source": "CNCF", "date": "2024-06-15"},
    {"category": "technology", "source": "W3C", "date": "2024-06-20"},
    {"category": "technology", "source": "Meta AI", "date": "2024-06-25"},
]

ids = [f"doc_{i}" for i in range(1, 21)]

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids   
)

print(f"collection size:{collection.count()}")

result_fields = ["documents", "distances", "metadatas"]

# Query 1: Finance topic
results_1 = collection.query(
    query_texts=["impact of interest rates on stock market"],
    n_results=3,
    include=result_fields
)

print("results_1:")
print(results_1)
print()

# Query 2: Space topic
results_2 = collection.query(
    query_texts=["discoveries about distanct galaxies and planets"],
    n_results=3,
    include=result_fields
)

print("results_2:")
print(results_2)
print()

# Query 3: Health/AI crossover
results_3 = collection.query(
    query_texts=["AI applications in medical diagnosis"],
    n_results=3,
    include=result_fields
)

print("results_3:")
print(results_3)
print()


# Metadata filter
result_filter = collection.query(
    query_texts=["impact of interest rates on the economy"],
    n_results=6,
    where={"category": "finance"}, # shorthand for $eq
    include=["documents", "distances", "metadatas"]
)

print("result_filter:")
print(result_filter)
print()

# Output:
# result_filter:
# {'ids': [['doc_2', 'doc_1', 'doc_4', 'doc_3', 'doc_5']], 'embeddings': None, 'documents': [['The Federal Reserve raised interest rates by 75 basis points.', 'Inflation rates rose sharply in Q3 due to energy prices.', 'Gold is traditionally seen as a hedge against inflation.', 'Stock markets fell amid recession fears and dollar strength.', 'Venture capital funding in AI startups doubled this year.']], 'uris': None, 'included': ['documents', 'distances', 'metadatas'], 'data': None, 'metadatas': [[{'category': 'finance', 'date': '2024-09-15', 'source': 'Bloomberg'}, {'date': '2024-09-10', 'source': 'Reuters', 'category': 'finance'}, {'date': '2024-09-22', 'category': 'finance', 'source': 'FT'}, {'source': 'CNBC', 'category': 'finance', 'date': '2024-09-20'}, {'date': '2024-09-25', 'source': 'TechCrunch', 'category': 'finance'}]], 'distances': [[0.5541154146194458, 0.6618324518203735, 0.7384829521179199, 0.7385424375534058, 0.8202033042907715]]}

# Observation: Even though n_results=6 is passed,since there are only 5 documents 
# where={"category": "finance"}, it returned 5 records. Hence, it proves the filter is working as expected.