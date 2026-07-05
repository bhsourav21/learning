from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "bhsourav17-pinecone-index"
NAMESPACE  = "AI_Ecosystem_4Layer_Stack.pdf"

pc    = Pinecone()
index = pc.Index(INDEX_NAME)

stats_before = index.describe_index_stats()
count_before = stats_before.namespaces.get(NAMESPACE, {}).get("vector_count", 0)
print(f"Vectors in '{NAMESPACE}' before delete: {count_before}")

index.delete(delete_all=True, namespace=NAMESPACE)
print(f"Deleted all records in namespace '{NAMESPACE}'.")

stats_after = index.describe_index_stats()
count_after = stats_after.namespaces.get(NAMESPACE, {}).get("vector_count", 0)
print(f"Vectors in '{NAMESPACE}' after delete: {count_after}")
