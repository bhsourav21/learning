from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "bhsourav17-pinecone-index"

pc = Pinecone()
index = pc.Index(INDEX_NAME)

stats_before = index.describe_index_stats()
total = stats_before.total_vector_count
print(f"Vectors before delete: {total}")

index.delete(delete_all=True)
print("All records deleted.")

stats_after = index.describe_index_stats()
print(f"Vectors after delete: {stats_after.total_vector_count}")
