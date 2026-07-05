from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "genai-learning-index"
NAMESPACES = ["acme", "globex", "hooli", "initech", "umbrella"]


class PineconeIndexCleaner:

    def __init__(self, index_name: str):
        self.index = Pinecone().Index(index_name)
        self.index_name = index_name

    # Returns the vector count per namespace from the index stats
    def get_stats(self) -> dict:
        return self.index.describe_index_stats()

    # Deletes all vectors in a single namespace
    def delete_namespace(self, namespace: str):
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"  Deleted all vectors in namespace: {namespace}")

    # Iterates over all namespaces and deletes every vector in each
    def delete_all(self, namespaces: list[str]):
        print(f"\nCleaning index: {self.index_name}")
        print(f"Namespaces to delete: {namespaces}\n")

        stats_before = self.get_stats()
        print(f"Stats before deletion:\n{stats_before}\n")

        for namespace in namespaces:
            self.delete_namespace(namespace)

        stats_after = self.get_stats()
        print(f"\nStats after deletion:\n{stats_after}")
        print("\nAll records deleted.")


if __name__ == "__main__":
    cleaner = PineconeIndexCleaner(index_name=INDEX_NAME)
    cleaner.delete_all(namespaces=NAMESPACES)
