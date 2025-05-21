import chromadb
from app.core.config import settings

class EnhancedVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        self.collection_name = "enhanced_docs"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, ids, embeddings, metadatas):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding, n_results=3):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
