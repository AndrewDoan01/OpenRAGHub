from app.services.vector_store import EnhancedVectorStore
from app.services.embedding_service import EmbeddingService
from app.utils.text_splitter import AdvancedTextSplitter

class ChatService:
    def __init__(self):
        self.vector_store = EnhancedVectorStore()
        self.embedding_service = EmbeddingService()

    def generate_response(self, query):
        query_embedding = self.embedding_service.generate_embeddings([query.query])[0]
        results = self.vector_store.query(query_embedding, n_results=query.n_results)
        return results

chat_service = ChatService()
