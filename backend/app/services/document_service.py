import uuid
from app.models.schemas import DocumentModel
from app.utils.text_splitter import AdvancedTextSplitter
from app.services.vector_store import EnhancedVectorStore
from app.services.embedding_service import EmbeddingService

class DocumentService:
    def __init__(self):
        self.vector_store = EnhancedVectorStore()
        self.embedding_service = EmbeddingService()

    def process_document(self, document):
        splitter = AdvancedTextSplitter()
        chunks = splitter.split_text(document)
        doc_models = [
            DocumentModel(id=str(uuid.uuid4()), content=chunk, metadata={})
            for chunk in chunks
        ]
        embeddings = self.embedding_service.generate_embeddings([doc.content for doc in doc_models])
        self.vector_store.add_documents(
            ids=[doc.id for doc in doc_models],
            embeddings=embeddings,
            metadatas=[{"original_document": True} for _ in doc_models]
        )
        return doc_models

document_service = DocumentService()
