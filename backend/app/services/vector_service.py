import faiss
import numpy as np

class VectorService:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.vectors = []
        self.docs = []

    async def add_vector(self, embedding, meta):
        vector = np.array([embedding]).astype('float32')
        self.index.add(vector)
        self.vectors.append(embedding)
        self.docs.append(meta)

    async def search(self, embedding, top_k=5):
        query = np.array([embedding]).astype('float32')
        D, I = self.index.search(query, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx])
        return results
