import httpx

class EmbeddingService:
    async def create_embedding(self, text: str):
        url = "http://localhost:11434/api/embeddings"
        payload = {"model": "nomic-embed-text", "prompt": text}
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            result = resp.json()
            return result["embedding"]
