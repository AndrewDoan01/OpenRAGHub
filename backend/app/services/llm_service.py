import httpx

class LLMService:
    async def query(self, prompt: str, context: str = None):
        url = "http://localhost:11434/api/generate"
        payload = {"model": "llama3", "prompt": prompt}
        if context:
            payload["context"] = context
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            result = resp.json()
            return result.get("response", "")

    async def model_response(self, input: str):
        return await self.query(input)
