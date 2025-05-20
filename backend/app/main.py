from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import upload, embedding, query_llm, vector_search, model_response

app = FastAPI(
    title="Open Source AI Stack Backend",
    description="Backend API for AI Stack (Upload, Embedding, LLM, Vector Search)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(embedding.router)
app.include_router(query_llm.router)
app.include_router(vector_search.router)
app.include_router(model_response.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
