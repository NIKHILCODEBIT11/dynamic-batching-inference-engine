from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.model_engine import ONNXEmbeddingEngine
from app.batcher import DynamicBatcher

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class EmbedResponse(BaseModel):
    dim: int
    embedding: list[float]

batcher: DynamicBatcher | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher
    print("Loading ONNX Model Engine...")
    engine = ONNXEmbeddingEngine(model_dir="models")
    batcher = DynamicBatcher(engine=engine, max_batch_size=16, max_wait_time_ms=10.0)
    batcher.start()
    yield
    print("Shutting down Dynamic Batcher...")
    await batcher.stop()

app = FastAPI(title="Dynamic Batching Inference Engine", lifespan=lifespan)

@app.post("/embed", response_model=EmbedResponse)
async def get_embedding(req: EmbedRequest):
    if batcher is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    try:
        vector = await batcher.enqueue(req.text)
        return EmbedResponse(dim=len(vector), embedding=vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}