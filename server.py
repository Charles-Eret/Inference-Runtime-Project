import asyncio
import time

from fastapi import FastAPI
from pydantic import BaseModel

from runtime_engine import RuntimeEngine

app = FastAPI()

engine = RuntimeEngine()
_inference_lock = asyncio.Lock()

class InferenceRequest(BaseModel):
    prompt: str

@app.post("/infer")
async def infer(request: InferenceRequest):
    enqueued_at = time.perf_counter()
    async with _inference_lock:
        started_at = time.perf_counter()
        queue_ms = (started_at - enqueued_at) * 1000

        compute_start = time.perf_counter()
        output = await engine.infer(request.prompt)
        compute_ms = (time.perf_counter() - compute_start) * 1000

    return {
        "output": output,
        "timings": {
            "queue_ms": queue_ms,
            "compute_ms": compute_ms,
        },
    }

@app.get("/health")
async def health():
    return {"status": "ok"}