from fastapi import FastAPI
from pydantic import BaseModel

from runtime_engine import RuntimeEngine

app = FastAPI()

engine = RuntimeEngine()

class InferenceRequest(BaseModel):
    prompt: str

@app.post("/infer")
async def infer(request: InferenceRequest):
    output = await engine.infer(request.prompt)
    return {"output": output}

@app.get("/health")
async def health():
    return {"status": "ok"}