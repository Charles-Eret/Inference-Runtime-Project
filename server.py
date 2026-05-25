import asyncio
import time

from dataclasses import dataclass
from fastapi import FastAPI
from pydantic import BaseModel

from runtime_engine import RuntimeEngine

app = FastAPI()

engine = RuntimeEngine()

request_queue: asyncio.Queue["InferenceJob"] = asyncio.Queue()

@dataclass
class InferenceRequest(BaseModel):
    prompt: str

@dataclass
class InferenceJob:
    prompt: str
    enqueued_at: float
    future: asyncio.Future

async def scheduler_loop():
    while True:
        job = await request_queue.get()

        try:
            started_at = time.perf_counter()
            queue_ms = (started_at - job.enqueued_at) * 1000

            compute_start = time.perf_counter()
            output = await engine.infer(job.prompt)
            compute_ms = (time.perf_counter() - compute_start) * 1000

            job.future.set_result({
                "output": output,
                "timings": {
                    "queue_ms": queue_ms,
                    "compute_ms": compute_ms,
                },
            })

        except Exception as e:
            job.future.set_exception(e)

        finally:
            request_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler_loop())

@app.post("/infer")
async def infer(request: InferenceRequest):
    loop = asyncio.get_running_loop()

    job = InferenceJob(
        prompt=request.prompt,
        enqueued_at=time.perf_counter(),
        future=loop.create_future(),
    )

    await request_queue.put(job)

    return await job.future

@app.get("/health")
async def health():
    return {"status": "ok"}