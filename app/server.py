import asyncio
import time

from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.runtime_engine import RuntimeEngine

app = FastAPI()

engine = RuntimeEngine()

request_queue: asyncio.Queue["InferenceJob"] = asyncio.Queue()

max_batch_size = 1
BATCH_TIMEOUT_S = 0.010

class InferenceRequest(BaseModel):
    prompt: str

class BatchSizeConfig(BaseModel):
    max_batch_size: int

@dataclass
class InferenceJob:
    prompt: str
    enqueued_at: float
    future: asyncio.Future

async def scheduler_loop():

    while True:

        first_job = await request_queue.get()
        batch = [first_job]

        batch_start = time.perf_counter()
        deadline = batch_start + BATCH_TIMEOUT_S # only queue up to BATCH_TIMEOUT_S worth of jobs

        # accumulate jobs until we reach the batch size or the deadline
        while len(batch) < max_batch_size:

            time_remaining = deadline - time.perf_counter()

            if time_remaining <= 0:
                break

            try:
                job = await asyncio.wait_for(
                    request_queue.get(),
                    timeout=time_remaining,
                )
                batch.append(job)

            except asyncio.TimeoutError:
                break
        
        await process_batch(batch)

async def process_batch(batch: list[InferenceJob]):

    started_at = time.perf_counter()

    prompts = [job.prompt for job in batch]

    queue_ms = [
        (started_at - job.enqueued_at) * 1000 for job in batch
    ]

    try:
        compute_start = time.perf_counter()
        outputs = await asyncio.to_thread(engine.infer_batch, prompts)
        compute_ms = (time.perf_counter() - compute_start) * 1000

        for job, output, q_ms in zip(batch, outputs, queue_ms):
            job.future.set_result({
                "output": output,
                "timings": {
                    "queue_ms": q_ms,
                    "compute_ms": compute_ms,
                    "batch_size": len(batch),
                },
            })
    
    except Exception as e:
        for job in batch:
            job.future.set_exception(e)
    
    finally:
        for _ in batch:
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

@app.get("/config/batch-size")
async def get_batch_size():
    return {"max_batch_size": max_batch_size}

@app.post("/config/batch-size")
async def set_batch_size(config: BatchSizeConfig):
    global max_batch_size
    if config.max_batch_size < 1:
        raise HTTPException(status_code=400, detail="max_batch_size must be at least 1")
    max_batch_size = config.max_batch_size
    return {"max_batch_size": max_batch_size}