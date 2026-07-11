import asyncio
import time

from fastapi import FastAPI, HTTPException

from app.types import BatchSizeConfig, InferenceJob, InferenceRequest, PolicyConfig
from app.runtime_engine import RuntimeEngine
from app.scheduler import RuntimeScheduler
from app.policy import FIFOPolicy, PriorityPolicy, DeadlinePolicy

app = FastAPI()

engine = RuntimeEngine()

POLICIES = {
    "fifo": FIFOPolicy(),
    "priority": PriorityPolicy(),
    "deadline": DeadlinePolicy(),
}
active_policy = "fifo"
scheduler = RuntimeScheduler(POLICIES[active_policy])

max_batch_size = 1
BATCH_TIMEOUT_MS = 25

async def scheduler_loop():
    while True:
        batch = await scheduler.get_batch(
            max_batch_size=max_batch_size,
            batch_timeout_ms=BATCH_TIMEOUT_MS,
        )

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

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler_loop())

@app.post("/infer")
async def infer(request: InferenceRequest):
    loop = asyncio.get_running_loop()

    job = InferenceJob(
        prompt=request.prompt,
        priority=request.priority,
        deadline_ms=request.deadline_ms,
        request_type=request.request_type,
        enqueued_at=time.perf_counter(),
        future=loop.create_future(),
    )

    await scheduler.submit(job)
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

@app.get("/config/policy")
async def get_policy():
    return {"policy": active_policy, "available": list(POLICIES.keys())}

@app.post("/config/policy")
async def set_policy(config: PolicyConfig):
    global active_policy
    policy_name = config.policy.lower()
    if policy_name not in POLICIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown policy '{config.policy}'. Available: {list(POLICIES.keys())}",
        )
    active_policy = policy_name
    scheduler.policy = POLICIES[policy_name]
    return {"policy": active_policy}