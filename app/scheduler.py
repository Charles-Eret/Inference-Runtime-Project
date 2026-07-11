import time
import asyncio

class RuntimeScheduler:
    def __init__(self, policy):
        self.policy = policy
        self.jobs = []
        self.condition = asyncio.Condition()

    async def submit(self, job):
        async with self.condition:
            self.jobs.append(job)
            self.condition.notify()

    async def get_batch(self, max_batch_size: int, batch_timeout_ms: int):
        async with self.condition:
            while not self.jobs:
                await self.condition.wait()

        await asyncio.sleep(batch_timeout_ms / 1000)

        async with self.condition:
            selected = self.policy.select(self.jobs, max_batch_size)

            for job in selected:
                self.jobs.remove(job)

            return selected