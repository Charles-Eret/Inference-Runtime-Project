from abc import ABC, abstractmethod
import time

class SchedulingPolicy(ABC):
    @abstractmethod
    def select(self, jobs, max_batch_size: int):
        pass


class FIFOPolicy(SchedulingPolicy):
    def select(self, jobs, max_batch_size: int):
        return sorted(jobs, key=lambda j: j.enqueued_at)[:max_batch_size]


class PriorityPolicy(SchedulingPolicy):
    def select(self, jobs, max_batch_size: int):
        return sorted(jobs, key=lambda j: (j.priority, j.enqueued_at))[:max_batch_size]


class DeadlinePolicy(SchedulingPolicy):
    def select(self, jobs, max_batch_size: int):
        now = time.perf_counter()

        def time_left(job):
            if job.deadline_ms is None:
                return float("inf")
            waited_ms = (now - job.enqueued_at) * 1000
            return job.deadline_ms - waited_ms

        return sorted(jobs, key=lambda j: (time_left(j), j.enqueued_at))[:max_batch_size]