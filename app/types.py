import asyncio
from dataclasses import dataclass

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    prompt: str
    priority: int
    deadline_ms: int | None
    request_type: str


class BatchSizeConfig(BaseModel):
    max_batch_size: int


class PolicyConfig(BaseModel):
    policy: str


@dataclass
class InferenceJob:
    prompt: str
    priority: int  # lower = more urgent
    deadline_ms: int | None
    request_type: str  # "control", "perception", "logging"
    enqueued_at: float
    future: asyncio.Future
