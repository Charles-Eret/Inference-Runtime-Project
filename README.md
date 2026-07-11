# Inference-Runtime-Project

## Versions

### V1: No Scheduler Policy No Batching

### V2: FIFO batching
After a set timeout or max batch size, run inference on the earliest arrivals.

### V3: Policy (FIFO, Priority, Deadline)
After a set timeout, select the batch of max batch size to run based on the selected policy.