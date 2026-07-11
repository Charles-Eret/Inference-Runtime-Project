import argparse
import asyncio
import random
import statistics
import time

import httpx
import numpy as np

BASE_URL = "http://localhost:8000"
INFER_URL = f"{BASE_URL}/infer"
BATCH_SIZE_URL = f"{BASE_URL}/config/batch-size"
POLICY_URL = f"{BASE_URL}/config/policy"

RANDOM_SEED = 42

REQUEST_TEMPLATES = [
    {"request_type": "emergency_control", "priority": 0, "deadline_ms": 5500, "weight": 0.05},
    {"request_type": "normal_control", "priority": 0, "deadline_ms": 7000, "weight": 0.10},
    {"request_type": "urgent_perception", "priority": 1, "deadline_ms": 6000, "weight": 0.25},
    {"request_type": "normal_perception", "priority": 1, "deadline_ms": 8500, "weight": 0.35},
    {"request_type": "analytics", "priority": 2, "deadline_ms": 14000, "weight": 0.20},
    {"request_type": "urgent_logging", "priority": 2, "deadline_ms": 7500, "weight": 0.05},
]

REQUEST_TYPES = [template["request_type"] for template in REQUEST_TEMPLATES]


def sample_request_profiles(num_requests: int) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    weights = [template["weight"] for template in REQUEST_TEMPLATES]
    profiles = []
    for _ in range(num_requests):
        template = rng.choices(REQUEST_TEMPLATES, weights=weights, k=1)[0]
        profiles.append({
            "request_type": template["request_type"],
            "priority": template["priority"],
            "deadline_ms": template["deadline_ms"],
        })
    return profiles


def describe_request_mix() -> str:
    parts = [
        f"{int(t['weight'] * 100)}% {t['request_type']} "
        f"(priority={t['priority']}, deadline={t['deadline_ms']}ms)"
        for t in REQUEST_TEMPLATES
    ]
    return ", ".join(parts)


def is_deadline_miss(result: dict) -> bool:
    if not result["success"]:
        return True
    return result["latency"] * 1000 > result["deadline_ms"]


def print_per_type_metrics(results: list[dict]):
    print("\n===== PER-TYPE METRICS =====")
    for request_type in REQUEST_TYPES:
        type_results = [r for r in results if r.get("request_type") == request_type]
        if not type_results:
            print(f"\n--- {request_type} ---")
            print("Requests: 0")
            continue

        latencies = [r["latency"] for r in type_results if r["success"]]
        misses = sum(1 for r in type_results if is_deadline_miss(r))
        miss_rate = misses / len(type_results) * 100

        print(f"\n--- {request_type} ---")
        print(f"Requests: {len(type_results)}")
        print(f"Deadline miss rate: {miss_rate:.1f}%")
        if latencies:
            print(f"P50 latency: {np.percentile(latencies, 50):.2f}s")
            print(f"P95 latency: {np.percentile(latencies, 95):.2f}s")
            print(f"P99 latency: {np.percentile(latencies, 99):.2f}s")
        else:
            print("P50 latency: n/a")
            print("P95 latency: n/a")
            print("P99 latency: n/a")


async def send_request(client, idx, profile, results):
    payload = {
        "prompt": f"Hello from request {idx}",
        **profile,
    }

    start = time.perf_counter()

    try:
        response = await client.post(INFER_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        end = time.perf_counter()
        latency = end - start

        timings = data.get("timings", {})
        queue_s = timings.get("queue_ms", 0) / 1000
        compute_s = timings.get("compute_ms", 0) / 1000
        server_s = queue_s + compute_s
        network_s = max(0.0, latency - server_s)

        results.append({
            "success": True,
            "request_type": profile["request_type"],
            "deadline_ms": profile["deadline_ms"],
            "latency": latency,
            "queue": queue_s,
            "compute": compute_s,
            "network": network_s,
            "batch_size": timings.get("batch_size", 1),
        })
    except Exception:
        results.append({
            "success": False,
            "request_type": profile["request_type"],
            "deadline_ms": profile["deadline_ms"],
        })


async def worker(semaphore, client, idx, profile, results):
    async with semaphore:
        await send_request(client, idx, profile, results)


async def configure_batch_size(client: httpx.AsyncClient, max_batch_size: int):
    response = await client.post(
        BATCH_SIZE_URL,
        json={"max_batch_size": max_batch_size},
    )
    response.raise_for_status()


async def configure_policy(client: httpx.AsyncClient, policy: str):
    response = await client.post(
        POLICY_URL,
        json={"policy": policy},
    )
    response.raise_for_status()


async def run_benchmark(
    num_requests: int, concurrency: int, max_batch_size: int, policy: str
):
    results = []
    semaphore = asyncio.Semaphore(concurrency)
    profiles = sample_request_profiles(num_requests)

    start_total = time.perf_counter()

    async with httpx.AsyncClient(timeout=120) as client:
        await configure_batch_size(client, max_batch_size)
        await configure_policy(client, policy)
        tasks = [
            worker(semaphore, client, i, profiles[i], results)
            for i in range(num_requests)
        ]
        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_total

    ok = [r for r in results if r["success"]]
    successes = len(ok)
    failures = num_requests - successes
    latencies = [r["latency"] for r in ok]
    queue_times = [r["queue"] for r in ok]
    compute_times = [r["compute"] for r in ok]
    network_times = [r["network"] for r in ok]
    batch_sizes = [r["batch_size"] for r in ok]

    throughput = num_requests / total_time if total_time > 0 else 0
    success_rate = (successes / num_requests) * 100 if num_requests > 0 else 0

    print("\n===== RESULTS =====")
    print(f"Total requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Max batch size (configured): {max_batch_size}")
    print(f"Policy (configured): {policy}")
    print(f"Random seed: {RANDOM_SEED}")
    # print(f"Target request mix: {describe_request_mix()}")
    # print(f"Successful: {successes}")
    # print(f"Failed: {failures}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests/sec: {throughput:.2f}")

    if latencies:
        print(f"Average latency: {statistics.mean(latencies):.2f}s")
        print(f"Min latency: {min(latencies):.2f}s")
        print(f"Max latency: {max(latencies):.2f}s")
        print(f"P95 latency: {np.percentile(latencies, 95):.2f}s")
        print(f"P99 latency: {np.percentile(latencies, 99):.2f}s")
        print(f"Avg queue time: {statistics.mean(queue_times):.2f}s")
        print(f"Avg compute time: {statistics.mean(compute_times):.2f}s")
        print(f"Avg network time: {statistics.mean(network_times):.2f}s")
        print(f"Avg batch size: {statistics.mean(batch_sizes):.2f}")

    print_per_type_metrics(results)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the inference server.")
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to send (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent in-flight requests (default: 4)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Server inference batch size to configure before the run (default: 1)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="fifo",
        choices=["fifo", "priority", "deadline"],
        help="Scheduling policy to configure before the run (default: fifo)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(
        run_benchmark(
            args.num_requests,
            args.concurrency,
            args.max_batch_size,
            args.policy,
        )
    )


if __name__ == "__main__":
    main()
