import argparse
import asyncio
import csv
import random
import statistics
import time
from pathlib import Path

import httpx
import numpy as np

BASE_URL = "http://localhost:8000"
INFER_URL = f"{BASE_URL}/infer"
BATCH_SIZE_URL = f"{BASE_URL}/config/batch-size"
POLICY_URL = f"{BASE_URL}/config/policy"

RANDOM_SEED = 42
WRITE_RESULTS = True
RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "requests.csv"
RUN_SUMMARY_CSV = RESULTS_DIR / "run_summary.csv"
TYPE_SUMMARY_CSV = RESULTS_DIR / "type_summary.csv"

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
    if not result.get("deadline_met", False):
        return True
    return False


def write_request_results(request_results: list[dict]):
    if not request_results:
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = list(request_results[0].keys())

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(request_results)


def write_run_summary(run_summary: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = list(run_summary.keys())

    with open(RUN_SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(run_summary)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def build_type_summaries(
    request_results: list[dict],
    policy: str,
    concurrency: int,
    max_batch_size: int,
    seed: int,
) -> list[dict]:
    type_summaries = []
    for request_type in REQUEST_TYPES:
        rows = [r for r in request_results if r.get("request_type") == request_type]
        if not rows:
            continue

        latencies = [
            row["latency_ms"]
            for row in rows
            if row.get("latency_ms") is not None
        ]
        type_summaries.append({
            "policy": policy,
            "concurrency": concurrency,
            "max_batch_size": max_batch_size,
            "seed": seed,
            "request_type": request_type,
            "request_count": len(rows),
            "deadline_success_rate": sum(
                row["deadline_met"] for row in rows
            ) / len(rows),
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
        })
    return type_summaries


def write_type_summaries(type_summaries: list[dict]):
    if not type_summaries:
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = list(type_summaries[0].keys())

    with open(TYPE_SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(type_summaries)


def write_all_results(
    request_results: list[dict],
    run_summary: dict,
    type_summaries: list[dict],
):
    write_request_results(request_results)
    write_run_summary(run_summary)
    write_type_summaries(type_summaries)


def print_per_type_metrics(results: list[dict]):
    print("\n===== PER-TYPE METRICS =====")
    for request_type in REQUEST_TYPES:
        type_results = [r for r in results if r.get("request_type") == request_type]
        if not type_results:
            print(f"\n--- {request_type} ---")
            print("Requests: 0")
            continue

        latencies_s = [
            r["latency_ms"] / 1000
            for r in type_results
            if r.get("latency_ms") is not None
        ]
        misses = sum(1 for r in type_results if is_deadline_miss(r))
        miss_rate = misses / len(type_results) * 100

        print(f"\n--- {request_type} ---")
        print(f"Requests: {len(type_results)}")
        print(f"Deadline miss rate: {miss_rate:.1f}%")
        if latencies_s:
            print(f"P50 latency: {np.percentile(latencies_s, 50):.2f}s")
            print(f"P95 latency: {np.percentile(latencies_s, 95):.2f}s")
            print(f"P99 latency: {np.percentile(latencies_s, 99):.2f}s")
        else:
            print("P50 latency: n/a")
            print("P95 latency: n/a")
            print("P99 latency: n/a")


async def send_request(client, idx, profile, results, run_config):
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
        latency_ms = (end - start) * 1000

        timings = data.get("timings", {})
        queue_time_ms = timings.get("queue_ms", 0)
        compute_time_ms = timings.get("compute_ms", 0)
        server_ms = queue_time_ms + compute_time_ms
        network_time_ms = max(0.0, latency_ms - server_ms)
        batch_size = timings.get("batch_size", 1)

        result = {
            "request_id": idx,
            "request_type": profile["request_type"],
            "priority": profile["priority"],
            "deadline_ms": profile["deadline_ms"],
            "latency_ms": latency_ms,
            "queue_time_ms": queue_time_ms,
            "compute_time_ms": compute_time_ms,
            "network_time_ms": network_time_ms,
            "batch_size": batch_size,
            "deadline_met": latency_ms <= profile["deadline_ms"],
        }
        result.update(run_config)
        results.append(result)
    except Exception:
        result = {
            "request_id": idx,
            "request_type": profile["request_type"],
            "priority": profile["priority"],
            "deadline_ms": profile["deadline_ms"],
            "latency_ms": None,
            "queue_time_ms": None,
            "compute_time_ms": None,
            "network_time_ms": None,
            "batch_size": None,
            "deadline_met": False,
        }
        result.update(run_config)
        results.append(result)


async def worker(semaphore, client, idx, profile, results, run_config):
    async with semaphore:
        await send_request(client, idx, profile, results, run_config)


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
    run_config = {
        "policy": policy,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "max_batch_size": max_batch_size,
        "seed": RANDOM_SEED,
    }

    start_total = time.perf_counter()

    async with httpx.AsyncClient(timeout=120) as client:
        await configure_batch_size(client, max_batch_size)
        await configure_policy(client, policy)
        tasks = [
            worker(semaphore, client, i, profiles[i], results, run_config)
            for i in range(num_requests)
        ]
        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_total

    ok = [r for r in results if r.get("latency_ms") is not None]
    successes = len(ok)
    failures = num_requests - successes
    latencies_ms = [r["latency_ms"] for r in ok]
    queue_times_ms = [r["queue_time_ms"] for r in ok]
    compute_times_ms = [r["compute_time_ms"] for r in ok]
    network_times_ms = [r["network_time_ms"] for r in ok]
    batch_sizes = [r["batch_size"] for r in ok]

    requests_per_second = num_requests / total_time if total_time > 0 else 0
    success_rate = (successes / num_requests) * 100 if num_requests > 0 else 0

    avg_latency_ms = statistics.mean(latencies_ms) if latencies_ms else None
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else None
    p99_latency_ms = float(np.percentile(latencies_ms, 99)) if latencies_ms else None
    avg_queue_time_ms = statistics.mean(queue_times_ms) if queue_times_ms else None
    avg_compute_time_ms = statistics.mean(compute_times_ms) if compute_times_ms else None
    avg_batch_size = statistics.mean(batch_sizes) if batch_sizes else None
    deadline_success_rate = (
        sum(row["deadline_met"] for row in results) / len(results)
        if results
        else 0.0
    )

    run_summary = {
        "policy": policy,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "max_batch_size": max_batch_size,
        "seed": RANDOM_SEED,
        "total_time_s": total_time,
        "requests_per_second": requests_per_second,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "p99_latency_ms": p99_latency_ms,
        "avg_queue_time_ms": avg_queue_time_ms,
        "avg_compute_time_ms": avg_compute_time_ms,
        "avg_batch_size": avg_batch_size,
        "deadline_success_rate": deadline_success_rate,
    }
    type_summaries = build_type_summaries(
        results,
        policy=policy,
        concurrency=concurrency,
        max_batch_size=max_batch_size,
        seed=RANDOM_SEED,
    )

    if WRITE_RESULTS:
        write_all_results(results, run_summary, type_summaries)

    print("\n===== RESULTS =====")
    print(f"Total requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Max batch size (configured): {max_batch_size}")
    print(f"Policy (configured): {policy}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests/sec: {requests_per_second:.2f}")
    if WRITE_RESULTS:
        print(f"Wrote {len(results)} rows to {RESULTS_CSV}")
        print(f"Wrote run summary to {RUN_SUMMARY_CSV}")
        print(f"Wrote {len(type_summaries)} type summary rows to {TYPE_SUMMARY_CSV}")
    else:
        print("CSV writing disabled (WRITE_RESULTS=False)")

    if latencies_ms:
        print(f"Average latency: {avg_latency_ms / 1000:.2f}s")
        print(f"Min latency: {min(latencies_ms) / 1000:.2f}s")
        print(f"Max latency: {max(latencies_ms) / 1000:.2f}s")
        print(f"P95 latency: {p95_latency_ms / 1000:.2f}s")
        print(f"P99 latency: {p99_latency_ms / 1000:.2f}s")
        print(f"Avg queue time: {avg_queue_time_ms / 1000:.2f}s")
        print(f"Avg compute time: {avg_compute_time_ms / 1000:.2f}s")
        print(f"Avg network time: {statistics.mean(network_times_ms) / 1000:.2f}s")
        print(f"Avg batch size: {avg_batch_size:.2f}")
        print(f"Deadline success rate: {deadline_success_rate:.1%}")

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
