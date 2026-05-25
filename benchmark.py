import argparse
import asyncio
import statistics
import time

import httpx
import numpy as np

URL = "http://localhost:8000/infer"


async def send_request(client, idx, results):
    payload = {
        "prompt": f"Hello from request {idx}"
    }

    start = time.perf_counter()

    try:
        response = await client.post(URL, json=payload)
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
            "latency": latency,
            "queue": queue_s,
            "compute": compute_s,
            "network": network_s,
        })
    except Exception:
        results.append({"success": False})


async def worker(semaphore, client, idx, results):
    async with semaphore:
        await send_request(client, idx, results)


async def run_benchmark(num_requests: int, concurrency: int):
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    start_total = time.perf_counter()

    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [
            worker(semaphore, client, i, results) for i in range(num_requests)
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

    throughput = num_requests / total_time if total_time > 0 else 0
    success_rate = (successes / num_requests) * 100 if num_requests > 0 else 0

    print("\n===== RESULTS =====")
    print(f"Total requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
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
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_benchmark(args.num_requests, args.concurrency))


if __name__ == "__main__":
    main()
