# python run_benchmarks.py
import subprocess
import sys
from pathlib import Path
import time

# (num_requests, concurrency, max_batch_size, policy) tuples to benchmark
BENCHMARK_CONFIGS = [
    # (300, 16, 8, "fifo"),
    # (300, 32, 8, "fifo"),
    (50, 32, 8, "priority"),
    # (300, 32, 8, "deadline"),
    # (300, 64, 8, "fifo"),
]

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = SCRIPT_DIR / "benchmark.py"


def main():
    total = len(BENCHMARK_CONFIGS)

    for i, (num_requests, concurrency, max_batch_size, policy) in enumerate(
        BENCHMARK_CONFIGS, start=1
    ):
        print(f"\n{'#' * 60}")
        print(
            f"Run {i}/{total}: num_requests={num_requests}, "
            f"concurrency={concurrency}, max_batch_size={max_batch_size}, "
            f"policy={policy}"
        )
        print(f"{'#' * 60}")

        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--num-requests",
                str(num_requests),
                "--concurrency",
                str(concurrency),
                "--max-batch-size",
                str(max_batch_size),
                "--policy",
                policy,
            ],
            cwd=SCRIPT_DIR,
        )

        if result.returncode != 0:
            print(
                f"Benchmark failed (exit {result.returncode}) for "
                f"num_requests={num_requests}, concurrency={concurrency}, "
                f"max_batch_size={max_batch_size}, policy={policy}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
        
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"Completed {total} benchmark run(s).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
