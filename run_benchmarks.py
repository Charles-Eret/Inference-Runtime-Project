# python run_benchmarks.py
import subprocess
import sys
from pathlib import Path
import time

# (num_requests, concurrency) pairs to benchmark
BENCHMARK_CONFIGS = [
    (30, 1),
    (100, 4),
    (200, 8),
    # (200, 32),
]

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = SCRIPT_DIR / "benchmark.py"


def main():
    total = len(BENCHMARK_CONFIGS)

    for i, (num_requests, concurrency) in enumerate(BENCHMARK_CONFIGS, start=1):
        print(f"\n{'#' * 60}")
        print(f"Run {i}/{total}: num_requests={num_requests}, concurrency={concurrency}")
        print(f"{'#' * 60}")

        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--num-requests",
                str(num_requests),
                "--concurrency",
                str(concurrency),
            ],
            cwd=SCRIPT_DIR,
        )

        if result.returncode != 0:
            print(
                f"Benchmark failed (exit {result.returncode}) for "
                f"num_requests={num_requests}, concurrency={concurrency}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
        
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"Completed {total} benchmark run(s).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
