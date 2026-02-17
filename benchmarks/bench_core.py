"""Performance benchmarks for the GPU scheduler.

Measures scheduling throughput, bin-packing efficiency,
and preemption overhead.
"""

from __future__ import annotations

import logging
import statistics
import time

from container_gpu_scheduler.core import GPUCluster
from container_gpu_scheduler.models import (
    GPUType,
    SchedulerConfig,
    SchedulingStrategy,
)
from container_gpu_scheduler.utils import create_training_job

logging.basicConfig(level=logging.WARNING)

SEPARATOR = "=" * 60


def bench_scheduling_throughput(num_jobs: int = 500) -> dict[str, float]:
    """Benchmark job scheduling throughput.

    Args:
        num_jobs: Number of jobs to schedule.

    Returns:
        Benchmark results.
    """
    config = SchedulerConfig(strategy=SchedulingStrategy.BIN_PACK)
    cluster = GPUCluster(config)
    cluster.add_nodes(32, 8, GPUType.A100_80GB)  # 256 GPUs

    latencies: list[float] = []
    scheduled = 0

    start = time.perf_counter()
    for i in range(num_jobs):
        job = create_training_job(f"job-{i}", 1, 1, priority=50)
        t0 = time.perf_counter()
        result = cluster.submit_job(job)
        latencies.append(time.perf_counter() - t0)
        if result.success:
            scheduled += 1
    elapsed = time.perf_counter() - start

    latencies_us = [lat * 1_000_000 for lat in latencies]
    latencies_us.sort()

    return {
        "total_jobs": num_jobs,
        "scheduled": scheduled,
        "elapsed_s": elapsed,
        "jobs_per_sec": num_jobs / elapsed,
        "p50_us": latencies_us[len(latencies_us) // 2],
        "p99_us": latencies_us[int(len(latencies_us) * 0.99)],
        "mean_us": statistics.mean(latencies_us),
    }


def bench_bin_pack_efficiency() -> dict[str, float]:
    """Measure GPU utilization with bin-packing vs spread.

    Returns:
        Utilization comparison.
    """
    results: dict[str, float] = {}

    for strategy_name in ["bin_pack", "spread"]:
        strategy = SchedulingStrategy(strategy_name)
        config = SchedulerConfig(strategy=strategy, enable_preemption=False)
        cluster = GPUCluster(config)
        cluster.add_nodes(8, 8, GPUType.A100_80GB)  # 64 GPUs

        # Submit jobs of varying sizes
        sizes = [1, 2, 1, 4, 2, 1, 2, 1, 4, 2, 1, 1, 2, 4, 1, 2]
        for i, gpu_count in enumerate(sizes):
            job = create_training_job(f"job-{i}", 1, gpu_count)
            cluster.submit_job(job)

        snap = cluster.snapshot()
        nodes_active = sum(
            1 for u in snap.per_node_utilization.values() if u > 0
        )
        results[f"{strategy_name}_utilization"] = snap.gpu_utilization
        results[f"{strategy_name}_nodes_active"] = nodes_active

    return results


def bench_gang_scheduling(num_trials: int = 100) -> dict[str, float]:
    """Benchmark gang scheduling latency.

    Args:
        num_trials: Number of gang jobs to schedule.

    Returns:
        Benchmark results.
    """
    latencies: list[float] = []

    for i in range(num_trials):
        config = SchedulerConfig(strategy=SchedulingStrategy.BIN_PACK)
        cluster = GPUCluster(config)
        cluster.add_nodes(8, 8, GPUType.A100_80GB)

        job = create_training_job(f"gang-{i}", 4, 4, gang=True)
        t0 = time.perf_counter()
        cluster.submit_job(job)
        latencies.append(time.perf_counter() - t0)

    latencies_us = [lat * 1_000_000 for lat in latencies]
    latencies_us.sort()

    return {
        "trials": num_trials,
        "p50_us": latencies_us[len(latencies_us) // 2],
        "p99_us": latencies_us[int(len(latencies_us) * 0.99)],
        "mean_us": statistics.mean(latencies_us),
    }


def bench_preemption(num_trials: int = 50) -> dict[str, float]:
    """Benchmark preemption overhead.

    Args:
        num_trials: Number of preemption events.

    Returns:
        Benchmark results.
    """
    latencies: list[float] = []

    for i in range(num_trials):
        config = SchedulerConfig(enable_preemption=True)
        cluster = GPUCluster(config)
        cluster.add_nodes(4, 8, GPUType.A100_80GB)

        # Fill with low-priority jobs
        for j in range(8):
            job = create_training_job(f"low-{j}", 1, 4, priority=10)
            cluster.submit_job(job)

        # Preempt with high-priority
        urgent = create_training_job(f"urgent-{i}", 1, 4, priority=95)
        t0 = time.perf_counter()
        cluster.submit_job(urgent)
        latencies.append(time.perf_counter() - t0)

    latencies_us = [lat * 1_000_000 for lat in latencies]
    latencies_us.sort()

    return {
        "trials": num_trials,
        "p50_us": latencies_us[len(latencies_us) // 2],
        "p99_us": latencies_us[int(len(latencies_us) * 0.99)],
        "mean_us": statistics.mean(latencies_us),
    }


def bench_cluster_scaling() -> list[dict[str, float]]:
    """Benchmark scheduling across cluster sizes.

    Returns:
        Results per cluster size.
    """
    results = []
    for num_nodes in [4, 8, 16, 32, 64]:
        config = SchedulerConfig()
        cluster = GPUCluster(config)
        cluster.add_nodes(num_nodes, 8, GPUType.A100_80GB)

        num_jobs = num_nodes * 4  # 4 jobs per node
        latencies: list[float] = []

        start = time.perf_counter()
        for i in range(num_jobs):
            job = create_training_job(f"j-{i}", 1, 2)
            t0 = time.perf_counter()
            cluster.submit_job(job)
            latencies.append(time.perf_counter() - t0)
        elapsed = time.perf_counter() - start

        latencies_us = [lat * 1_000_000 for lat in latencies]
        latencies_us.sort()

        results.append({
            "nodes": num_nodes,
            "total_gpus": num_nodes * 8,
            "jobs": num_jobs,
            "jobs_per_sec": num_jobs / elapsed,
            "p50_us": latencies_us[len(latencies_us) // 2],
            "p99_us": latencies_us[int(len(latencies_us) * 0.99)],
        })

    return results


def print_results(title: str, results: dict[str, float]) -> None:
    """Format and print benchmark results.

    Args:
        title: Benchmark name.
        results: Result metrics.
    """
    print(f"\n{title}")
    print("-" * 40)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key:>25s}: {value:>12.2f}")
        else:
            print(f"  {key:>25s}: {value:>12}")


def main() -> None:
    """Run all benchmarks."""
    print(SEPARATOR)
    print("Container GPU Scheduler -- Benchmark Suite")
    print(SEPARATOR)

    # Scheduling throughput
    throughput = bench_scheduling_throughput()
    print_results("Scheduling Throughput (500 jobs, 256 GPUs)", throughput)

    # Bin-pack efficiency
    efficiency = bench_bin_pack_efficiency()
    print_results("Packing Efficiency (8 nodes, 64 GPUs)", efficiency)

    # Gang scheduling
    gang = bench_gang_scheduling()
    print_results("Gang Scheduling (4x4 GPU jobs)", gang)

    # Preemption
    preemption = bench_preemption()
    print_results("Preemption Overhead", preemption)

    # Scaling
    print(f"\nCluster Size Scaling")
    print("-" * 60)
    print(f"  {'Nodes':>5s} | {'GPUs':>5s} | {'Jobs':>5s} | {'Jobs/sec':>10s} | {'p50 (us)':>10s} | {'p99 (us)':>10s}")
    print(f"  {'-'*5} | {'-'*5} | {'-'*5} | {'-'*10} | {'-'*10} | {'-'*10}")
    for r in bench_cluster_scaling():
        print(
            f"  {int(r['nodes']):>5d} | {int(r['total_gpus']):>5d} | "
            f"{int(r['jobs']):>5d} | {r['jobs_per_sec']:>10.0f} | "
            f"{r['p50_us']:>10.1f} | {r['p99_us']:>10.1f}"
        )

    print(f"\n{SEPARATOR}")
    print("Benchmark complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
