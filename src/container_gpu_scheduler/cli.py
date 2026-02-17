"""CLI entry point for the GPU scheduler.

Provides a command-line interface for demonstrating GPU-aware
scheduling with bin-packing and preemption.
"""

from __future__ import annotations

import argparse
import logging
import sys

from container_gpu_scheduler.core import GPUCluster
from container_gpu_scheduler.models import (
    GPUType,
    SchedulerConfig,
    SchedulingStrategy,
)
from container_gpu_scheduler.utils import (
    create_training_job,
    format_cluster_snapshot,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="gpu-scheduler",
        description="GPU-aware batch scheduler with bin-packing and preemption",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=4,
        help="Number of GPU nodes in the cluster",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=8,
        help="GPUs per node",
    )
    parser.add_argument(
        "--strategy",
        choices=["bin_pack", "spread"],
        default="bin_pack",
        help="Scheduling strategy",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a scheduling demo",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def run_demo(nodes: int, gpus_per_node: int, strategy: str) -> None:
    """Run a scheduling demonstration.

    Args:
        nodes: Number of cluster nodes.
        gpus_per_node: GPUs per node.
        strategy: Scheduling strategy name.
    """
    strat = SchedulingStrategy(strategy)
    config = SchedulerConfig(strategy=strat, enable_preemption=True)
    cluster = GPUCluster(config)
    cluster.add_nodes(nodes, gpus_per_node, GPUType.A100_80GB)

    logger.info("Cluster: %s", format_cluster_snapshot(cluster.snapshot()))

    # Submit training jobs
    jobs = [
        create_training_job("llm-pretrain", 2, 4, priority=90, gang=True),
        create_training_job("fine-tune-v1", 1, 2, priority=70),
        create_training_job("eval-suite", 1, 1, priority=30),
        create_training_job("inference-bench", 1, 1, priority=20),
    ]

    for job in jobs:
        result = cluster.submit_job(job)
        status = "OK" if result.success else f"FAILED: {result.reason}"
        logger.info("Job %s (%s): %s", job.name, job.job_id, status)

    logger.info("After scheduling: %s", format_cluster_snapshot(cluster.snapshot()))

    # Submit a high-priority job that requires preemption
    urgent = create_training_job("urgent-train", 1, 4, priority=95)
    result = cluster.submit_job(urgent)
    if result.success:
        logger.info(
            "Urgent job scheduled with %d preemptions",
            result.total_preemptions,
        )

    logger.info("Final: %s", format_cluster_snapshot(cluster.snapshot()))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list.

    Returns:
        Exit code.
    """
    args = parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.demo:
        run_demo(args.nodes, args.gpus_per_node, args.strategy)
        return 0

    logger.info("Use --demo to run a scheduling demonstration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
