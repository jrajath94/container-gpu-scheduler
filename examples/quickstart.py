"""Quick-start example for the GPU scheduler.

Demonstrates cluster creation, job scheduling with different
strategies, gang scheduling, and priority-based preemption.
"""

from __future__ import annotations

import logging

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the quick-start demo."""
    # ── Create a mixed GPU cluster ───────────────────────────
    logger.info("Creating a heterogeneous GPU cluster...")
    config = SchedulerConfig(
        strategy=SchedulingStrategy.BIN_PACK,
        enable_preemption=True,
    )
    cluster = GPUCluster(config)

    # 4 nodes with A100-80GB (training), 2 nodes with T4 (inference)
    cluster.add_nodes(4, 8, GPUType.A100_80GB)
    cluster.add_nodes(2, 4, GPUType.T4)
    logger.info("Cluster: %s", format_cluster_snapshot(cluster.snapshot()))

    # ── Submit training jobs ─────────────────────────────────
    logger.info("\n--- Submitting training jobs ---")

    # 1. Large distributed training (gang scheduled)
    llm_train = create_training_job(
        name="llm-pretrain-7b",
        num_pods=4,
        gpus_per_pod=4,
        gpu_type=GPUType.A100_80GB,
        priority=80,
        gang=True,
    )
    result = cluster.submit_job(llm_train)
    logger.info(
        "LLM pretrain: %s (16 GPUs, gang)",
        "SCHEDULED" if result.success else f"FAILED: {result.reason}",
    )

    # 2. Fine-tuning job (single pod)
    finetune = create_training_job(
        name="rlhf-finetune",
        num_pods=1,
        gpus_per_pod=4,
        gpu_type=GPUType.A100_80GB,
        priority=60,
    )
    result = cluster.submit_job(finetune)
    logger.info("Fine-tune: %s (4 GPUs)", "SCHEDULED" if result.success else "FAILED")

    # 3. Evaluation job (small)
    eval_job = create_training_job(
        name="eval-harness",
        num_pods=1,
        gpus_per_pod=2,
        gpu_type=GPUType.A100_80GB,
        priority=40,
    )
    result = cluster.submit_job(eval_job)
    logger.info("Eval: %s (2 GPUs)", "SCHEDULED" if result.success else "FAILED")

    # 4. Inference benchmark on T4s
    inference = create_training_job(
        name="inference-bench",
        num_pods=1,
        gpus_per_pod=2,
        gpu_type=GPUType.T4,
        priority=30,
    )
    result = cluster.submit_job(inference)
    logger.info("Inference: %s (2 T4 GPUs)", "SCHEDULED" if result.success else "FAILED")

    # ── Show cluster state ───────────────────────────────────
    snap = cluster.snapshot()
    logger.info("\nAfter scheduling: %s", format_cluster_snapshot(snap))

    logger.info("Per-node utilization:")
    for node_id, util in snap.per_node_utilization.items():
        node = next(n for n in cluster.nodes if n.node_id == node_id)
        logger.info(
            "  %s (%s): %.0f%% (%d/%d GPUs used)",
            node_id, node.gpu_type.value,
            util * 100, node.used_gpus, node.total_gpus,
        )

    # ── Priority preemption demo ─────────────────────────────
    logger.info("\n--- Priority preemption ---")

    # Fill remaining A100 GPUs
    filler = create_training_job(
        name="background-sweep",
        num_pods=1,
        gpus_per_pod=8,
        gpu_type=GPUType.A100_80GB,
        priority=20,
    )
    cluster.submit_job(filler)
    logger.info("Filled cluster. Free GPUs: %d", cluster.snapshot().free_gpus)

    # Submit urgent high-priority job
    urgent = create_training_job(
        name="critical-safety-eval",
        num_pods=1,
        gpus_per_pod=4,
        gpu_type=GPUType.A100_80GB,
        priority=95,
    )
    result = cluster.submit_job(urgent)
    logger.info(
        "Urgent job: %s (preempted %d jobs)",
        "SCHEDULED" if result.success else "FAILED",
        result.total_preemptions,
    )

    # ── Final state ──────────────────────────────────────────
    logger.info("\nFinal: %s", format_cluster_snapshot(cluster.snapshot()))

    # Show job states
    logger.info("\nJob summary:")
    for job in cluster.jobs.values():
        logger.info(
            "  %-25s priority=%-3d state=%s gpus=%d",
            job.name, job.priority, job.state.value, job.total_gpus,
        )

    logger.info("\nQuick-start complete.")


if __name__ == "__main__":
    main()
