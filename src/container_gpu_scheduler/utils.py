"""Utility functions for the GPU scheduler.

Provides scoring, formatting, and helper functions used
across the scheduling implementation.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from container_gpu_scheduler.models import (
    ClusterSnapshot,
    GPUType,
    JobSpec,
    NodeResources,
    PodSpec,
)

logger = logging.getLogger(__name__)


def generate_job_id() -> str:
    """Generate a unique job identifier.

    Returns:
        Short UUID-based job ID.
    """
    return f"job-{uuid.uuid4().hex[:8]}"


def generate_pod_id(job_id: str, index: int) -> str:
    """Generate a pod identifier for a job.

    Args:
        job_id: Parent job ID.
        index: Pod index within the job.

    Returns:
        Pod ID string.
    """
    return f"{job_id}-pod-{index}"


def bin_pack_score(node: NodeResources) -> float:
    """Score a node for bin-packing (higher = more packed, preferred).

    Prefers nodes that are already partially used to consolidate
    workloads on fewer nodes.

    Args:
        node: Node to score.

    Returns:
        Score in [0, 1]. Higher means more utilized.
    """
    return node.utilization


def spread_score(node: NodeResources) -> float:
    """Score a node for spreading (higher = less packed, preferred).

    Prefers nodes with more free resources to spread load.

    Args:
        node: Node to score.

    Returns:
        Score in [0, 1]. Higher means less utilized.
    """
    return 1.0 - node.utilization


def can_fit_pod(node: NodeResources, pod: PodSpec, gpu_type: GPUType) -> bool:
    """Check if a pod can fit on a node.

    Args:
        node: Target node.
        pod: Pod to place.
        gpu_type: Required GPU type.

    Returns:
        True if the node has sufficient resources.
    """
    if node.gpu_type != gpu_type:
        return False
    if node.free_gpus < pod.gpu_count:
        return False
    return True


def format_cluster_snapshot(snapshot: ClusterSnapshot) -> str:
    """Format cluster snapshot for logging.

    Args:
        snapshot: Current cluster state.

    Returns:
        Formatted string.
    """
    return (
        f"nodes={snapshot.total_nodes} | "
        f"gpus={snapshot.free_gpus}/{snapshot.total_gpus} free | "
        f"util={snapshot.gpu_utilization:.1%} | "
        f"running={snapshot.running_jobs} | "
        f"pending={snapshot.pending_jobs}"
    )


def create_training_job(
    name: str,
    num_pods: int,
    gpus_per_pod: int,
    gpu_type: GPUType = GPUType.A100_80GB,
    priority: int = 50,
    gang: bool = False,
) -> JobSpec:
    """Helper to create a training job specification.

    Args:
        name: Job name.
        num_pods: Number of worker pods.
        gpus_per_pod: GPUs per pod.
        gpu_type: Required GPU type.
        priority: Job priority (0-100).
        gang: Require all-or-nothing scheduling.

    Returns:
        Configured JobSpec.
    """
    job_id = generate_job_id()
    pods = [
        PodSpec(
            pod_id=generate_pod_id(job_id, i),
            gpu_count=gpus_per_pod,
        )
        for i in range(num_pods)
    ]
    return JobSpec(
        job_id=job_id,
        name=name,
        priority=priority,
        gpu_type=gpu_type,
        pods=pods,
        gang_schedule=gang,
    )
