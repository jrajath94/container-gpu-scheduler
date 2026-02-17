"""Shared fixtures for GPU scheduler tests."""

from __future__ import annotations

import pytest

from container_gpu_scheduler.core import BinPackScheduler, GPUCluster, GangScheduler
from container_gpu_scheduler.models import (
    GPUType,
    JobSpec,
    JobState,
    NodeResources,
    PodSpec,
    SchedulerConfig,
    SchedulingStrategy,
)
from container_gpu_scheduler.utils import create_training_job


@pytest.fixture
def scheduler_config() -> SchedulerConfig:
    """Default bin-packing config."""
    return SchedulerConfig(
        strategy=SchedulingStrategy.BIN_PACK,
        enable_preemption=True,
    )


@pytest.fixture
def spread_config() -> SchedulerConfig:
    """Spread scheduling config."""
    return SchedulerConfig(
        strategy=SchedulingStrategy.SPREAD,
        enable_preemption=False,
    )


@pytest.fixture
def single_node() -> NodeResources:
    """A single 8-GPU A100 node."""
    return NodeResources(
        node_id="node-0",
        total_gpus=8,
        gpu_type=GPUType.A100_80GB,
    )


@pytest.fixture
def small_cluster(scheduler_config: SchedulerConfig) -> GPUCluster:
    """4-node cluster with 8 GPUs each (32 total)."""
    cluster = GPUCluster(scheduler_config)
    cluster.add_nodes(4, 8, GPUType.A100_80GB)
    return cluster


@pytest.fixture
def small_job() -> JobSpec:
    """Single-pod job requesting 2 GPUs."""
    return create_training_job("small-job", 1, 2, priority=50)


@pytest.fixture
def large_job() -> JobSpec:
    """Single-pod job requesting 8 GPUs (full node)."""
    return create_training_job("large-job", 1, 8, priority=50)


@pytest.fixture
def distributed_job() -> JobSpec:
    """4-pod distributed job, 4 GPUs each (16 total), gang scheduled."""
    return create_training_job("dist-job", 4, 4, priority=70, gang=True)


@pytest.fixture
def high_priority_job() -> JobSpec:
    """High-priority job for preemption tests."""
    return create_training_job("urgent-job", 1, 4, priority=95)


@pytest.fixture
def low_priority_job() -> JobSpec:
    """Low-priority job for preemption tests."""
    return create_training_job("background-job", 1, 2, priority=10)
