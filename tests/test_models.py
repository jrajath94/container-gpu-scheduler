"""Tests for data models and validation."""

from __future__ import annotations

import pytest

from container_gpu_scheduler.models import (
    ClusterSnapshot,
    GPU_COMPUTE_UNITS,
    GPU_MEMORY_GB,
    GPUSlot,
    GPUType,
    JobSpec,
    JobState,
    NodeResources,
    PodSpec,
    SchedulerConfig,
    SchedulingDecision,
    SchedulingResult,
    SchedulingStrategy,
)


class TestGPUType:
    """Tests for GPU type enum."""

    def test_all_types_present(self) -> None:
        assert len(GPUType) == 6

    def test_memory_mapping(self) -> None:
        assert GPU_MEMORY_GB[GPUType.A100_40GB] == 40
        assert GPU_MEMORY_GB[GPUType.H100] == 80
        assert GPU_MEMORY_GB[GPUType.T4] == 16

    def test_compute_units(self) -> None:
        assert GPU_COMPUTE_UNITS[GPUType.H100] > GPU_COMPUTE_UNITS[GPUType.T4]


class TestGPUSlot:
    """Tests for GPU slot tracking."""

    def test_defaults(self) -> None:
        slot = GPUSlot()
        assert slot.is_free
        assert slot.allocated_to is None

    def test_allocation(self) -> None:
        slot = GPUSlot()
        slot.allocated_to = "job-1"
        assert not slot.is_free
        assert slot.allocated_to == "job-1"


class TestNodeResources:
    """Tests for node resource management."""

    def test_creation(self, single_node: NodeResources) -> None:
        assert single_node.total_gpus == 8
        assert single_node.free_gpus == 8
        assert single_node.used_gpus == 0
        assert single_node.utilization == 0.0

    def test_gpu_slots_auto_created(self, single_node: NodeResources) -> None:
        assert len(single_node.gpu_slots) == 8
        assert all(s.gpu_type == GPUType.A100_80GB for s in single_node.gpu_slots)

    def test_allocate(self, single_node: NodeResources) -> None:
        gpu_ids = single_node.allocate(3, "job-1")
        assert len(gpu_ids) == 3
        assert single_node.free_gpus == 5
        assert single_node.utilization == 3 / 8

    def test_release(self, single_node: NodeResources) -> None:
        single_node.allocate(4, "job-1")
        released = single_node.release("job-1")
        assert released == 4
        assert single_node.free_gpus == 8

    def test_multiple_jobs(self, single_node: NodeResources) -> None:
        single_node.allocate(3, "job-1")
        single_node.allocate(2, "job-2")
        assert single_node.free_gpus == 3

        single_node.release("job-1")
        assert single_node.free_gpus == 6

    def test_gpus_for_job(self, single_node: NodeResources) -> None:
        single_node.allocate(2, "job-1")
        single_node.allocate(3, "job-2")
        assert len(single_node.gpus_for_job("job-1")) == 2
        assert len(single_node.gpus_for_job("job-2")) == 3
        assert len(single_node.gpus_for_job("job-3")) == 0

    @pytest.mark.parametrize(
        "total_gpus,allocate_count,expected_free",
        [
            (1, 1, 0),
            (4, 2, 2),
            (8, 0, 8),
            (8, 8, 0),
        ],
    )
    def test_utilization_scenarios(
        self, total_gpus: int, allocate_count: int, expected_free: int
    ) -> None:
        node = NodeResources(node_id="test", total_gpus=total_gpus)
        node.allocate(allocate_count, "job-1")
        assert node.free_gpus == expected_free


class TestPodSpec:
    """Tests for pod specification."""

    def test_defaults(self) -> None:
        pod = PodSpec(pod_id="p1")
        assert pod.gpu_count == 1
        assert not pod.is_scheduled

    def test_scheduled(self) -> None:
        pod = PodSpec(pod_id="p1", assigned_node="node-0")
        assert pod.is_scheduled


class TestJobSpec:
    """Tests for job specification."""

    def test_total_gpus(self) -> None:
        job = JobSpec(
            job_id="j1",
            pods=[PodSpec(gpu_count=4), PodSpec(gpu_count=4)],
        )
        assert job.total_gpus == 8

    def test_is_distributed(self) -> None:
        single = JobSpec(job_id="j1", pods=[PodSpec()])
        assert not single.is_distributed

        multi = JobSpec(job_id="j2", pods=[PodSpec(), PodSpec()])
        assert multi.is_distributed

    def test_states(self) -> None:
        assert len(JobState) == 6


class TestSchedulerConfig:
    """Tests for scheduler configuration."""

    def test_defaults(self) -> None:
        config = SchedulerConfig()
        assert config.strategy == SchedulingStrategy.BIN_PACK
        assert config.enable_preemption is True

    def test_custom(self) -> None:
        config = SchedulerConfig(
            strategy=SchedulingStrategy.SPREAD,
            enable_preemption=False,
        )
        assert config.strategy == SchedulingStrategy.SPREAD


class TestSchedulingResult:
    """Tests for scheduling result."""

    def test_success(self) -> None:
        result = SchedulingResult(job_id="j1", success=True)
        assert result.success

    def test_failure(self) -> None:
        result = SchedulingResult(
            job_id="j1", success=False, reason="no GPUs"
        )
        assert not result.success
        assert "no GPUs" in result.reason


class TestClusterSnapshot:
    """Tests for cluster snapshot."""

    def test_defaults(self) -> None:
        snap = ClusterSnapshot()
        assert snap.total_nodes == 0
        assert snap.gpu_utilization == 0.0
