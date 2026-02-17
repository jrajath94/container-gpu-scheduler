"""Tests for core GPU scheduling algorithms."""

from __future__ import annotations

import pytest

from container_gpu_scheduler.core import (
    BinPackScheduler,
    GPUCluster,
    GangScheduler,
    SpreadScheduler,
)
from container_gpu_scheduler.exceptions import (
    GangSchedulingError,
    InvalidJobError,
)
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


# ── BinPackScheduler Tests ─────────────────────────────────────────────────────


class TestBinPackScheduler:
    """Tests for bin-packing scheduler."""

    def test_prefers_busy_nodes(self) -> None:
        """Bin-packing should prefer nodes with higher utilization."""
        scheduler = BinPackScheduler()
        nodes = [
            NodeResources(node_id="empty", total_gpus=8),
            NodeResources(node_id="half", total_gpus=8),
        ]
        # Fill half the GPUs on node "half"
        nodes[1].allocate(4, "existing-job")

        pod = PodSpec(pod_id="p1", gpu_count=2)
        scored = scheduler.score_nodes(nodes, pod, GPUType.A100_80GB)

        # "half" should score higher (more utilized)
        assert scored[0][0].node_id == "half"

    def test_schedule_pod(self) -> None:
        scheduler = BinPackScheduler()
        nodes = [NodeResources(node_id="n0", total_gpus=4)]
        pod = PodSpec(pod_id="p1", gpu_count=2)
        decision = scheduler.schedule_pod(nodes, pod, GPUType.A100_80GB, "j1")
        assert decision is not None
        assert decision.node_id == "n0"
        assert len(decision.gpu_ids) == 2

    def test_schedule_pod_no_fit(self) -> None:
        scheduler = BinPackScheduler()
        nodes = [NodeResources(node_id="n0", total_gpus=1)]
        pod = PodSpec(pod_id="p1", gpu_count=4)
        decision = scheduler.schedule_pod(nodes, pod, GPUType.A100_80GB, "j1")
        assert decision is None

    def test_wrong_gpu_type(self) -> None:
        scheduler = BinPackScheduler()
        nodes = [NodeResources(node_id="n0", total_gpus=8, gpu_type=GPUType.T4)]
        pod = PodSpec(pod_id="p1", gpu_count=1)
        decision = scheduler.schedule_pod(nodes, pod, GPUType.A100_80GB, "j1")
        assert decision is None


# ── SpreadScheduler Tests ──────────────────────────────────────────────────────


class TestSpreadScheduler:
    """Tests for spread scheduler."""

    def test_prefers_empty_nodes(self) -> None:
        """Spread should prefer nodes with lower utilization."""
        scheduler = SpreadScheduler()
        nodes = [
            NodeResources(node_id="empty", total_gpus=8),
            NodeResources(node_id="half", total_gpus=8),
        ]
        nodes[1].allocate(4, "existing")

        pod = PodSpec(pod_id="p1", gpu_count=2)
        scored = scheduler.score_nodes(nodes, pod, GPUType.A100_80GB)

        assert scored[0][0].node_id == "empty"


# ── GangScheduler Tests ───────────────────────────────────────────────────────


class TestGangScheduler:
    """Tests for gang scheduling."""

    def test_all_or_nothing_success(self) -> None:
        scheduler = GangScheduler()
        nodes = [
            NodeResources(node_id="n0", total_gpus=8),
            NodeResources(node_id="n1", total_gpus=8),
        ]
        job = create_training_job("gang-test", 2, 4, gang=True)
        decisions = scheduler.schedule_gang(nodes, job)
        assert len(decisions) == 2
        total_gpus = sum(len(d.gpu_ids) for d in decisions)
        assert total_gpus == 8

    def test_all_or_nothing_failure(self) -> None:
        scheduler = GangScheduler()
        nodes = [NodeResources(node_id="n0", total_gpus=4)]
        job = create_training_job("too-big", 2, 4, gang=True)
        with pytest.raises(GangSchedulingError):
            scheduler.schedule_gang(nodes, job)

    def test_large_pods_placed_first(self) -> None:
        """Larger pods should be placed before smaller ones."""
        scheduler = GangScheduler()
        nodes = [
            NodeResources(node_id="n0", total_gpus=8),
            NodeResources(node_id="n1", total_gpus=4),
        ]
        job = JobSpec(
            job_id="mixed",
            gpu_type=GPUType.A100_80GB,
            gang_schedule=True,
            pods=[
                PodSpec(pod_id="small", gpu_count=2),
                PodSpec(pod_id="large", gpu_count=6),
            ],
        )
        decisions = scheduler.schedule_gang(nodes, job)
        # Large pod should be on n0 (8 GPUs), small on remainder
        large_decision = next(d for d in decisions if d.pod_id == "large")
        assert large_decision.node_id == "n0"


# ── GPUCluster Tests ──────────────────────────────────────────────────────────


class TestGPUCluster:
    """Tests for the GPU cluster orchestrator."""

    def test_add_nodes(self, small_cluster: GPUCluster) -> None:
        assert len(small_cluster.nodes) == 4
        snap = small_cluster.snapshot()
        assert snap.total_gpus == 32
        assert snap.free_gpus == 32

    def test_submit_small_job(
        self, small_cluster: GPUCluster, small_job: JobSpec
    ) -> None:
        result = small_cluster.submit_job(small_job)
        assert result.success
        assert small_job.state == JobState.SCHEDULED

    def test_submit_large_job(
        self, small_cluster: GPUCluster, large_job: JobSpec
    ) -> None:
        result = small_cluster.submit_job(large_job)
        assert result.success

    def test_submit_distributed_job(
        self, small_cluster: GPUCluster, distributed_job: JobSpec
    ) -> None:
        result = small_cluster.submit_job(distributed_job)
        assert result.success
        assert len(result.decisions) == 4

    def test_release_job(
        self, small_cluster: GPUCluster, small_job: JobSpec
    ) -> None:
        small_cluster.submit_job(small_job)
        released = small_cluster.release_job(small_job.job_id)
        assert released == 2
        snap = small_cluster.snapshot()
        assert snap.free_gpus == 32

    def test_preemption(self, small_cluster: GPUCluster) -> None:
        """High-priority job should preempt low-priority ones."""
        # Fill cluster with low-priority jobs
        for i in range(8):
            job = create_training_job(f"low-{i}", 1, 4, priority=10)
            small_cluster.submit_job(job)

        snap = small_cluster.snapshot()
        assert snap.free_gpus == 0

        # Submit high-priority job
        urgent = create_training_job("urgent", 1, 4, priority=95)
        result = small_cluster.submit_job(urgent)
        assert result.success
        assert result.total_preemptions > 0

    def test_preemption_disabled(self) -> None:
        config = SchedulerConfig(enable_preemption=False)
        cluster = GPUCluster(config)
        cluster.add_nodes(1, 4, GPUType.A100_80GB)

        low = create_training_job("low", 1, 4, priority=10)
        cluster.submit_job(low)

        high = create_training_job("high", 1, 4, priority=90)
        result = cluster.submit_job(high)
        assert not result.success

    def test_invalid_job_no_pods(self, small_cluster: GPUCluster) -> None:
        job = JobSpec(job_id="bad", name="no-pods")
        with pytest.raises(InvalidJobError):
            small_cluster.submit_job(job)

    def test_duplicate_job_id(
        self, small_cluster: GPUCluster, small_job: JobSpec
    ) -> None:
        small_cluster.submit_job(small_job)
        with pytest.raises(InvalidJobError):
            small_cluster.submit_job(small_job)

    def test_snapshot(
        self, small_cluster: GPUCluster, small_job: JobSpec
    ) -> None:
        small_cluster.submit_job(small_job)
        snap = small_cluster.snapshot()
        assert snap.total_nodes == 4
        assert snap.running_jobs == 1
        assert snap.gpu_utilization > 0

    def test_multiple_jobs_utilization(
        self, small_cluster: GPUCluster
    ) -> None:
        for i in range(4):
            job = create_training_job(f"job-{i}", 1, 4, priority=50)
            result = small_cluster.submit_job(job)
            assert result.success

        snap = small_cluster.snapshot()
        assert snap.free_gpus == 16
        assert snap.gpu_utilization == pytest.approx(0.5, abs=0.01)

    def test_spread_strategy(self) -> None:
        config = SchedulerConfig(strategy=SchedulingStrategy.SPREAD)
        cluster = GPUCluster(config)
        cluster.add_nodes(4, 8, GPUType.A100_80GB)

        # Submit two small jobs
        j1 = create_training_job("spread-1", 1, 2)
        j2 = create_training_job("spread-2", 1, 2)
        cluster.submit_job(j1)
        cluster.submit_job(j2)

        # With spread, jobs should be on different nodes
        nodes_used = set()
        for job in [j1, j2]:
            for pod in job.pods:
                if pod.assigned_node:
                    nodes_used.add(pod.assigned_node)
        assert len(nodes_used) == 2

    def test_bin_pack_consolidation(self) -> None:
        config = SchedulerConfig(strategy=SchedulingStrategy.BIN_PACK)
        cluster = GPUCluster(config)
        cluster.add_nodes(4, 8, GPUType.A100_80GB)

        # Put a small job on node-0 first
        seed = create_training_job("seed", 1, 1)
        cluster.submit_job(seed)

        # Next job should also go to node-0 (bin-packing)
        j2 = create_training_job("pack", 1, 2)
        cluster.submit_job(j2)

        # Both should be on the same node
        seed_node = seed.pods[0].assigned_node
        j2_node = j2.pods[0].assigned_node
        assert seed_node == j2_node

    def test_get_preemptible_jobs(
        self, small_cluster: GPUCluster
    ) -> None:
        low = create_training_job("low", 1, 2, priority=10)
        med = create_training_job("med", 1, 2, priority=50)
        small_cluster.submit_job(low)
        small_cluster.submit_job(med)

        preemptible = small_cluster.get_preemptible_jobs(60)
        assert len(preemptible) == 2
        # Sorted by priority ascending
        assert preemptible[0].priority <= preemptible[1].priority

    def test_mixed_gpu_types(self) -> None:
        cluster = GPUCluster()
        cluster.add_nodes(2, 8, GPUType.A100_80GB)
        cluster.add_nodes(2, 4, GPUType.T4)

        # Job requiring A100 should not be placed on T4
        job = create_training_job("a100-job", 1, 2, gpu_type=GPUType.A100_80GB)
        result = cluster.submit_job(job)
        assert result.success
        node_id = job.pods[0].assigned_node
        assert node_id is not None
        node = next(n for n in cluster.nodes if n.node_id == node_id)
        assert node.gpu_type == GPUType.A100_80GB

    def test_insufficient_resources(self) -> None:
        cluster = GPUCluster(SchedulerConfig(enable_preemption=False))
        cluster.add_nodes(1, 2, GPUType.A100_80GB)
        job = create_training_job("too-big", 1, 4)
        result = cluster.submit_job(job)
        assert not result.success

    @pytest.mark.parametrize(
        "num_pods,gpus_per_pod,expected_success",
        [
            (1, 4, True),
            (2, 4, True),
            (4, 4, True),
            (4, 8, True),
            (5, 8, False),  # 40 GPUs needed, only 32 available
        ],
    )
    def test_various_job_sizes(
        self,
        small_cluster: GPUCluster,
        num_pods: int,
        gpus_per_pod: int,
        expected_success: bool,
    ) -> None:
        job = create_training_job(
            "size-test", num_pods, gpus_per_pod, gang=True
        )
        result = small_cluster.submit_job(job)
        assert result.success == expected_success
