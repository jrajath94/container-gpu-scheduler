"""Core GPU-aware scheduling algorithms.

Three components with clear separation:
  1. BinPackScheduler -- consolidates workloads on fewer nodes
  2. GangScheduler -- all-or-nothing placement for distributed training
  3. GPUCluster -- simulated cluster with scheduling, preemption, and metrics

The scheduler supports priority-based preemption: high-priority training
jobs can evict lower-priority jobs to claim GPU resources.
"""

from __future__ import annotations

import logging
from typing import Optional

from container_gpu_scheduler.exceptions import (
    GangSchedulingError,
    InsufficientResourcesError,
    InvalidJobError,
    PreemptionError,
)
from container_gpu_scheduler.models import (
    ClusterSnapshot,
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
from container_gpu_scheduler.utils import (
    bin_pack_score,
    can_fit_pod,
    spread_score,
)

logger = logging.getLogger(__name__)


class BinPackScheduler:
    """Bin-packing scheduler that consolidates GPU workloads.

    Scores nodes by utilization and prefers placing pods on
    already-busy nodes to maximize GPU density and reduce
    fragmentation.
    """

    def score_nodes(
        self,
        nodes: list[NodeResources],
        pod: PodSpec,
        gpu_type: GPUType,
    ) -> list[tuple[NodeResources, float]]:
        """Score and rank eligible nodes for a pod.

        Args:
            nodes: Available nodes.
            pod: Pod to place.
            gpu_type: Required GPU type.

        Returns:
            List of (node, score) tuples, sorted by score descending.
        """
        scored: list[tuple[NodeResources, float]] = []
        for node in nodes:
            if can_fit_pod(node, pod, gpu_type):
                score = bin_pack_score(node)
                scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def schedule_pod(
        self,
        nodes: list[NodeResources],
        pod: PodSpec,
        gpu_type: GPUType,
        job_id: str,
    ) -> Optional[SchedulingDecision]:
        """Schedule a single pod using bin-packing.

        Args:
            nodes: Available nodes.
            pod: Pod to place.
            gpu_type: Required GPU type.
            job_id: Parent job ID.

        Returns:
            SchedulingDecision if successful, None otherwise.
        """
        scored = self.score_nodes(nodes, pod, gpu_type)
        if not scored:
            return None

        best_node = scored[0][0]
        gpu_ids = best_node.allocate(pod.gpu_count, job_id)
        pod.assigned_node = best_node.node_id
        pod.assigned_gpus = gpu_ids

        logger.debug(
            "Bin-packed pod %s on %s (gpus=%s, util=%.1f%%)",
            pod.pod_id,
            best_node.node_id,
            gpu_ids,
            best_node.utilization * 100,
        )

        return SchedulingDecision(
            pod_id=pod.pod_id,
            node_id=best_node.node_id,
            gpu_ids=gpu_ids,
        )


class SpreadScheduler:
    """Spread scheduler that distributes workloads across nodes.

    Prefers placing pods on least-utilized nodes to balance
    load and improve fault tolerance.
    """

    def score_nodes(
        self,
        nodes: list[NodeResources],
        pod: PodSpec,
        gpu_type: GPUType,
    ) -> list[tuple[NodeResources, float]]:
        """Score nodes for spreading (prefer emptier nodes).

        Args:
            nodes: Available nodes.
            pod: Pod to place.
            gpu_type: Required GPU type.

        Returns:
            Scored nodes sorted by emptiness descending.
        """
        scored: list[tuple[NodeResources, float]] = []
        for node in nodes:
            if can_fit_pod(node, pod, gpu_type):
                score = spread_score(node)
                scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def schedule_pod(
        self,
        nodes: list[NodeResources],
        pod: PodSpec,
        gpu_type: GPUType,
        job_id: str,
    ) -> Optional[SchedulingDecision]:
        """Schedule a single pod using spread strategy.

        Args:
            nodes: Available nodes.
            pod: Pod to place.
            gpu_type: Required GPU type.
            job_id: Parent job ID.

        Returns:
            SchedulingDecision if successful, None otherwise.
        """
        scored = self.score_nodes(nodes, pod, gpu_type)
        if not scored:
            return None

        best_node = scored[0][0]
        gpu_ids = best_node.allocate(pod.gpu_count, job_id)
        pod.assigned_node = best_node.node_id
        pod.assigned_gpus = gpu_ids

        return SchedulingDecision(
            pod_id=pod.pod_id,
            node_id=best_node.node_id,
            gpu_ids=gpu_ids,
        )


class GangScheduler:
    """Gang scheduler for distributed training jobs.

    Ensures all pods in a job are placed simultaneously.
    If any pod cannot be placed, no pods are placed (all-or-nothing).
    Uses bin-packing within the gang for consolidation.
    """

    def __init__(self) -> None:
        self._bin_packer = BinPackScheduler()

    def schedule_gang(
        self,
        nodes: list[NodeResources],
        job: JobSpec,
    ) -> list[SchedulingDecision]:
        """Schedule all pods in a job atomically.

        Args:
            nodes: Available nodes.
            job: Job with multiple pods.

        Returns:
            List of scheduling decisions for all pods.

        Raises:
            GangSchedulingError: If not all pods can be placed.
        """
        # Trial placement -- check without committing
        decisions: list[SchedulingDecision] = []
        allocated_per_node: dict[str, int] = {}

        # Sort pods by GPU count descending (place large pods first)
        sorted_pods = sorted(
            job.pods, key=lambda p: p.gpu_count, reverse=True
        )

        for pod in sorted_pods:
            placed = False
            scored = self._bin_packer.score_nodes(
                nodes, pod, job.gpu_type
            )

            for node, _ in scored:
                trial_used = allocated_per_node.get(node.node_id, 0)
                available = node.free_gpus - trial_used
                if available >= pod.gpu_count:
                    allocated_per_node[node.node_id] = (
                        trial_used + pod.gpu_count
                    )
                    decisions.append(
                        SchedulingDecision(
                            pod_id=pod.pod_id,
                            node_id=node.node_id,
                            gpu_ids=[],  # Filled during commit
                        )
                    )
                    placed = True
                    break

            if not placed:
                raise GangSchedulingError(
                    f"Cannot place pod {pod.pod_id} "
                    f"({pod.gpu_count} GPUs): insufficient resources"
                )

        # Commit allocations
        node_map = {n.node_id: n for n in nodes}
        for decision in decisions:
            pod = next(
                p for p in job.pods if p.pod_id == decision.pod_id
            )
            node = node_map[decision.node_id]
            gpu_ids = node.allocate(pod.gpu_count, job.job_id)
            decision.gpu_ids = gpu_ids
            pod.assigned_node = decision.node_id
            pod.assigned_gpus = gpu_ids

        logger.info(
            "Gang-scheduled job %s: %d pods across %d nodes",
            job.job_id,
            len(decisions),
            len(set(d.node_id for d in decisions)),
        )

        return decisions


class GPUCluster:
    """Simulated GPU cluster with scheduling and preemption.

    Manages a set of compute nodes and schedules training jobs
    using configurable strategies.

    Args:
        config: Scheduler configuration.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self._config = config or SchedulerConfig()
        self._nodes: list[NodeResources] = []
        self._jobs: dict[str, JobSpec] = {}
        self._bin_packer = BinPackScheduler()
        self._spread_scheduler = SpreadScheduler()
        self._gang_scheduler = GangScheduler()

        logger.info("GPU cluster initialized with strategy=%s", self._config.strategy)

    @property
    def nodes(self) -> list[NodeResources]:
        """Access cluster nodes."""
        return self._nodes

    @property
    def jobs(self) -> dict[str, JobSpec]:
        """Access all jobs."""
        return self._jobs

    def add_node(self, node: NodeResources) -> None:
        """Add a compute node to the cluster.

        Args:
            node: Node to add.
        """
        self._nodes.append(node)
        logger.debug(
            "Added node %s: %d x %s GPUs",
            node.node_id, node.total_gpus, node.gpu_type.value,
        )

    def add_nodes(
        self,
        count: int,
        gpus_per_node: int = 8,
        gpu_type: GPUType = GPUType.A100_80GB,
    ) -> list[str]:
        """Add multiple identical nodes to the cluster.

        Args:
            count: Number of nodes to add.
            gpus_per_node: GPUs per node.
            gpu_type: GPU type for all nodes.

        Returns:
            List of created node IDs.
        """
        node_ids: list[str] = []
        base = len(self._nodes)
        for i in range(count):
            node_id = f"node-{base + i}"
            node = NodeResources(
                node_id=node_id,
                total_gpus=gpus_per_node,
                gpu_type=gpu_type,
            )
            self._nodes.append(node)
            node_ids.append(node_id)

        logger.info(
            "Added %d nodes (%d x %s each)",
            count, gpus_per_node, gpu_type.value,
        )
        return node_ids

    def submit_job(self, job: JobSpec) -> SchedulingResult:
        """Submit and schedule a training job.

        Args:
            job: Job to schedule.

        Returns:
            Scheduling result.

        Raises:
            InvalidJobError: If job spec is invalid.
        """
        if not job.pods:
            raise InvalidJobError(f"Job {job.job_id} has no pods")
        if job.job_id in self._jobs:
            raise InvalidJobError(f"Job {job.job_id} already exists")

        self._jobs[job.job_id] = job

        result = self._try_schedule(job)

        if result.success:
            job.state = JobState.SCHEDULED
            logger.info(
                "Scheduled job %s (%s): %d pods, %d GPUs",
                job.job_id,
                job.name,
                job.num_pods,
                job.total_gpus,
            )
        elif self._config.enable_preemption:
            result = self._try_preempt_and_schedule(job)
            if result.success:
                job.state = JobState.SCHEDULED

        if not result.success:
            job.state = JobState.PENDING
            logger.warning(
                "Could not schedule job %s: %s",
                job.job_id, result.reason,
            )

        return result

    def release_job(self, job_id: str) -> int:
        """Release all resources held by a job.

        Args:
            job_id: Job to release.

        Returns:
            Number of GPUs released.
        """
        if job_id not in self._jobs:
            return 0

        total_released = 0
        for node in self._nodes:
            total_released += node.release(job_id)

        job = self._jobs[job_id]
        job.state = JobState.COMPLETED
        for pod in job.pods:
            pod.assigned_node = None
            pod.assigned_gpus = []

        logger.info("Released job %s: %d GPUs freed", job_id, total_released)
        return total_released

    def preempt_job(self, job_id: str, preempted_by: str) -> int:
        """Preempt a running job to free resources.

        Args:
            job_id: Job to preempt.
            preempted_by: Job ID requesting preemption.

        Returns:
            Number of GPUs freed.
        """
        if job_id not in self._jobs:
            return 0

        job = self._jobs[job_id]
        job.state = JobState.PREEMPTED
        job.preempted_by = preempted_by

        total_freed = 0
        for node in self._nodes:
            total_freed += node.release(job_id)

        for pod in job.pods:
            pod.assigned_node = None
            pod.assigned_gpus = []

        logger.info(
            "Preempted job %s (priority=%d) by %s: %d GPUs freed",
            job_id, job.priority, preempted_by, total_freed,
        )
        return total_freed

    def snapshot(self) -> ClusterSnapshot:
        """Take a point-in-time snapshot of cluster state.

        Returns:
            Current cluster metrics.
        """
        total_gpus = sum(n.total_gpus for n in self._nodes)
        free_gpus = sum(n.free_gpus for n in self._nodes)

        running = sum(
            1 for j in self._jobs.values()
            if j.state in (JobState.SCHEDULED, JobState.RUNNING)
        )
        pending = sum(
            1 for j in self._jobs.values()
            if j.state == JobState.PENDING
        )

        util = 1.0 - (free_gpus / total_gpus) if total_gpus > 0 else 0.0
        per_node = {n.node_id: n.utilization for n in self._nodes}

        return ClusterSnapshot(
            total_nodes=len(self._nodes),
            total_gpus=total_gpus,
            free_gpus=free_gpus,
            running_jobs=running,
            pending_jobs=pending,
            gpu_utilization=util,
            per_node_utilization=per_node,
        )

    def get_preemptible_jobs(
        self, min_priority: int
    ) -> list[JobSpec]:
        """Find jobs that can be preempted by a higher-priority job.

        Args:
            min_priority: Only return jobs with priority below this.

        Returns:
            List of preemptible jobs sorted by priority ascending.
        """
        preemptible = [
            j for j in self._jobs.values()
            if j.state in (JobState.SCHEDULED, JobState.RUNNING)
            and j.priority < min_priority
        ]
        preemptible.sort(key=lambda j: j.priority)
        return preemptible

    # ── Internal Scheduling ──────────────────────────────────

    def _try_schedule(self, job: JobSpec) -> SchedulingResult:
        """Attempt to schedule a job without preemption.

        Args:
            job: Job to schedule.

        Returns:
            Scheduling result.
        """
        if job.gang_schedule or job.is_distributed:
            return self._schedule_gang(job)
        return self._schedule_single(job)

    def _schedule_single(self, job: JobSpec) -> SchedulingResult:
        """Schedule a single-pod job.

        Args:
            job: Job with one or more independent pods.

        Returns:
            Scheduling result.
        """
        decisions: list[SchedulingDecision] = []

        for pod in job.pods:
            if self._config.strategy == SchedulingStrategy.SPREAD:
                decision = self._spread_scheduler.schedule_pod(
                    self._nodes, pod, job.gpu_type, job.job_id,
                )
            else:
                decision = self._bin_packer.schedule_pod(
                    self._nodes, pod, job.gpu_type, job.job_id,
                )

            if decision is None:
                # Rollback any already-placed pods
                self._rollback_decisions(decisions, job.job_id)
                return SchedulingResult(
                    job_id=job.job_id,
                    success=False,
                    reason=f"Cannot place pod {pod.pod_id}: "
                           f"insufficient {job.gpu_type.value} GPUs",
                )
            decisions.append(decision)

        return SchedulingResult(
            job_id=job.job_id,
            success=True,
            decisions=decisions,
        )

    def _schedule_gang(self, job: JobSpec) -> SchedulingResult:
        """Schedule a distributed job using gang scheduling.

        Args:
            job: Distributed training job.

        Returns:
            Scheduling result.
        """
        try:
            decisions = self._gang_scheduler.schedule_gang(
                self._nodes, job,
            )
            return SchedulingResult(
                job_id=job.job_id,
                success=True,
                decisions=decisions,
            )
        except GangSchedulingError as e:
            return SchedulingResult(
                job_id=job.job_id,
                success=False,
                reason=str(e),
            )

    def _try_preempt_and_schedule(
        self, job: JobSpec
    ) -> SchedulingResult:
        """Try to preempt lower-priority jobs to schedule this one.

        Args:
            job: High-priority job needing resources.

        Returns:
            Scheduling result.
        """
        threshold = self._config.preemption_priority_threshold
        preemptible = self.get_preemptible_jobs(job.priority)

        if not preemptible:
            return SchedulingResult(
                job_id=job.job_id,
                success=False,
                reason="No preemptible jobs available",
            )

        # Preempt lowest-priority jobs until enough GPUs
        preempted_ids: list[str] = []
        gpus_freed = 0
        needed = job.total_gpus

        for victim in preemptible:
            if gpus_freed >= needed:
                break
            priority_diff = job.priority - victim.priority
            if priority_diff < threshold:
                continue
            freed = self.preempt_job(victim.job_id, job.job_id)
            gpus_freed += freed
            preempted_ids.append(victim.job_id)

        # Try scheduling again
        result = self._try_schedule(job)
        result.total_preemptions = len(preempted_ids)

        if result.success:
            for decision in result.decisions:
                decision.preempted_jobs = preempted_ids

        return result

    def _rollback_decisions(
        self, decisions: list[SchedulingDecision], job_id: str
    ) -> None:
        """Rollback partial scheduling decisions.

        Args:
            decisions: Decisions to undo.
            job_id: Job being rolled back.
        """
        node_map = {n.node_id: n for n in self._nodes}
        for decision in decisions:
            if decision.node_id in node_map:
                node_map[decision.node_id].release(job_id)
                pod = next(
                    (p for j in self._jobs.values()
                     for p in j.pods if p.pod_id == decision.pod_id),
                    None,
                )
                if pod:
                    pod.assigned_node = None
                    pod.assigned_gpus = []
