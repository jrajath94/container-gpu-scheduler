"""Data models for the GPU-aware scheduler.

Defines node resources, GPU types, jobs, pods, and scheduling results
using Pydantic for validation and dataclasses for hot-path structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_GPUS_PER_NODE = 8
MAX_NODES = 256
DEFAULT_PREEMPTION_PRIORITY_THRESHOLD = 50


class GPUType(str, Enum):
    """Supported GPU types with their memory capacities."""

    A100_40GB = "a100-40gb"
    A100_80GB = "a100-80gb"
    H100 = "h100"
    L40S = "l40s"
    T4 = "t4"
    V100 = "v100"


GPU_MEMORY_GB: dict[GPUType, int] = {
    GPUType.A100_40GB: 40,
    GPUType.A100_80GB: 80,
    GPUType.H100: 80,
    GPUType.L40S: 48,
    GPUType.T4: 16,
    GPUType.V100: 32,
}

GPU_COMPUTE_UNITS: dict[GPUType, float] = {
    GPUType.A100_40GB: 1.0,
    GPUType.A100_80GB: 1.2,
    GPUType.H100: 2.0,
    GPUType.L40S: 0.8,
    GPUType.T4: 0.3,
    GPUType.V100: 0.5,
}


class JobState(str, Enum):
    """Lifecycle states for a training job."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PREEMPTED = "preempted"
    COMPLETED = "completed"
    FAILED = "failed"


class SchedulingStrategy(str, Enum):
    """Available scheduling strategies."""

    BIN_PACK = "bin_pack"
    SPREAD = "spread"
    GANG = "gang"


@dataclass
class GPUSlot:
    """A single GPU slot on a node.

    Attributes:
        gpu_id: Unique identifier within the node.
        gpu_type: Type/model of GPU.
        allocated_to: Job ID if allocated, None if free.
        memory_gb: Total GPU memory.
        compute_units: Relative compute power.
    """

    gpu_id: int = 0
    gpu_type: GPUType = GPUType.A100_80GB
    allocated_to: str | None = None
    memory_gb: int = 80
    compute_units: float = 1.2

    @property
    def is_free(self) -> bool:
        """Check if this GPU slot is available."""
        return self.allocated_to is None


@dataclass
class NodeResources:
    """Resources available on a single compute node.

    Attributes:
        node_id: Unique node identifier.
        total_gpus: Total GPU count.
        gpu_type: Type of GPUs on this node.
        total_cpu_cores: Total CPU cores.
        total_memory_gb: Total system RAM in GB.
        gpu_slots: Individual GPU slot tracking.
        labels: Metadata labels (e.g., zone, pool).
    """

    node_id: str = ""
    total_gpus: int = 8
    gpu_type: GPUType = GPUType.A100_80GB
    total_cpu_cores: int = 96
    total_memory_gb: int = 1024
    gpu_slots: list[GPUSlot] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.gpu_slots:
            mem = GPU_MEMORY_GB.get(self.gpu_type, 80)
            cu = GPU_COMPUTE_UNITS.get(self.gpu_type, 1.0)
            self.gpu_slots = [
                GPUSlot(
                    gpu_id=i,
                    gpu_type=self.gpu_type,
                    memory_gb=mem,
                    compute_units=cu,
                )
                for i in range(self.total_gpus)
            ]

    @property
    def free_gpus(self) -> int:
        """Count of available GPU slots."""
        return sum(1 for s in self.gpu_slots if s.is_free)

    @property
    def used_gpus(self) -> int:
        """Count of allocated GPU slots."""
        return self.total_gpus - self.free_gpus

    @property
    def utilization(self) -> float:
        """GPU utilization as a fraction [0, 1]."""
        if self.total_gpus == 0:
            return 0.0
        return self.used_gpus / self.total_gpus

    def allocate(self, count: int, job_id: str) -> list[int]:
        """Allocate GPUs to a job.

        Args:
            count: Number of GPUs to allocate.
            job_id: Job receiving the allocation.

        Returns:
            List of allocated GPU IDs.
        """
        allocated: list[int] = []
        for slot in self.gpu_slots:
            if slot.is_free and len(allocated) < count:
                slot.allocated_to = job_id
                allocated.append(slot.gpu_id)
        return allocated

    def release(self, job_id: str) -> int:
        """Release all GPUs allocated to a job.

        Args:
            job_id: Job to release.

        Returns:
            Number of GPUs released.
        """
        released = 0
        for slot in self.gpu_slots:
            if slot.allocated_to == job_id:
                slot.allocated_to = None
                released += 1
        return released

    def gpus_for_job(self, job_id: str) -> list[int]:
        """Get GPU IDs allocated to a specific job.

        Args:
            job_id: Job to query.

        Returns:
            List of GPU IDs.
        """
        return [s.gpu_id for s in self.gpu_slots if s.allocated_to == job_id]


@dataclass
class PodSpec:
    """Specification for a single pod in a job.

    Attributes:
        pod_id: Unique pod identifier.
        gpu_count: Number of GPUs required.
        cpu_cores: CPU cores required.
        memory_gb: System RAM required in GB.
        assigned_node: Node ID if scheduled.
        assigned_gpus: GPU IDs allocated.
    """

    pod_id: str = ""
    gpu_count: int = 1
    cpu_cores: int = 8
    memory_gb: int = 64
    assigned_node: str | None = None
    assigned_gpus: list[int] = field(default_factory=list)

    @property
    def is_scheduled(self) -> bool:
        """Check if this pod has been placed."""
        return self.assigned_node is not None


@dataclass
class JobSpec:
    """Specification for a training job.

    Attributes:
        job_id: Unique job identifier.
        name: Human-readable name.
        priority: Higher priority preempts lower (0-100).
        gpu_type: Required GPU type.
        pods: List of pod specifications.
        gang_schedule: Whether all pods must be placed together.
        state: Current job lifecycle state.
        preempted_by: Job ID that preempted this job.
    """

    job_id: str = ""
    name: str = ""
    priority: int = 50
    gpu_type: GPUType = GPUType.A100_80GB
    pods: list[PodSpec] = field(default_factory=list)
    gang_schedule: bool = False
    state: JobState = JobState.PENDING
    preempted_by: str | None = None

    @property
    def total_gpus(self) -> int:
        """Total GPUs needed across all pods."""
        return sum(p.gpu_count for p in self.pods)

    @property
    def num_pods(self) -> int:
        """Number of pods in this job."""
        return len(self.pods)

    @property
    def is_distributed(self) -> bool:
        """Check if this is a multi-pod distributed job."""
        return len(self.pods) > 1


@dataclass
class SchedulingDecision:
    """Result of a scheduling decision for a single pod.

    Attributes:
        pod_id: Pod being scheduled.
        node_id: Target node.
        gpu_ids: Allocated GPU IDs.
        preempted_jobs: Jobs that were preempted for this placement.
    """

    pod_id: str = ""
    node_id: str = ""
    gpu_ids: list[int] = field(default_factory=list)
    preempted_jobs: list[str] = field(default_factory=list)


@dataclass
class SchedulingResult:
    """Complete result of scheduling a job.

    Attributes:
        job_id: Job that was scheduled.
        success: Whether scheduling succeeded.
        decisions: Per-pod scheduling decisions.
        total_preemptions: Total jobs preempted.
        reason: Failure reason if unsuccessful.
    """

    job_id: str = ""
    success: bool = False
    decisions: list[SchedulingDecision] = field(default_factory=list)
    total_preemptions: int = 0
    reason: str = ""


class SchedulerConfig(BaseModel):
    """Configuration for the GPU scheduler.

    Attributes:
        strategy: Default scheduling strategy.
        enable_preemption: Allow priority-based preemption.
        preemption_priority_threshold: Min priority diff to preempt.
        max_retries: Max scheduling retries per job.
        prefer_consolidation: Pack jobs on fewer nodes.
    """

    strategy: SchedulingStrategy = SchedulingStrategy.BIN_PACK
    enable_preemption: bool = True
    preemption_priority_threshold: int = Field(
        default=DEFAULT_PREEMPTION_PRIORITY_THRESHOLD, ge=0, le=100
    )
    max_retries: int = Field(default=3, ge=1)
    prefer_consolidation: bool = True


@dataclass
class ClusterSnapshot:
    """Point-in-time snapshot of cluster state.

    Attributes:
        total_nodes: Number of nodes.
        total_gpus: Total GPUs across cluster.
        free_gpus: Available GPUs.
        running_jobs: Number of running jobs.
        pending_jobs: Number of pending jobs.
        gpu_utilization: Cluster-wide GPU utilization.
        per_node_utilization: Utilization per node.
    """

    total_nodes: int = 0
    total_gpus: int = 0
    free_gpus: int = 0
    running_jobs: int = 0
    pending_jobs: int = 0
    gpu_utilization: float = 0.0
    per_node_utilization: dict[str, float] = field(default_factory=dict)
