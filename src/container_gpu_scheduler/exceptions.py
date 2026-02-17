"""Custom exceptions for GPU scheduler."""


class SchedulerError(Exception):
    """Base exception for all scheduler errors."""


class InsufficientResourcesError(SchedulerError):
    """Raised when a cluster cannot satisfy resource requirements."""


class GangSchedulingError(SchedulerError):
    """Raised when gang scheduling cannot place all pods."""


class PreemptionError(SchedulerError):
    """Raised when preemption cannot free enough resources."""


class InvalidJobError(SchedulerError):
    """Raised when a job specification is invalid."""


class NodeCapacityError(SchedulerError):
    """Raised when a node's capacity is exceeded."""
