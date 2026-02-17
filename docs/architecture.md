# Architecture: Container GPU Scheduler

## Overview

A GPU-aware batch scheduler that implements bin-packing, gang scheduling, and priority-based preemption for ML training workloads. Simulates Kubernetes-style scheduling without requiring a real cluster.

## Component Architecture

```
┌──────────────────────────────────────────────────┐
│                   GPUCluster                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │NodeResrc │  │NodeResrc │  │NodeResrc │       │
│  │8xA100    │  │8xA100    │  │4xT4     │       │
│  │[GPUSlot] │  │[GPUSlot] │  │[GPUSlot] │       │
│  └──────────┘  └──────────┘  └──────────┘       │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │           Scheduling Layer               │    │
│  │  ┌────────────┐  ┌───────────────────┐  │    │
│  │  │BinPack     │  │GangScheduler      │  │    │
│  │  │Scheduler   │  │(all-or-nothing)   │  │    │
│  │  └────────────┘  └───────────────────┘  │    │
│  │  ┌────────────┐  ┌───────────────────┐  │    │
│  │  │Spread      │  │Preemption Engine  │  │    │
│  │  │Scheduler   │  │(priority-based)   │  │    │
│  │  └────────────┘  └───────────────────┘  │    │
│  └──────────────────────────────────────────┘    │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │  Job Queue: [JobSpec] -> SchedulingResult │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

## Scheduling Strategies

### Bin-Packing

Consolidates workloads on fewer nodes. Scores nodes by current utilization (higher = preferred). Reduces GPU fragmentation and allows idle nodes to be powered down.

### Spread

Distributes workloads across nodes. Scores nodes by inverse utilization (lower = preferred). Improves fault tolerance but increases node count.

### Gang Scheduling

All-or-nothing placement for distributed training. If any pod cannot be placed, no pods are placed. Prevents resource deadlocks in multi-GPU training.

### Priority Preemption

High-priority jobs can evict lower-priority ones. Preemption threshold prevents unnecessary churn. Lowest-priority jobs are evicted first.

## Key Design Decisions

| Decision                     | Rationale                                      | Alternative                   |
| ---------------------------- | ---------------------------------------------- | ----------------------------- |
| Simulated cluster            | No K8s dependency, deterministic testing       | kopf operator on real cluster |
| Per-GPU slot tracking        | Fine-grained allocation, supports MIG          | Per-node GPU count only       |
| Dataclasses for resources    | Low overhead on hot path                       | Pydantic everywhere           |
| Priority integer (0-100)     | Simple, comparable                             | Priority classes              |
| Gang places large pods first | Reduces fragmentation                          | FIFO pod ordering             |
| Configurable threshold       | Prevents priority-1 from preempting priority-0 | Always allow preemption       |
