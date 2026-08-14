"""Low-level PyTorch/CUDA C++ memory subsystem leak and fragmentation detector."""

from __future__ import annotations
import gc
import dataclasses
from typing import Dict, List, Optional
import torch


@dataclasses.dataclass(frozen=True)
class MemorySnapshot:
    """Immutable GPU memory state snapshot."""

    allocated_bytes: int
    reserved_bytes: int
    active_bytes: int
    inactive_split_bytes: int
    max_allocated_bytes: int
    free_bytes: int
    total_bytes: int
    fragmentation_ratio: float


class GPUMemoryTracker:
    """Tracks GPU allocation spikes, fragmentation levels, and detects memory leaks."""

    def __init__(self, device: int = 0) -> None:
        self.device = device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for GPUMemoryTracker.")
        self.snapshots: List[tuple[str, MemorySnapshot]] = []

    def capture(self, label: str) -> MemorySnapshot:
        """Records current GPU memory allocator telemetry."""
        gc.collect()
        torch.cuda.empty_cache()  # Standardize comparison baseline

        stats = torch.cuda.memory_stats(self.device)
        mem_info = torch.cuda.mem_get_info(self.device)

        allocated = stats.get("allocated_bytes.all.current", 0)
        reserved = stats.get("reserved_bytes.all.current", 0)
        active = stats.get("active_bytes.all.current", 0)
        inactive_split = stats.get("inactive_split_bytes.all.current", 0)
        max_allocated = stats.get("allocated_bytes.all.peak", 0)

        free_gpu, total_gpu = mem_info

        # Calculate fragmentation: unallocated reserved memory ratio
        unallocated_reserved = reserved - allocated
        frag_ratio = (unallocated_reserved / reserved) if reserved > 0 else 0.0

        snapshot = MemorySnapshot(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            active_bytes=active,
            inactive_split_bytes=inactive_split,
            max_allocated_bytes=max_allocated,
            free_bytes=free_gpu,
            total_bytes=total_gpu,
            fragmentation_ratio=frag_ratio,
        )

        self.snapshots.append((label, snapshot))
        return snapshot

    def assert_no_leak(
        self,
        baseline_label: str,
        target_label: str,
        threshold_mb: float = 1.0,
    ) -> None:
        """Verifies allocated memory growth between two checkpoints does not exceed threshold."""
        snaps = dict(self.snapshots)
        if baseline_label not in snaps or target_label not in snaps:
            raise KeyError(
                f"Missing snapshot comparison labels: {baseline_label}, {target_label}"
            )

        base = snaps[baseline_label]
        target = snaps[target_label]

        diff_bytes = target.allocated_bytes - base.allocated_bytes
        diff_mb = diff_bytes / (1024 * 1024)

        if diff_mb > threshold_mb:
            raise AssertionError(
                f"GPU Memory Leak Detected between '{baseline_label}' and '{target_label}'! "
                f"Leaked: {diff_mb:.3f} MB (Threshold: {threshold_mb:.3f} MB). "
                f"Base Allocated: {base.allocated_bytes / 1e6:.2f}MB -> Target: {target.allocated_bytes / 1e6:.2f}MB"
            )

    def print_summary(self) -> None:
        """Prints tabular overview of captured snapshots."""
        print(
            f"\n{'Label':<25} | {'Allocated (MB)':<15} | {'Reserved (MB)':<15} | {'Frag %':<8}"
        )
        print("-" * 70)
        for label, snap in self.snapshots:
            alloc_mb = snap.allocated_bytes / (1024 * 1024)
            res_mb = snap.reserved_bytes / (1024 * 1024)
            frag_pct = snap.fragmentation_ratio * 100
            print(
                f"{label:<25} | {alloc_mb:<15.2f} | {res_mb:<15.2f} | {frag_pct:<8.2f}%"
            )
