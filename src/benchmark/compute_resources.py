from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
import os
import psutil
import torch


@dataclass
class ComputeStats:
    tokens: int = 0
    seconds: float = 0.0
    peak_mem_gb: float = 0.0

    @property
    def tok_per_s(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def gpu_hours(self) -> float:
        return self.seconds / 3600.0


@contextmanager
def measure(stats: ComputeStats):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    t0 = perf_counter()
    try:
        yield stats
    finally:
        stats.seconds += perf_counter() - t0
        if torch.cuda.is_available():
            stats.peak_mem_gb = max(stats.peak_mem_gb,
                                    torch.cuda.max_memory_allocated() / (1024 ** 3))
        else:
            rss1 = proc.memory_info().rss
            stats.peak_mem_gb = max(stats.peak_mem_gb, rss1 / (1024 ** 3))
