"""GPU telemetry sampling via NVML + torch."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TelemetrySnapshot:
    t: float
    used_bytes: Optional[int] = None
    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    util_percent: Optional[int] = None
    torch_allocated_bytes: Optional[int] = None
    torch_reserved_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TelemetryAggregate:
    samples: List[TelemetrySnapshot] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def append(self, snap: TelemetrySnapshot) -> None:
        self.samples.append(snap)

    def summary(self) -> Dict[str, Any]:
        def _max(attr: str) -> Optional[int]:
            vals = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
            return max(vals) if vals else None

        def _avg(attr: str) -> Optional[float]:
            vals = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
            return (sum(vals) / len(vals)) if vals else None

        return {
            "num_samples": len(self.samples),
            "duration_s": round(self.end_time - self.start_time, 4),
            "max_used_bytes": _max("used_bytes"),
            "max_torch_allocated_bytes": _max("torch_allocated_bytes"),
            "max_torch_reserved_bytes": _max("torch_reserved_bytes"),
            "avg_util_percent": _avg("util_percent"),
        }


class GPUTelemetry:
    """Thin wrapper that combines torch and NVML telemetry.

    Designed to never fail the training loop — if NVML or CUDA is unavailable
    everything degrades to ``None`` fields.
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._torch = None
        self._pynvml = None
        self._handle = None
        try:
            import torch

            self._torch = torch if torch.cuda.is_available() else None
        except Exception:
            self._torch = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._pynvml = None
            self._handle = None

    def snapshot(self) -> TelemetrySnapshot:
        snap = TelemetrySnapshot(t=time.time())
        if self._pynvml is not None and self._handle is not None:
            try:
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                snap.used_bytes = int(mem.used)
                snap.free_bytes = int(mem.free)
                snap.total_bytes = int(mem.total)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                snap.util_percent = int(util.gpu)
            except Exception:
                pass
        if self._torch is not None:
            try:
                snap.torch_allocated_bytes = int(
                    self._torch.cuda.max_memory_allocated(self.device_index)
                )
                snap.torch_reserved_bytes = int(
                    self._torch.cuda.max_memory_reserved(self.device_index)
                )
            except Exception:
                pass
        return snap

    def reset_peak(self) -> None:
        if self._torch is not None:
            try:
                self._torch.cuda.reset_peak_memory_stats(self.device_index)
            except Exception:
                pass

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self._pynvml = None
        self._handle = None


@contextmanager
def telemetry_window(device_index: int = 0):
    tele = GPUTelemetry(device_index=device_index)
    tele.reset_peak()
    agg = TelemetryAggregate(start_time=time.time())
    try:
        yield tele, agg
    finally:
        agg.end_time = time.time()
        agg.append(tele.snapshot())
        tele.close()
