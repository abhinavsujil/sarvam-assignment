"""Process memory monitoring tracking RSS and USS via psutil."""

import os
import threading
import time
from typing import Tuple, Dict, Any, Optional
import psutil


def get_process_memory_mb() -> Tuple[float, float]:
    """Return (RSS_MB, USS_MB) for the current process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_full_info() if hasattr(process, 'memory_full_info') else process.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    uss_mb = getattr(mem_info, 'uss', mem_info.rss) / (1024 * 1024)
    return rss_mb, uss_mb


class MemoryMonitor:
    """Threaded memory sampler capturing per-turn sampled peak RSS, peak private bytes,
    and separate OS lifetime peak working set during execution."""

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.start_rss_mb: float = 0.0
        self.start_private_mb: float = 0.0
        self.sampled_peak_rss_mb: float = 0.0
        self.sampled_peak_private_mb: float = 0.0
        self.end_rss_mb: float = 0.0
        self.end_private_mb: float = 0.0
        self.lifetime_peak_wset_mb: float = 0.0

    @property
    def peak_rss_mb(self) -> float:
        """Backward-compatible alias for sampled_peak_rss_mb."""
        return self.sampled_peak_rss_mb

    @property
    def peak_wset_mb(self) -> float:
        """Backward-compatible alias for lifetime_peak_wset_mb."""
        return self.lifetime_peak_wset_mb

    def _sample_loop(self) -> None:
        while self._running:
            try:
                mem = self.process.memory_info()
                rss = mem.rss / (1024 * 1024)
                priv = getattr(mem, 'private', getattr(mem, 'pagefile', mem.rss)) / (1024 * 1024)
                if rss > self.sampled_peak_rss_mb:
                    self.sampled_peak_rss_mb = rss
                if priv > self.sampled_peak_private_mb:
                    self.sampled_peak_private_mb = priv
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval_s)

    def start(self) -> None:
        """Start memory sampling thread."""
        current_mem = self.process.memory_info()
        current_rss = current_mem.rss / (1024 * 1024)
        current_priv = getattr(current_mem, 'private', getattr(current_mem, 'pagefile', current_mem.rss)) / (1024 * 1024)
        self.start_rss_mb = current_rss
        self.start_private_mb = current_priv
        self.sampled_peak_rss_mb = current_rss
        self.sampled_peak_private_mb = current_priv
        self.end_rss_mb = current_rss
        self.end_private_mb = current_priv
        self.lifetime_peak_wset_mb = getattr(current_mem, 'peak_wset', current_mem.rss) / (1024 * 1024)
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True, name="MemoryMonitor")
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop sampling and record end/peak metrics with labeled keys."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        try:
            mem = self.process.memory_info()
            self.end_rss_mb = mem.rss / (1024 * 1024)
            self.end_private_mb = getattr(mem, 'private', getattr(mem, 'pagefile', mem.rss)) / (1024 * 1024)
            if self.end_rss_mb > self.sampled_peak_rss_mb:
                self.sampled_peak_rss_mb = self.end_rss_mb
            if self.end_private_mb > self.sampled_peak_private_mb:
                self.sampled_peak_private_mb = self.end_private_mb
            self.lifetime_peak_wset_mb = getattr(mem, 'peak_wset', mem.rss) / (1024 * 1024)
        except Exception:
            pass

        return {
            "start_rss_mb": round(self.start_rss_mb, 2),
            "sampled_peak_rss_mb": round(self.sampled_peak_rss_mb, 2),
            "sampled_peak_private_mb": round(self.sampled_peak_private_mb, 2),
            "lifetime_peak_wset_mb": round(self.lifetime_peak_wset_mb, 2),
            "end_rss_mb": round(self.end_rss_mb, 2),
            # Aliases for backward compatibility
            "peak_rss_mb": round(self.sampled_peak_rss_mb, 2),
            "peak_wset_mb": round(self.lifetime_peak_wset_mb, 2),
        }

    def __enter__(self) -> "MemoryMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
