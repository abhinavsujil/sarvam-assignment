"""Hardware-precision microsecond latency tracking for LLM and TTS pipeline."""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class LatencyTracker:
    """Microsecond latency tracker utilizing time.perf_counter_ns()."""
    t_start_ns: int = field(default_factory=time.perf_counter_ns)
    t_first_token_ns: Optional[int] = None
    t_first_audio_ns: Optional[int] = None
    t_llm_end_ns: Optional[int] = None
    t_end_ns: Optional[int] = None

    total_tokens: int = 0
    total_characters: int = 0
    audio_duration_s: float = 0.0
    synthesis_duration_s: float = 0.0
    voice_swap_ms: float = 0.0

    def mark_start(self) -> None:
        """Reset and start request timer."""
        self.t_start_ns = time.perf_counter_ns()
        self.t_first_token_ns = None
        self.t_first_audio_ns = None
        self.t_llm_end_ns = None
        self.t_end_ns = None
        self.total_tokens = 0
        self.total_characters = 0
        self.audio_duration_s = 0.0
        self.synthesis_duration_s = 0.0

    def record_first_token(self) -> None:
        """Record timestamp of first token emitted by LLM if not already set."""
        if self.t_first_token_ns is None:
            self.t_first_token_ns = time.perf_counter_ns()

    def record_first_audio(self) -> None:
        """Record timestamp of first audio chunk emitted to consumer if not already set."""
        if self.t_first_audio_ns is None:
            self.t_first_audio_ns = time.perf_counter_ns()

    def record_llm_end(self, tokens_count: int, char_count: int = 0) -> None:
        """Record completion of LLM generation."""
        self.t_llm_end_ns = time.perf_counter_ns()
        self.total_tokens = tokens_count
        self.total_characters = char_count

    def record_audio_chunk(self, audio_duration_s: float, synthesis_time_s: float) -> None:
        """Accumulate audio duration and synthesis computation time."""
        self.audio_duration_s += audio_duration_s
        self.synthesis_duration_s += synthesis_time_s

    def mark_end(self) -> None:
        """Record completion of turn."""
        self.t_end_ns = time.perf_counter_ns()

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to First Token (TTFT) in milliseconds."""
        if self.t_first_token_ns is None:
            return None
        return (self.t_first_token_ns - self.t_start_ns) / 1_000_000.0

    @property
    def ttfa_ms(self) -> Optional[float]:
        """Time to First Audio (TTFA) in milliseconds."""
        if self.t_first_audio_ns is None:
            return None
        return (self.t_first_audio_ns - self.t_start_ns) / 1_000_000.0

    @property
    def total_latency_ms(self) -> float:
        """Total turn latency in milliseconds."""
        end_ns = self.t_end_ns if self.t_end_ns is not None else time.perf_counter_ns()
        return (end_ns - self.t_start_ns) / 1_000_000.0

    @property
    def tokens_per_second(self) -> float:
        """Generation speed (tok/s)."""
        if self.t_first_token_ns is None or self.t_llm_end_ns is None:
            return 0.0
        delta_s = (self.t_llm_end_ns - self.t_first_token_ns) / 1_000_000_000.0
        if delta_s <= 0 or self.total_tokens <= 1:
            return 0.0
        # Tokens generated after the first token
        return (self.total_tokens - 1) / delta_s

    @property
    def rtf(self) -> float:
        """Real-Time Factor: synthesis compute time / audio duration (lower is faster)."""
        if self.audio_duration_s <= 0.0:
            return 0.0
        return self.synthesis_duration_s / self.audio_duration_s

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics to dictionary."""
        return {
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "ttfa_ms": round(self.ttfa_ms, 2) if self.ttfa_ms is not None else None,
            "voice_swap_ms": round(self.voice_swap_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "audio_duration_s": round(self.audio_duration_s, 3),
            "synthesis_duration_s": round(self.synthesis_duration_s, 3),
            "rtf": round(self.rtf, 4),
        }


def calculate_percentile(data: list, p: float) -> float:
    """Calculate the p-th percentile (0 <= p <= 100) using linear interpolation.
    
    Matches standard numpy.percentile default ('linear') behavior.
    """
    if not data:
        return 0.0
    sorted_d = sorted(data)
    n = len(sorted_d)
    if n == 1:
        return float(sorted_d[0])
    idx = (n - 1) * (p / 100.0)
    floor_idx = int(idx)
    ceil_idx = floor_idx + 1
    if ceil_idx >= n:
        return float(sorted_d[-1])
    fraction = idx - floor_idx
    return float(sorted_d[floor_idx] + fraction * (sorted_d[ceil_idx] - sorted_d[floor_idx]))

