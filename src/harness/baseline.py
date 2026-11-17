"""Sequential baseline harness for direct latency comparison against pipelined streaming."""

import time
from typing import Generator, Optional, Any, Callable, Dict

from src.harness.timers import LatencyTracker
from src.harness.memory import MemoryMonitor
from src.harness.normalizer import TextNormalizer
from src.harness.pipeline import GenerationChunk


class SequentialHarness:
    """Sequential (naive) execution harness.
    
    Generates all LLM tokens first, accumulates the entire response text,
    normalizes it in one piece, and then calls TTS synthesis on the entire text.
    Used as an empirical baseline to measure latency improvements from pipelining.
    """

    def __init__(self, llm_engine: Any = None, tts_engine: Any = None):
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine

    def run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
        lang: str = "auto",
        llm_generate_fn: Optional[Callable[..., Any]] = None,
        tts_synthesize_fn: Optional[Callable[[str], Any]] = None,
    ) -> Generator[GenerationChunk, None, None]:
        """Execute full generation followed sequentially by full TTS synthesis."""
        tracker = LatencyTracker()
        mem_monitor = MemoryMonitor(interval_s=0.05)
        mem_monitor.start()
        tracker.mark_start()

        gen_fn = llm_generate_fn or getattr(self.llm_engine, "generate_stream", None)
        raw_synth_fn = tts_synthesize_fn or getattr(self.tts_engine, "synthesize", None)
        synth_fn = (lambda text: raw_synth_fn(text, lang=lang)) if raw_synth_fn else None

        tokens = []
        token_count = 0
        char_count = 0

        # Phase 1: Wait for ALL tokens from LLM
        stream = gen_fn(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        for raw_token in stream:
            token_count += 1
            char_count += len(raw_token)
            tokens.append(raw_token)
            if token_count == 1:
                tracker.record_first_token()
                yield GenerationChunk(text=raw_token, is_first_token=True)
            else:
                yield GenerationChunk(text=raw_token)

        tracker.record_llm_end(token_count, char_count)
        full_text = "".join(tokens)

        # Phase 2: Sequential normalization and full TTS synthesis
        norm_text = TextNormalizer.normalize(full_text, lang=lang)

        t_synth_start = time.perf_counter_ns()
        pcm_bytes, sr, duration_s = synth_fn(norm_text)
        t_synth_end = time.perf_counter_ns()
        synth_s = (t_synth_end - t_synth_start) / 1_000_000_000.0

        # TTFA occurs only after all tokens are generated + synthesis starts
        tracker.record_first_audio()
        tracker.record_audio_chunk(duration_s, synth_s)

        yield GenerationChunk(
            audio_pcm=pcm_bytes,
            sample_rate=sr,
            sentence_idx=1,
            is_first_audio=True
        )

        tracker.mark_end()
        mem_stats = mem_monitor.stop()

        telemetry = tracker.to_dict()
        telemetry.update(mem_stats)
        telemetry["interrupted"] = False

        yield GenerationChunk(is_final=True, telemetry=telemetry)
