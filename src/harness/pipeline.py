"""Pipelined streaming harness with producer-consumer multithreading and atomic cancellation."""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Generator, Optional, Any, Callable, Dict

from src.harness.timers import LatencyTracker
from src.harness.memory import MemoryMonitor, get_process_memory_mb
from src.harness.splitter import SentenceSplitter
from src.harness.normalizer import TextNormalizer


@dataclass
class GenerationChunk:
    """A streaming output chunk containing text, audio, or telemetry."""
    text: Optional[str] = None
    audio_pcm: Optional[bytes] = None
    sample_rate: int = 22050
    sentence_idx: int = 0
    is_first_token: bool = False
    is_first_audio: bool = False
    is_final: bool = False
    telemetry: Optional[Dict[str, Any]] = None


class PipelinedHarness:
    """Orchestrates token generation, sentence boundary detection, and background TTS.
    
    Architecture:
    - Producer Thread: Consumes synchronous llama_cpp generator, feeds SentenceSplitter.
    - Thread-Safe Queue: Transfers normalized complete sentences.
    - Consumer Thread: Synthesizes audio using Piper ONNX in parallel.
    - Interruption Event: Halts producer and clears queue atomically on barge-in.
    """

    def __init__(self, llm_engine: Any = None, tts_engine: Any = None):
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine
        self._current_cancel_event: Optional[threading.Event] = None
        self._lock = threading.Lock()

    @property
    def interruption_event(self) -> threading.Event:
        """Backward-compatible access to the active turn cancellation event."""
        with self._lock:
            if self._current_cancel_event is None:
                self._current_cancel_event = threading.Event()
            return self._current_cancel_event

    def interrupt(self) -> None:
        """Atomic cancellation signal for barge-in on the active turn."""
        with self._lock:
            if self._current_cancel_event is not None:
                self._current_cancel_event.set()

    def run_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
        lang: str = "auto",
        llm_generate_fn: Optional[Callable[..., Any]] = None,
        tts_synthesize_fn: Optional[Callable[[str], Any]] = None,
    ) -> Generator[GenerationChunk, None, None]:
        """Stream tokens and audio chunks concurrently via producer-consumer threads."""
        # Use a fresh per-turn cancellation event; never clear flags lost-interrupt style
        turn_cancel_event = threading.Event()
        with self._lock:
            self._current_cancel_event = turn_cancel_event
        
        # Preload active voice before timing starts, reporting swap time separately
        voice_swap_ms = 0.0
        if self.tts_engine and hasattr(self.tts_engine, "preload_voice"):
            voice_swap_ms = self.tts_engine.preload_voice(lang)

        tracker = LatencyTracker()
        tracker.mark_start()
        tracker.voice_swap_ms = voice_swap_ms
        mem_monitor = MemoryMonitor(interval_s=0.05)
        mem_monitor.start()

        # Thread-safe queue for normalized sentences: (sentence_idx, sentence_text)
        # None marks end of stream
        sentence_queue: queue.Queue = queue.Queue(maxsize=32)
        # Output event queue for the caller generator: GenerationChunk
        out_queue: queue.Queue = queue.Queue()

        # Resolve generators/functions
        gen_fn = llm_generate_fn or getattr(self.llm_engine, "generate_stream", None)
        raw_synth_fn = tts_synthesize_fn or getattr(self.tts_engine, "synthesize", None)
        synth_fn = (lambda text: raw_synth_fn(text, lang=lang)) if raw_synth_fn else None

        splitter = SentenceSplitter()
        producer_error = []
        consumer_error = []

        # PRODUCER THREAD: LLM Generation -> Splitter -> Normalizer -> sentence_queue
        def producer_worker():
            stream = None
            try:
                token_count = 0
                char_count = 0
                sent_idx = 0

                # Consume synchronous token stream
                stream = gen_fn(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                for raw_token in stream:
                    if turn_cancel_event.is_set():
                        break

                    token_count += 1
                    char_count += len(raw_token)

                    # First token telemetry
                    if token_count == 1:
                        tracker.record_first_token()
                        out_queue.put(GenerationChunk(
                            text=raw_token,
                            is_first_token=True
                        ))
                    else:
                        out_queue.put(GenerationChunk(text=raw_token))

                    # Feed sentence splitter
                    complete_sentences = splitter.feed(raw_token)
                    for sent in complete_sentences:
                        if turn_cancel_event.is_set():
                            break
                        norm_sent = TextNormalizer.normalize(sent, lang=lang)
                        sent_idx += 1
                        sentence_queue.put((sent_idx, norm_sent))

                # Flush remaining text in splitter
                if not turn_cancel_event.is_set():
                    remaining = splitter.flush()
                    for sent in remaining:
                        norm_sent = TextNormalizer.normalize(sent, lang=lang)
                        sent_idx += 1
                        sentence_queue.put((sent_idx, norm_sent))

                tracker.record_llm_end(token_count, char_count)

            except Exception as e:
                producer_error.append(e)
                turn_cancel_event.set()
            finally:
                # Explicitly close generator to halt background LLM work immediately
                if stream is not None and hasattr(stream, "close"):
                    try:
                        stream.close()
                    except Exception:
                        pass
                # Signal consumer that producer is finished
                sentence_queue.put(None)

        # CONSUMER THREAD: sentence_queue -> Piper TTS -> out_queue
        def consumer_worker():
            try:
                first_audio_emitted = False
                sample_rate = 22050

                while True:
                    if turn_cancel_event.is_set():
                        # Purge remaining items in queue
                        while not sentence_queue.empty():
                            try:
                                sentence_queue.get_nowait()
                                sentence_queue.task_done()
                            except queue.Empty:
                                break
                        break

                    item = sentence_queue.get()
                    if item is None:
                        sentence_queue.task_done()
                        break

                    s_idx, s_text = item
                    if turn_cancel_event.is_set() or not s_text.strip():
                        sentence_queue.task_done()
                        continue

                    t_synth_start = time.perf_counter_ns()
                    # Execute TTS synthesis
                    pcm_bytes, sr, duration_s = synth_fn(s_text)
                    t_synth_end = time.perf_counter_ns()
                    synth_s = (t_synth_end - t_synth_start) / 1_000_000_000.0
                    sample_rate = sr

                    tracker.record_audio_chunk(duration_s, synth_s)

                    is_first_a = False
                    if not first_audio_emitted:
                        first_audio_emitted = True
                        is_first_a = True
                        tracker.record_first_audio()

                    out_queue.put(GenerationChunk(
                        audio_pcm=pcm_bytes,
                        sample_rate=sr,
                        sentence_idx=s_idx,
                        is_first_audio=is_first_a
                    ))
                    sentence_queue.task_done()

            except Exception as e:
                consumer_error.append(e)
                turn_cancel_event.set()
            finally:
                out_queue.put(None)  # Signal caller that consumer is finished

        producer_thread = threading.Thread(target=producer_worker, daemon=True, name="LLMProducer")
        consumer_thread = threading.Thread(target=consumer_worker, daemon=True, name="TTSConsumer")

        producer_thread.start()
        consumer_thread.start()

        # Yield items as they arrive to caller generator
        finished_workers = 0
        while finished_workers < 1:
            chunk = out_queue.get()
            if chunk is None:
                finished_workers += 1
                out_queue.task_done()
                continue
            yield chunk
            out_queue.task_done()

        producer_thread.join(timeout=1.0)
        consumer_thread.join(timeout=1.0)

        tracker.mark_end()
        mem_stats = mem_monitor.stop()

        if producer_error:
            raise producer_error[0]
        if consumer_error:
            raise consumer_error[0]

        # Final chunk with telemetry
        telemetry = tracker.to_dict()
        telemetry.update(mem_stats)
        telemetry["interrupted"] = turn_cancel_event.is_set()

        yield GenerationChunk(is_final=True, telemetry=telemetry)
