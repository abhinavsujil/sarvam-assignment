"""Comprehensive unit test suite for the streaming harness."""

import time
import pytest
from src.harness.timers import LatencyTracker
from src.harness.memory import MemoryMonitor, get_process_memory_mb
from src.harness.splitter import SentenceSplitter
from src.harness.normalizer import TextNormalizer, num_to_en, num_to_hi
from src.harness.pipeline import PipelinedHarness, GenerationChunk
from src.harness.baseline import SequentialHarness


# ---------------------------------------------------------
# 1. SPLITTER TESTS
# ---------------------------------------------------------
class TestSentenceSplitter:
    def test_english_sentence_splitting(self):
        splitter = SentenceSplitter()
        tokens = ["This", " is", " sentence", " one.", " And", " this", " is", " two!"]
        results = []
        for t in tokens:
            results.extend(splitter.feed(t))
        results.extend(splitter.flush())

        assert len(results) == 2
        assert results[0] == "This is sentence one."
        assert results[1] == "And this is two!"

    def test_hindi_danda_splitting(self):
        splitter = SentenceSplitter()
        tokens = ["यह", " पहला", " वाक्य", " है।", " यह", " दूसरा", " वाक्य", " है।"]
        results = []
        for t in tokens:
            results.extend(splitter.feed(t))
        results.extend(splitter.flush())

        assert len(results) == 2
        assert "पहला वाक्य है।" in results[0]
        assert "दूसरा वाक्य है।" in results[1]

    def test_abbreviations_do_not_split(self):
        splitter = SentenceSplitter()
        tokens = ["Dr.", " Sharma", " charged", " Rs.", " 500", " for", " the", " test."]
        results = []
        for t in tokens:
            results.extend(splitter.feed(t))
        results.extend(splitter.flush())

        # Should not split at Dr. or Rs.
        assert len(results) == 1
        assert results[0] == "Dr. Sharma charged Rs. 500 for the test."

    def test_decimals_do_not_split(self):
        splitter = SentenceSplitter()
        tokens = ["The", " version", " is", " 3.14", " right", " now."]
        results = []
        for t in tokens:
            results.extend(splitter.feed(t))
        results.extend(splitter.flush())

        assert len(results) == 1
        assert results[0] == "The version is 3.14 right now."

    def test_flush_handles_unterminated_trailing_text(self):
        splitter = SentenceSplitter()
        splitter.feed("Here is a sentence without punctuation")
        flushed = splitter.flush()
        assert len(flushed) == 1
        assert flushed[0] == "Here is a sentence without punctuation"

    def test_abbreviation_with_real_boundary_emits_before_flush(self):
        splitter = SentenceSplitter()
        tokens = ["The", " fee", " is", " Rs.", " 50", " for", " this", " service.", " Bring", " your", " card."]
        emitted_before_flush = []
        for t in tokens:
            emitted_before_flush.extend(splitter.feed(t))
        assert len(emitted_before_flush) >= 1, "Sentence 1 must be emitted during feed() before flush()"
        assert emitted_before_flush[0] == "The fee is Rs. 50 for this service."

    def test_doctor_abbreviation_with_boundary_emits_before_flush(self):
        splitter = SentenceSplitter()
        tokens = ["Contact", " Dr.", " Rao.", " He", " helps."]
        emitted_before_flush = []
        for t in tokens:
            emitted_before_flush.extend(splitter.feed(t))
        assert len(emitted_before_flush) >= 1, "Sentence 1 must be emitted during feed() before flush()"
        assert emitted_before_flush[0] == "Contact Dr. Rao."


# ---------------------------------------------------------
# 2. NORMALIZER TESTS
# ---------------------------------------------------------
class TestTextNormalizer:
    def test_markdown_cleaning(self):
        raw = "**Income Certificate** is required for *scholarships*. See [here](http://example.com)."
        cleaned = TextNormalizer.clean_markdown(raw)
        assert "**" not in cleaned
        assert "*" not in cleaned
        assert "http" not in cleaned
        assert "Income Certificate is required for scholarships. See here." == cleaned

    def test_english_currency_normalization(self):
        text = "The service fee is ₹500 and the penalty is Rs. 1000."
        norm = TextNormalizer.normalize(text, lang="en")
        assert "five hundred rupees" in norm
        assert "one thousand rupees" in norm
        assert "₹" not in norm

    def test_hindi_currency_normalization(self):
        text = "आय प्रमाण पत्र का शुल्क ₹500 है।"
        norm = TextNormalizer.normalize(text, lang="hi")
        assert "पाँच सौ रुपये" in norm
        assert "₹" not in norm

    def test_auto_detection_hindi(self):
        text = "आवेदन शुल्क ₹1000 है।"
        norm = TextNormalizer.normalize(text, lang="auto")
        assert "एक हज़ार रुपये" in norm

    def test_num_to_en_and_hi(self):
        assert num_to_en(50) == "fifty"
        assert num_to_en(250) == "two hundred fifty"
        assert num_to_hi(50) == "पचास"
        assert num_to_hi(500) == "पाँच सौ"

    def test_hindi_numbers_full_table(self):
        """Verify full 0-99 Hindi table for complex numbers."""
        test_samples = {
            17: "सत्रह",
            28: "अट्ठाईस",
            43: "तैंतालीस",
            67: "सरसठ",
            73: "तिहत्तर",
            89: "नवासी",
            99: "निन्यानवे"
        }
        for num, expected in test_samples.items():
            assert num_to_hi(num) == expected, f"Failed for {num}: expected {expected}, got {num_to_hi(num)}"

    def test_decimal_currency_normalization(self):
        """Verify rupees and paise conversion for decimal currency."""
        # English
        en_text = "The fee is ₹50.50 for this service."
        norm_en = TextNormalizer.normalize(en_text, lang="en")
        assert norm_en == "The fee is fifty rupees and fifty paise for this service."

        # Hindi
        hi_text = "आवेदन शुल्क ₹50.50 है।"
        norm_hi = TextNormalizer.normalize(hi_text, lang="hi")
        assert norm_hi == "आवेदन शुल्क पचास रुपये पचास पैसे है।"

    def test_plain_numbers_and_decimals(self):
        """Verify plain numbers and decimals in both languages."""
        # English plain numbers and decimals
        en_text = "Delivery takes 15 days with 2.5 hours notice."
        norm_en = TextNormalizer.normalize(en_text, lang="en")
        assert "fifteen days" in norm_en
        assert "two point five hours" in norm_en

        # Hindi plain numbers and decimals
        hi_text = "यह कार्य 15 दिनों में 2.5 प्रतिशत होगा।"
        norm_hi = TextNormalizer.normalize(hi_text, lang="hi")
        assert "पंद्रह दिनों" in norm_hi
        assert "दो दशमलव पाँच प्रतिशत" in norm_hi


# ---------------------------------------------------------
# 3. TIMERS & MEMORY TESTS
# ---------------------------------------------------------
class TestTimersAndMemory:
    def test_latency_tracker_calculations(self):
        tracker = LatencyTracker()
        tracker.mark_start()
        time.sleep(0.01)  # 10ms
        tracker.record_first_token()
        time.sleep(0.01)  # 10ms
        tracker.record_first_audio()
        tracker.record_audio_chunk(audio_duration_s=2.5, synthesis_time_s=0.25)
        tracker.record_llm_end(tokens_count=10)
        tracker.mark_end()

        assert tracker.ttft_ms is not None
        assert tracker.ttfa_ms is not None
        assert tracker.ttft_ms >= 8.0  # At least ~10ms
        assert tracker.ttfa_ms >= 16.0  # At least ~20ms
        assert tracker.ttfa_ms > tracker.ttft_ms
        assert tracker.rtf == pytest.approx(0.1, rel=1e-2)  # 0.25 / 2.5 = 0.1

    def test_memory_monitor(self):
        rss, uss = get_process_memory_mb()
        assert rss > 10.0  # Python process should have > 10MB
        with MemoryMonitor(interval_s=0.01) as monitor:
            time.sleep(0.03)
        assert monitor.sampled_peak_rss_mb >= monitor.start_rss_mb
        assert monitor.sampled_peak_private_mb >= monitor.start_private_mb
        assert monitor.lifetime_peak_wset_mb >= monitor.sampled_peak_rss_mb


# ---------------------------------------------------------
# 4. PIPELINE VS BASELINE TESTS (MOCK ENGINES)
# ---------------------------------------------------------
class TestHarnessExecution:
    @staticmethod
    def mock_llm_stream(prompt, system_prompt=None, max_tokens=256, temperature=0.2):
        """Simulates 2 sentences generated token-by-token with 25ms delay per token."""
        sentence_1 = ["Here", " is", " the", " first", " sentence."]
        sentence_2 = [" And", " here", " is", " the", " second", " sentence."]
        for tok in sentence_1 + sentence_2:
            time.sleep(0.025)  # 25ms per token
            yield tok

    @staticmethod
    def mock_tts_synthesize(text, lang="en", **kwargs):
        """Simulates TTS synthesis taking 40ms to create 1.5s audio."""
        time.sleep(0.040)
        sample_rate = 22050
        fake_pcm = b"\x00" * int(sample_rate * 1.5 * 2)  # 1.5s 16-bit PCM
        return fake_pcm, sample_rate, 1.5

    def test_pipelined_ttfa_is_faster_than_sequential(self):
        # 1. Run Pipelined Harness
        pipelined = PipelinedHarness()
        pipe_chunks = list(pipelined.run_stream(
            prompt="Tell me about income certificate",
            llm_generate_fn=self.mock_llm_stream,
            tts_synthesize_fn=self.mock_tts_synthesize
        ))
        pipe_final = pipe_chunks[-1]
        assert pipe_final.is_final is True
        pipe_telemetry = pipe_final.telemetry
        pipe_ttfa = pipe_telemetry["ttfa_ms"]

        # 2. Run Sequential Harness
        sequential = SequentialHarness()
        seq_chunks = list(sequential.run(
            prompt="Tell me about income certificate",
            llm_generate_fn=self.mock_llm_stream,
            tts_synthesize_fn=self.mock_tts_synthesize
        ))
        seq_final = seq_chunks[-1]
        assert seq_final.is_final is True
        seq_telemetry = seq_final.telemetry
        seq_ttfa = seq_telemetry["ttfa_ms"]

        # In sequential, TTFA must wait for ALL 11 tokens (11 * 25ms = 275ms) + TTS (40ms) = ~315ms+
        # In pipelined, TTFA only waits for Sentence 1 (5 tokens * 25ms = 125ms) + TTS (40ms) = ~165ms
        print(f"\n[BENCHMARK PROOF] Pipelined TTFA: {pipe_ttfa:.2f}ms | Sequential TTFA: {seq_ttfa:.2f}ms")
        assert pipe_ttfa < seq_ttfa, f"Expected pipelined TTFA ({pipe_ttfa}ms) < sequential TTFA ({seq_ttfa}ms)"
        # Pipelined should be at least 60ms faster in this controlled mock
        assert (seq_ttfa - pipe_ttfa) > 50.0

    def test_pipelined_barge_in_cancellation(self):
        pipelined = PipelinedHarness()
        
        chunks = []
        gen = pipelined.run_stream(
            prompt="Tell me about income certificate",
            llm_generate_fn=self.mock_llm_stream,
            tts_synthesize_fn=self.mock_tts_synthesize
        )
        
        # Pull 3 chunks, then trigger interruption
        for i, chunk in enumerate(gen):
            chunks.append(chunk)
            if i == 2:
                pipelined.interrupt()
        
        final_chunk = chunks[-1]
        assert final_chunk.is_final is True
        assert final_chunk.telemetry["interrupted"] is True

    def test_milestone_timing_order(self):
        """Milestones must satisfy: t_start < t_first_token < t_first_audio < t_end."""
        pipelined = PipelinedHarness()
        chunks = list(pipelined.run_stream(
            prompt="Test milestone order",
            llm_generate_fn=self.mock_llm_stream,
            tts_synthesize_fn=self.mock_tts_synthesize
        ))
        final_chunk = chunks[-1]
        assert final_chunk.is_final is True
        telem = final_chunk.telemetry

        ttft = telem["ttft_ms"]
        ttfa = telem["ttfa_ms"]
        total = telem["total_latency_ms"]

        assert ttft is not None and ttft > 0, f"Expected ttft > 0, got {ttft}"
        assert ttfa is not None and ttfa > 0, f"Expected ttfa > 0, got {ttfa}"
        assert total > 0, f"Expected total > 0, got {total}"
        # Milestone ordering: TTFT must happen before TTFA, and TTFA before total end
        assert ttft < ttfa, f"Expected TTFT ({ttft}ms) < TTFA ({ttfa}ms)"
        assert ttfa <= total, f"Expected TTFA ({ttfa}ms) <= total latency ({total}ms)"

    def test_tts_worker_exception_propagates_without_hang(self):
        """Exceptions in the background TTS worker must propagate to caller without deadlock."""
        def crashing_tts(text, lang="en", **kwargs):
            time.sleep(0.01)
            raise RuntimeError("TTS ONNX engine crashed")

        pipelined = PipelinedHarness()
        gen = pipelined.run_stream(
            prompt="Test error propagation",
            llm_generate_fn=self.mock_llm_stream,
            tts_synthesize_fn=crashing_tts
        )

        with pytest.raises(RuntimeError) as exc_info:
            for _ in gen:
                pass
        assert "TTS ONNX engine crashed" in str(exc_info.value)

    def test_cancel_stops_further_synthesis(self):
        """Cancelling / barge-in must immediately stop further TTS calls."""
        tts_calls = []

        def counting_tts(text, lang="en", **kwargs):
            tts_calls.append(text)
            time.sleep(0.050)
            return b"\x00" * 4410, 22050, 0.1

        def ten_sentence_llm(prompt, **kwargs):
            for i in range(1, 11):
                time.sleep(0.010)
                yield f" Sentence {i} is being generated."

        pipelined = PipelinedHarness()
        gen = pipelined.run_stream(
            prompt="Test multi sentence cancel",
            llm_generate_fn=ten_sentence_llm,
            tts_synthesize_fn=counting_tts
        )

        audio_received = 0
        for chunk in gen:
            if chunk.audio_pcm is not None:
                audio_received += 1
                # Interrupt immediately after the first audio chunk is delivered
                pipelined.interrupt()

        # Without interruption, all 10 sentences would be synthesized
        # With interruption, synthesis stops immediately (calls <= 2)
        assert audio_received >= 1
        assert len(tts_calls) < 10, f"Expected < 10 calls, but got {len(tts_calls)}"
        assert len(tts_calls) <= 3, f"Expected at most 3 TTS calls before cancellation halted worker, got {len(tts_calls)}"

    def test_voice_preload_and_swap_reporting(self):
        """Preload active voice before timing starts; report swap time separately."""
        class MockTTSWithPreload:
            def __init__(self):
                self.active_voice = None
                self.swap_count = 0

            def preload_voice(self, lang):
                if self.active_voice != lang:
                    self.active_voice = lang
                    self.swap_count += 1
                    return 45.0
                return 0.0

            def synthesize(self, text, lang="en", **kwargs):
                return b"\x00" * 4410, 22050, 0.1

        mock_tts = MockTTSWithPreload()
        pipelined = PipelinedHarness(tts_engine=mock_tts)

        # Turn 1: Voice swap needed (en)
        chunks1 = list(pipelined.run_stream(
            prompt="Hello",
            lang="en",
            llm_generate_fn=self.mock_llm_stream
        ))
        telem1 = chunks1[-1].telemetry
        assert telem1["voice_swap_ms"] == 45.0
        assert mock_tts.swap_count == 1

        # Turn 2: Same voice (en) -> 0.0ms swap time
        chunks2 = list(pipelined.run_stream(
            prompt="Hello again",
            lang="en",
            llm_generate_fn=self.mock_llm_stream
        ))
        telem2 = chunks2[-1].telemetry
        assert telem2["voice_swap_ms"] == 0.0
        assert mock_tts.swap_count == 1


# ---------------------------------------------------------
# 5. STATISTICAL PERCENTILE TESTS
# ---------------------------------------------------------
class TestPercentileMaths:
    def test_empty_and_single_element(self):
        from src.harness.timers import calculate_percentile
        assert calculate_percentile([], 50) == 0.0
        assert calculate_percentile([42.5], 50) == 42.5
        assert calculate_percentile([42.5], 90) == 42.5

    def test_known_distribution_and_numpy_parity(self):
        import numpy as np
        from src.harness.timers import calculate_percentile

        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        # P50 = 30.0, P90 = 46.0
        assert calculate_percentile(data, 50) == 30.0
        assert calculate_percentile(data, 90) == 46.0

        # Parity against numpy linear interpolation for arbitrary series
        dataset = [14.2, 5.8, 92.1, 45.3, 110.0, 33.7, 88.4, 12.1]
        for p in [10, 25, 50, 75, 90, 95, 99]:
            expected = float(np.percentile(dataset, p))
            actual = calculate_percentile(dataset, p)
            assert abs(actual - expected) < 1e-4, f"Mismatch at P{p}: {actual} != {expected}"

