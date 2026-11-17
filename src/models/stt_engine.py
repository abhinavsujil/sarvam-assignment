"""
Offline Dual Speech-To-Text Engine
- English STT: Faster-Whisper Base with civic context prompt anchoring (cpu_threads=2, num_workers=1).
- Hindi STT: IIT Madras (AI4Bharat) IndicConformer INT8 ONNX (via sherpa-onnx).
- Single-slot active model memory management on CPU.
"""

try:
    import av
except ImportError:
    av = None
import gc
import io
import time
import numpy as np
from pathlib import Path
from typing import Optional, Union, BinaryIO, Tuple
import sherpa_onnx
from faster_whisper import WhisperModel


class STTEngine:
    """Offline Hybrid STT Engine: Whisper Base for English and IIT Madras AI4Bharat IndicConformer for Hindi."""

    CIVIC_EN_PROMPT = (
        "Aadhaar card, Akshaya e-Kendra, Kerala, Income certificate, "
        "Ration card, Death certificate, Caste certificate, Birth certificate, "
        "Driving license renewal, e-District portal."
    )

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        whisper_path: Optional[Union[str, Path]] = None,
        indic_hi_path: Optional[Union[str, Path]] = None
    ):
        root = Path(__file__).resolve().parent.parent.parent
        self.whisper_path = Path(whisper_path) if whisper_path else (
            root / "models" / "stt" / "models--Systran--faster-whisper-base" / "snapshots" / "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
        )
        self.indic_hi_model = Path(indic_hi_path) if indic_hi_path else (
            root / "models" / "indic_conformer" / "hi" / "model.int8.onnx"
        )
        self.indic_hi_tokens = self.indic_hi_model.parent / "tokens.txt"

        # Lazy loaded model handles
        self._whisper: Optional[WhisperModel] = None
        self._indic_recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None

    def _get_active_indic(self) -> sherpa_onnx.OfflineRecognizer:
        """Loads IndicConformer and releases Whisper C++ weights to stay within 850 MB RSS."""
        if self._whisper is not None:
            try:
                self._whisper.model.unload_model()
            except Exception:
                pass
        if self._indic_recognizer is None:
            if not self.indic_hi_model.exists():
                raise FileNotFoundError(f"IIT Madras Hindi model not found at: {self.indic_hi_model}")
            self._indic_recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(self.indic_hi_model),
                tokens=str(self.indic_hi_tokens),
                num_threads=2,
            )
        return self._indic_recognizer

    def _get_active_whisper(self) -> WhisperModel:
        """Loads Whisper and frees IndicConformer to stay within 850 MB RSS."""
        if self._indic_recognizer is not None:
            del self._indic_recognizer
            self._indic_recognizer = None
            gc.collect()
        if self._whisper is None:
            self._whisper = WhisperModel(
                str(self.whisper_path),
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                num_workers=1
            )
        else:
            try:
                if hasattr(self._whisper.model, "model_is_loaded") and not self._whisper.model.model_is_loaded:
                    self._whisper.model.load_model()
            except Exception:
                pass
        return self._whisper

    @property
    def whisper_model(self) -> WhisperModel:
        return self._get_active_whisper()

    @property
    def indic_recognizer(self) -> sherpa_onnx.OfflineRecognizer:
        return self._get_active_indic()

    def activate_language(self, lang: str):
        """Pre-activates the model for the requested language and releases the inactive one to strictly budget memory."""
        if lang and lang.lower().startswith("hi"):
            return self._get_active_indic()
        else:
            return self._get_active_whisper()

    @staticmethod
    def _decode_to_pcm16k(audio_input: Union[str, BinaryIO, bytes], target_sr: int = 16000) -> bytes:
        """Decodes WebM, WAV, OGG, or MP3 into 16kHz mono 16-bit PCM in memory."""
        if isinstance(audio_input, bytes):
            audio_input = io.BytesIO(audio_input)

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=target_sr)
        container = av.open(audio_input)
        raw_buffer = io.BytesIO()

        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                raw_buffer.write(resampled.to_ndarray().tobytes())

        return raw_buffer.getvalue()

    def transcribe(
        self,
        audio: Union[str, BinaryIO, bytes],
        language: Optional[str] = "en"
    ) -> Tuple[str, str, float]:
        """
        Transcribes audio completely offline into native script.
        Returns:
            Tuple[str, str, float]: (transcript_text, detected_language, latency_ms)
        """
        t_start = time.perf_counter_ns()

        pcm_16k = self._decode_to_pcm16k(audio, target_sr=16000)
        if not pcm_16k:
            return "", language or "en", 0.0

        is_hi = bool(language and language.lower().startswith("hi"))

        if is_hi:
            # IIT Madras (AI4Bharat) IndicConformer Recognition
            samples = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
            stream = self.indic_recognizer.create_stream()
            stream.accept_waveform(16000, samples)
            self.indic_recognizer.decode_stream(stream)
            transcript = stream.result.text.strip()
            resolved_lang = "hi"
        else:
            # Faster-Whisper Base with Civic Context Anchoring (Recognizes "Aadhaar", "Akshaya", etc.)
            audio_arr = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = self.whisper_model.transcribe(
                audio_arr,
                language="en",
                initial_prompt=self.CIVIC_EN_PROMPT,
                condition_on_previous_text=False,
                vad_filter=True,
                beam_size=1
            )
            transcript = " ".join(s.text for s in segments).strip()
            resolved_lang = "en"

        latency_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
        return transcript, resolved_lang, latency_ms
