"""
Neural TTS Engine wrapping Piper ONNX models for low-latency edge speech synthesis.
Supports English (Lessac) and Hindi (Rohan, trained on IndicTTS).
"""

import gc
import io
import json
import wave
from pathlib import Path
from typing import Tuple, Dict, Optional
import onnxruntime
import piper
from piper import PiperVoice, PiperConfig


class TTSEngine:
    """Piper ONNX speech synthesis engine with explicit language binding."""

    SUPPORTED_LANGUAGES = ("en", "hi")

    def __init__(
        self,
        en_model_path: Path,
        hi_model_path: Path,
        num_threads: int = 2
    ):
        self.model_paths: Dict[str, Path] = {
            "en": Path(en_model_path),
            "hi": Path(hi_model_path)
        }
        self.num_threads = num_threads
        self.voices: Dict[str, PiperVoice] = {}
        self.last_voice_swap_ms: float = 0.0

        # Validate model files exist on disk
        for lang, path in self.model_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing Piper ONNX model for [{lang}]: {path}")
            config_path = Path(f"{path}.json")
            if not config_path.exists():
                raise FileNotFoundError(f"Missing Piper ONNX config for [{lang}]: {config_path}")

    def preload_voice(self, lang: str) -> float:
        """Preload the requested voice prior to inference timing.
        
        Returns:
            float: Voice swap load time in milliseconds (0.0 if already loaded).
        """
        import time
        normalized_lang = "hi" if lang.lower().startswith("hi") else "en"
        if normalized_lang in self.voices:
            self.last_voice_swap_ms = 0.0
            return 0.0

        t0 = time.perf_counter_ns()
        self.get_voice(normalized_lang)
        t_elapsed = (time.perf_counter_ns() - t0) / 1_000_000.0
        self.last_voice_swap_ms = t_elapsed
        return t_elapsed

    def get_voice(self, lang: str) -> PiperVoice:
        """Retrieve or lazily load the exact PiperVoice for the requested language."""
        normalized_lang = "hi" if lang.lower().startswith("hi") else "en"

        # Release inactive voice to stay within < 850 MB RSS budget
        other_lang = "en" if normalized_lang == "hi" else "hi"
        if other_lang in self.voices:
            del self.voices[other_lang]
            gc.collect()

        if normalized_lang not in self.voices:
            model_path = self.model_paths[normalized_lang]
            config_path = f"{model_path}.json"

            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = self.num_threads
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_cpu_mem_arena = False

            session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )

            self.voices[normalized_lang] = PiperVoice(
                config=PiperConfig.from_dict(config_dict),
                session=session,
                espeak_data_dir=Path(piper.voice.ESPEAK_DATA_DIR),
                download_dir=Path.cwd()
            )

        return self.voices[normalized_lang]

    @staticmethod
    def is_hindi_text(text: str) -> bool:
        """Check if string contains Devanagari characters (Unicode range 0x0900 - 0x097F)."""
        return any('\u0900' <= c <= '\u097F' for c in text)

    def synthesize(self, text: str, lang: str = "auto") -> Tuple[bytes, int, float]:
        """Synthesize text into raw 16-bit mono PCM bytes.

        Args:
            text: Text to synthesize.
            lang: Language key ('en', 'hi', or 'auto').

        Returns:
            Tuple[bytes, int, float]: (pcm_bytes, sample_rate, duration_seconds)
        """
        clean_text = text.strip()
        if not clean_text:
            return b"", 22050, 0.0

        if lang == "auto":
            target_lang = "hi" if self.is_hindi_text(clean_text) else "en"
        else:
            target_lang = "hi" if lang.lower().startswith("hi") else "en"

        voice = self.get_voice(target_lang)

        # Synthesize into an in-memory WAV stream
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(clean_text, wav_file)

        buf.seek(0)
        with wave.open(buf, "rb") as wav_read:
            sample_rate = wav_read.getframerate()
            frames = wav_read.getnframes()
            pcm_bytes = wav_read.readframes(frames)
            duration_s = frames / float(sample_rate)

        return pcm_bytes, sample_rate, duration_s
