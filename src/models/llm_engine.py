"""LLM Engine wrapping llama-cpp-python for synchronous token streaming."""

from pathlib import Path
from typing import Generator, Optional, List
from llama_cpp import Llama


class LLMEngine:
    """Wrapper around Llama model providing stream generation."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 1024,
        n_batch: int = 256,
        n_threads: int = 4,
        verbose: bool = False
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"LLM model not found at: {self.model_path}")

        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            verbose=verbose
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
        repeat_penalty: float = 1.15,
        stop: Optional[List[str]] = None
    ) -> Generator[str, None, None]:
        """Synchronously stream tokens from llama_cpp."""
        if stop is None:
            stop = ["<|im_end|>", "<|endoftext|>"]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
            stream=True,
            stop=stop
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
