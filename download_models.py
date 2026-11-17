#!/usr/bin/env python3
"""
Single-command model downloader for Akshaya Civic Voice Desk.
Cross-platform: Runs on macOS (Apple Silicon / Intel), Linux, and Windows.
Downloads only the verified production weights into the models/ folder.
"""
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

MODELS = [
    {
        "name": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf (LLM)",
        "dest": MODELS_DIR / "llm" / "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    },
    {
        "name": "en_US-lessac-medium.onnx (English Piper TTS)",
        "dest": MODELS_DIR / "piper" / "en" / "en_US-lessac-medium.onnx",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    },
    {
        "name": "en_US-lessac-medium.onnx.json (Config)",
        "dest": MODELS_DIR / "piper" / "en" / "en_US-lessac-medium.onnx.json",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    },
    {
        "name": "hi_IN-rohan-medium.onnx (Hindi Piper TTS)",
        "dest": MODELS_DIR / "piper" / "hi" / "hi_IN-rohan-medium.onnx",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx"
    },
    {
        "name": "hi_IN-rohan-medium.onnx.json (Config)",
        "dest": MODELS_DIR / "piper" / "hi" / "hi_IN-rohan-medium.onnx.json",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json"
    }
]

def download(url: str, dest: Path, name: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"  [✓] {name} already exists ({dest.stat().st_size / (1024*1024):.1f} MB)")
        return
    print(f"  [↓] Downloading {name}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := resp.read(1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                print(f"\r      Progress: {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)
        print()
    print(f"  [✓] Finished {name}")

def main():
    print("=" * 65)
    print("  AKSHAYA CIVIC VOICE DESK - OFFLINE MODEL ASSET DOWNLOADER")
    print("=" * 65)
    for m in MODELS:
        download(m["url"], m["dest"], m["name"])
    print("=" * 65)
    print("  All production models ready in models/ folder!")
    print("  Run 'python voice_app.py' to launch the voice studio.")
    print("=" * 65)

if __name__ == "__main__":
    main()
