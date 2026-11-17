"""
Root Benchmark CLI for Sarvam AI Edge Voice Desk Take-Home.
Runs automated empirical benchmark evaluation across English and Hindi queries,
calculates P50 and P90 metrics, verifies memory bounds, and updates benchmark_results.json.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.run_benchmarks import run_benchmark

if __name__ == "__main__":
    run_benchmark()
