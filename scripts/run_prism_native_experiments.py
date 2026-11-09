#!/usr/bin/env python3
"""Remote-host entry point for independently recorded native-v1 experiments."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prism_evolve.native_runner import main

if __name__ == "__main__":
    main()
