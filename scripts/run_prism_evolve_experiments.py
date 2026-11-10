#!/usr/bin/env python3
"""Default experiment entry point: Host B native-v1.

The retired controlled runner and its collected sources remain in
paper/acl2027/evidence/sources/. Old result directories are never rewritten.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prism_evolve.native_runner import main

if __name__ == "__main__":
    main()
