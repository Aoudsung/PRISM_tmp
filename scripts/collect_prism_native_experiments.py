#!/usr/bin/env python3
"""Server-side final collection of the two transferred native experiment shards."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prism_evolve.native_collection import main

if __name__ == "__main__":
    main()
