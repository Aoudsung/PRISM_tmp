"""Launch the r3 Host A study with explicit output and physical GPU arguments.

All selectors and --resume are forwarded to the common runner. Historical v1
lane assignments and resource snapshots are deliberately not reused.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fgpo.runner import main


if __name__ == "__main__":
    raise SystemExit(main([
        "--config", str(ROOT / "configs/fgpo-r3-host-a.json"),
        *sys.argv[1:],
    ]))
