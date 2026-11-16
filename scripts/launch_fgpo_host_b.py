"""Launch the r3 Host B study with explicit output and physical GPU arguments.

Starting fresh is the default. --resume continues only compatible r3 state;
there is no implicit migration or continuation of the former full/PPO lanes.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fgpo.runner import main


if __name__ == "__main__":
    raise SystemExit(main([
        "--config", str(ROOT / "configs/fgpo-r3-host-b.json"),
        *sys.argv[1:],
    ]))
