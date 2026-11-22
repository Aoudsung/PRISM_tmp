#!/usr/bin/env bash
set -euo pipefail

# The TextCraft supervisor calls this while holding the shared GPU 3+4 lane lock.
root=$1
predecessor="$root/study-textcraft-v4-joint-20261120/comparison/textcraft/fgpo/seed-0"
"$root/runtime/venv/bin/python" - "$predecessor" <<'PY'
import json
import sys
from pathlib import Path

cell = Path(sys.argv[1])
try:
    marker = json.loads((cell / ".finalization_complete.json").read_text())
    summary = json.loads((cell / "cell_summary.json").read_text())
except (OSError, ValueError) as exc:
    raise SystemExit(f"handoff blocked: predecessor terminal publication unavailable: {exc}")

training = summary.get("training", {})
validation = summary.get("evaluation", {}).get("val", {})
if not (
    marker.get("terminal_export_completed") is True
    and summary.get("status") == "complete"
    and summary.get("finalization_stage") == "finalized"
    and training.get("rounds_completed") == 50
    and validation.get("dataset_complete") is True
    and validation.get("evaluated_count") == validation.get("official_task_count") == 79
):
    raise SystemExit(f"handoff blocked: {cell} lacks 50 completed rounds and finalized val 79")

print("handoff ready: predecessor finalized after 50 rounds and official val 79")
PY
