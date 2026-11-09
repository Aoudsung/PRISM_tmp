import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def format_time(ts: datetime, use_compact_format: bool = False) -> str:
    if use_compact_format:
        return ts.strftime("%Y%m%d_%H%M%S")
    return ts.isoformat()


def get_now(use_compact_format: bool = False) -> str:
    return format_time(datetime.now(), use_compact_format=use_compact_format)


def get_dict_hash(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
