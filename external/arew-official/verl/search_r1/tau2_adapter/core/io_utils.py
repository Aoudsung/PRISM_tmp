import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

try:
    import tomli_w
except ImportError:  # pragma: no cover - optional dependency
    tomli_w = None


def load_file(path: str | Path) -> Any:
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    if path.suffix == ".toml":
        with open(path, "rb") as fp:
            return tomllib.load(fp)
    if path.suffix in {".md", ".txt"}:
        with open(path, "r", encoding="utf-8") as fp:
            return fp.read()
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def dump_file(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        return
    if path.suffix == ".toml":
        if tomli_w is None:
            raise ValueError("Writing TOML requires tomli-w to be installed")
        with open(path, "wb") as fp:
            fp.write(tomli_w.dumps(data).encode("utf-8"))
        return
    if path.suffix in {".md", ".txt"}:
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(str(data))
        return
    raise ValueError(f"Unsupported file extension: {path.suffix}")
