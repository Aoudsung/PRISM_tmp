"""Account for concurrent full checkpoint generations on the Host A results filesystem."""
import json
import os
from pathlib import Path
import sys


ATOMIC_GENERATIONS_PER_CELL = 2
GENERATION_BYTES = 125_000_000_000
RUN_OUTPUT_HEADROOM_BYTES = 20_000_000_000


def published_generation(cell):
    pointer = cell / "checkpoint" / "CURRENT"
    try:
        return pointer, json.loads(pointer.read_text())["snapshot"]
    except FileNotFoundError:
        return pointer, None


def allocated_inodes(directory):
    allocated = {}
    try:
        for path in (directory, *directory.rglob("*")):
            stat = path.lstat()
            allocated[(stat.st_dev, stat.st_ino)] = stat.st_blocks * 512
    except FileNotFoundError:
        # An active worker can publish its next snapshot and reclaim this one.
        return {}
    return allocated


def main(study, cells):
    published = []
    for cell in cells:
        pointer, snapshot = published_generation(cell)
        if snapshot is not None:
            generation = pointer.parent / "snapshots" / snapshot
            published.append((pointer, snapshot, allocated_inodes(generation)))
    filesystem = os.statvfs(study)
    allocated = {}
    for pointer, snapshot, generation_inodes in published:
        _, current = published_generation(pointer.parent.parent)
        if current == snapshot:
            allocated.update(generation_inodes)
    free = filesystem.f_bavail * filesystem.f_frsize
    stable_owned = sum(allocated.values())
    required = ATOMIC_GENERATIONS_PER_CELL * GENERATION_BYTES * len(cells) + RUN_OUTPUT_HEADROOM_BYTES
    print(f"disk free_bytes={free} stable_active_bytes={stable_owned} required_bytes={required} active_cells={len(cells)}")
    return 0 if free + stable_owned >= required else 75


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), [Path(path) for path in sys.argv[2:]]))
