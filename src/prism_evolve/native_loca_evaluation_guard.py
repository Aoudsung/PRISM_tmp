"""Stable, confined filesystem boundary for trusted LOCA artifact consumers.

MCP process trees are paused after the native call has returned, while the
official adapter inspects files. The terminal evaluator receives a materialized
regular-file tree; existing tool bind mounts keep referring to the old inodes.
This is an evaluation guard, not an environment snapshot or replay backend.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import stat
import tempfile
import time


class UnsafeLOCAOutput(ValueError):
    """An actor-created filesystem entry escapes the public workspace."""


def _children(pid: int) -> set[int]:
    children: set[int] = set()
    try:
        tasks = list((Path("/proc") / str(pid) / "task").iterdir())
    except FileNotFoundError:
        return children
    for task in tasks:
        try:
            children.update(int(value) for value in (task / "children").read_text().split())
        except FileNotFoundError:
            continue
    return children


def _process_state(pid: int) -> str | None:
    try:
        lines = (Path("/proc") / str(pid) / "status").read_text().splitlines()
    except FileNotFoundError:
        return None
    return next((line.split()[1] for line in lines if line.startswith("State:")), None)


class PausedMCPProcesses:
    """Pause every descendant of the dedicated native environment worker.

Stopping parents before traversing their children prevents further forks along
already traversed edges. PID file descriptors avoid signalling reused PIDs.
Already stopped processes retain their prior stopped state on exit.
"""

    def __init__(self) -> None:
        self._handles: dict[int, tuple[int, bool]] = {}

    def freeze(self) -> None:
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("LOCA trusted file inspection requires Linux pidfd signalling.")
        deadline = time.monotonic() + 10.0
        try:
            pending = list(_children(os.getpid()))
            while pending:
                pid = pending.pop()
                if pid in self._handles:
                    continue
                try:
                    fd = os.pidfd_open(pid)
                except ProcessLookupError:
                    continue
                before = _process_state(pid)
                if before in (None, "Z", "X"):
                    os.close(fd)
                    continue
                resume = before not in {"T", "t"}
                self._handles[pid] = (fd, resume)
                try:
                    signal.pidfd_send_signal(fd, signal.SIGSTOP)
                except ProcessLookupError:
                    continue
                while _process_state(pid) not in {None, "T", "t", "Z", "X"}:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("MCP descendants did not quiesce before trusted file inspection.")
                    time.sleep(0.001)
                pending.extend(_children(pid))
            # A dead descendant can be reparented to its namespace init while
            # the tree is being stopped. Revisit each stopped parent's children.
            missing = _children(os.getpid()) - self._handles.keys()
            for pid in tuple(self._handles):
                missing.update(_children(pid) - self._handles.keys())
            if missing:
                raise RuntimeError("MCP descendant topology changed while establishing the inspection boundary.")
        except BaseException:
            self.resume()
            raise

    def resume(self) -> None:
        for fd, should_resume in reversed(tuple(self._handles.values())):
            try:
                if should_resume:
                    signal.pidfd_send_signal(fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)
        self._handles.clear()


def validate_public_workspace(workspace: Path) -> None:
    """Inspect a quiescent tree, following only links confined to this root."""
    if workspace.is_symlink():
        raise UnsafeLOCAOutput("The public workspace root must be an ordinary directory.")
    root = workspace.resolve(strict=True)

    def visit(path: Path, ancestors: frozenset[tuple[int, int]]) -> None:
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise UnsafeLOCAOutput("An output contains a dangling or cyclic filesystem link.") from error
        if resolved != root and root not in resolved.parents:
            raise UnsafeLOCAOutput("An output link escapes the public workspace.")
        metadata = resolved.stat()
        if stat.S_ISDIR(metadata.st_mode):
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in ancestors:
                raise UnsafeLOCAOutput("An output directory link creates a cycle.")
            for child in resolved.iterdir():
                visit(child, ancestors | {identity})
        elif not stat.S_ISREG(metadata.st_mode):
            raise UnsafeLOCAOutput("Task outputs must contain ordinary files and directories.")

    visit(root, frozenset())


def materialize_evaluation_workspace(workspace: Path) -> Path:
    """Replace a validated, quiescent public tree with independent normal files.

The caller must keep all MCP descendants paused through evaluator completion.
Their bind mounts retain the quarantined original tree, so later resumes cannot
change the copy read by the trusted evaluator at the original absolute path.
"""
    validate_public_workspace(workspace)
    staging = Path(tempfile.mkdtemp(prefix=".loca-evaluation-", dir=workspace.parent))
    copied = staging / "files"
    try:
        shutil.copytree(workspace, copied, symlinks=False)
        # Defense against special files or an unexpected external writer. The
        # copy has no shared source inodes and no remaining symbolic links.
        validate_public_workspace(copied)
        if any(path.is_symlink() for path in copied.rglob("*")):
            raise UnsafeLOCAOutput("Terminal materialization retained a symbolic link.")
        original = staging / "original-tool-workspace"
        workspace.rename(original)
        try:
            copied.rename(workspace)
        except BaseException:
            original.rename(workspace)
            raise
        return original
    except BaseException:
        if copied.exists():
            shutil.rmtree(copied)
        raise
