"""Per-service mount and process isolation for the official LOCA MCP tools.

The evaluator runs outside these namespaces. Arbitrary agent Python/terminal
code receives only its public workspace, public tool artifacts and runtime.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


_MOCK_DATA_ENV = {
    "calendar": "CALENDAR_DATA_DIR", "canvas": "CANVAS_DATA_DIR",
    "email": "EMAIL_DATA_DIR", "google_cloud": "GOOGLE_CLOUD_DATA_DIR",
    "google_sheet": "GOOGLE_SHEET_DATA_DIR", "snowflake": "SNOWFLAKE_DATA_DIR",
    "woocommerce": "WOOCOMMERCE_DATA_DIR",
}
_PUBLIC_ENV = frozenset({
    "ALLOWED_DIR", "ALLOWED_COMMANDS", "ALLOWED_FLAGS", "MAX_COMMAND_LENGTH",
    "COMMAND_TIMEOUT", "ALLOW_SHELL_OPERATORS", "MAX_OUTPUT_LENGTH",
    "MAX_STDOUT_LENGTH", "MAX_STDERR_LENGTH", "CLI_PROXY_ENABLED",
    "EXCEL_FILES_PATH", "MEMORY_FILE_PATH", "PYTHON_EXECUTE_WORKSPACE",
    "WORKSPACE_PATH", "TEMPFILE_DIR", * _MOCK_DATA_ENV.values(),
})
_SERVICES = frozenset({*_MOCK_DATA_ENV, "filesystem", "memory", "excel",
                       "pdf_tools", "python_execute", "terminal", "claim_done"})


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class LOCASandbox:
    """Build only allowlisted, offline MCP subprocess definitions."""

    def __init__(self, loca_root: Path, private_dir: Path, *,
                 runtime_paths: Sequence[str], runtime_roots: Sequence[str],
                 bwrap_executable: str) -> None:
        self.loca_root = loca_root.resolve()
        self.private_dir = private_dir.resolve()
        self.runtime_roots = tuple(Path(p).expanduser().resolve() for p in runtime_roots)
        if not self.runtime_roots:
            raise ValueError("LOCA requires explicit isolated runtime_roots.")
        forbidden = (Path.home().resolve(), Path("/"), self.loca_root,
                     self.loca_root.parent, self.loca_root.parent.parent)
        for root in self.runtime_roots:
            if not root.is_dir() or any(_under(value, root) for value in forbidden):
                raise ValueError(f"Unsafe or missing LOCA runtime root: {root}")
        system_paths = ("/usr/bin", "/bin")
        paths = tuple(str(Path(p).expanduser().absolute()) for p in runtime_paths)
        for entry in paths:
            if not any(_under(Path(entry).resolve(), root) for root in self.runtime_roots):
                raise ValueError(f"LOCA runtime PATH is outside isolated roots: {entry}")
        self.path = os.pathsep.join((*paths, *system_paths))
        self.bwrap = shutil.which(bwrap_executable)
        if not self.bwrap:
            raise RuntimeError("LOCA requires bubblewrap; unsandboxed MCP startup is forbidden.")
        self.code_root = self.private_dir / "public-tool-code"
        self.code_root.mkdir(parents=True, exist_ok=True)
        self._copy_code()

    def _copy_code(self) -> None:
        # Do not expose gem/envs, task configs, evaluator code, or reference files.
        for relative in ("mcp_convert/common", "mcp_convert/mcps",
                         "gem/tools/mcp_server/python_execute",
                         "gem/tools/mcp_server/claim_done"):
            source = self.loca_root / relative
            if not source.is_dir():
                raise FileNotFoundError(source)
            for path in source.rglob("*.py"):
                if path.is_symlink():
                    raise ValueError(f"Symlink in pinned LOCA tool source: {path}")
                target = self.code_root / path.relative_to(self.loca_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def wrap(self, definitions: Mapping[str, Any], *,
             server_types: Mapping[str, str], task_workspace: Path,
             agent_workspace: Path, artifact_dir: Path) -> dict[str, Any]:
        wrapped: dict[str, Any] = {}
        for name, original in definitions.items():
            service = server_types[name]
            if service not in _SERVICES:
                raise ValueError(f"Unreviewed LOCA MCP service: {service}")
            definition = deepcopy(dict(original))
            environment = dict(definition.get("env", {}))
            command, args = str(definition["command"]), list(definition.get("args", []))
            if service in {"filesystem", "memory"}:
                package = f"@modelcontextprotocol/server-{service}"
                if Path(command).name != "npx" or package not in args:
                    raise ValueError(f"Unexpected pinned npm command for {service}")
                package_index = args.index(package)
                command = f"mcp-server-{service}"
                args = args[package_index + 1:]
            elif command in {"python", "python3"}:
                command = sys.executable
            executable = shutil.which(command, path=self.path)
            if executable is None:
                raise FileNotFoundError(f"Offline LOCA executable unavailable: {command}")
            # Collapse '..' without resolving the venv's Python symlink: the
            # lexical venv path selects its own site-packages inside the sandbox.
            executable = os.path.abspath(executable)
            resolved = Path(executable).resolve()
            if not any(_under(resolved, root) for root in self.runtime_roots):
                raise ValueError(f"LOCA executable is outside isolated runtimes: {resolved}")
            allowed_env = {key: str(value) for key, value in environment.items()
                           if key in _PUBLIC_ENV}
            if allowed_env.get("CLI_PROXY_ENABLED", "false").lower() != "false":
                raise ValueError("LOCA terminal proxy is incompatible with offline isolation.")
            mounts: list[tuple[str, Path, Path]] = []
            for path in ("/usr/bin", "/usr/lib", "/usr/lib64", "/usr/share",
                         "/bin", "/lib", "/lib64", "/etc/ld.so.cache",
                         "/etc/alternatives", "/etc/localtime", "/etc/fonts"):
                source = Path(path)
                if source.exists():
                    mounts.append(("--ro-bind", source, source))
            mounts.extend(("--ro-bind", root, root) for root in self.runtime_roots)
            if service == "excel":
                sources = [path for root in self.runtime_roots
                           for path in root.glob("lib/python*/site-packages/excel_mcp/server.py")]
                if len(sources) != 1:
                    raise ValueError("Expected one isolated excel-mcp-server installation.")
                # Excel 0.1.8 hardcodes a package-root FileHandler. Keep code
                # read-only and bind only a task-private regular log file there.
                log_file = sources[0].parents[2] / "excel-mcp.log"
                log_file.touch(exist_ok=True)
                private_log = self.private_dir / "service-logs" / f"{task_workspace.name}-excel.log"
                private_log.parent.mkdir(parents=True, exist_ok=True)
                private_log.touch(mode=0o600, exist_ok=True)
                mounts.append(("--bind", private_log, log_file))
            # Each service has its own PID namespace. Its process list cannot
            # reveal another service's mock login arguments or trusted evaluator.
            mounts.append(("--bind", agent_workspace, agent_workspace))
            mounts.append(("--ro-bind", artifact_dir, artifact_dir))
            for relative in ("mcp_convert", "gem/tools/mcp_server/python_execute",
                             "gem/tools/mcp_server/claim_done"):
                mounts.append(("--ro-bind", self.code_root / relative,
                               self.loca_root / relative))
            if service in _MOCK_DATA_ENV:
                key = _MOCK_DATA_ENV[service]
                if key not in allowed_env:
                    raise ValueError(f"Missing native {service} data directory")
                database = Path(allowed_env[key]).resolve()
                database_root = (task_workspace / "local_db").resolve()
                if database == database_root or not _under(database, database_root):
                    raise ValueError(f"Unexpected native {service} database location")
                mounts.append(("--bind", database, database))
            for key in ("ALLOWED_DIR", "EXCEL_FILES_PATH", "MEMORY_FILE_PATH",
                        "PYTHON_EXECUTE_WORKSPACE", "WORKSPACE_PATH", "TEMPFILE_DIR"):
                if key in allowed_env and not _under(Path(allowed_env[key]).resolve(), agent_workspace):
                    raise ValueError(f"Native {service} public path escapes agent workspace: {key}")
            argv = ["--unshare-all", "--die-with-parent", "--new-session",
                    "--clearenv", "--proc", "/proc", "--dev", "/dev",
                    "--tmpfs", "/tmp", "--tmpfs", "/home", "--dir", "/home/agent"]
            for option, source, target in mounts:
                argv.extend((option, str(source), str(target)))
            clean_env = {
                "PATH": self.path, "HOME": "/home/agent", "TMPDIR": "/tmp",
                "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "LOCA_QUIET": "1",
                "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1", "UV_OFFLINE": "1",
                "npm_config_offline": "true", **allowed_env,
            }
            for key, value in clean_env.items():
                argv.extend(("--setenv", key, value))
            argv.extend(("--chdir", str(agent_workspace), "--", executable, *map(str, args)))
            wrapped[name] = {"command": self.bwrap, "args": argv,
                             "env": {"PATH": self.path, "HOME": "/nonexistent"}}
        return {"mcpServers": wrapped}

    def vm_service_specs(
        self, definitions: Mapping[str, Any], *,
        server_types: Mapping[str, str], task_workspace: Path,
        agent_workspace: Path, artifact_dir: Path,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Path], list[tuple[Path, Path]]]:
        specs: dict[str, dict[str, Any]] = {}
        databases: dict[str, Path] = {}
        for name, original in definitions.items():
            service = server_types[name]
            if service not in _SERVICES:
                raise ValueError(f"Unreviewed LOCA MCP service: {service}")
            definition = deepcopy(dict(original))
            environment = dict(definition.get("env", {}))
            command, args = str(definition["command"]), list(definition.get("args", []))
            if service in {"filesystem", "memory"}:
                package = f"@modelcontextprotocol/server-{service}"
                if Path(command).name != "npx" or package not in args:
                    raise ValueError(f"Unexpected pinned npm command for {service}")
                package_index = args.index(package)
                command = f"mcp-server-{service}"
                args = args[package_index + 1:]
            elif command in {"python", "python3"}:
                command = sys.executable
            executable = shutil.which(command, path=self.path)
            if executable is None:
                raise FileNotFoundError(f"Offline LOCA executable unavailable: {command}")
            executable = os.path.abspath(executable)
            resolved = Path(executable).resolve()
            if not any(_under(resolved, root) for root in self.runtime_roots):
                raise ValueError(f"LOCA executable is outside isolated runtimes: {resolved}")
            allowed_env = {key: str(value) for key, value in environment.items()
                           if key in _PUBLIC_ENV}
            if allowed_env.get("CLI_PROXY_ENABLED", "false").lower() != "false":
                raise ValueError("LOCA terminal proxy is incompatible with offline isolation.")
            if service in _MOCK_DATA_ENV:
                key = _MOCK_DATA_ENV[service]
                if key not in allowed_env:
                    raise ValueError(f"Missing native {service} data directory")
                database = Path(allowed_env[key]).resolve()
                database_root = (task_workspace / "local_db").resolve()
                if database == database_root or not _under(database, database_root):
                    raise ValueError(f"Unexpected native {service} database location")
                databases[name] = database
            for key in ("ALLOWED_DIR", "EXCEL_FILES_PATH", "MEMORY_FILE_PATH",
                        "PYTHON_EXECUTE_WORKSPACE", "WORKSPACE_PATH", "TEMPFILE_DIR"):
                if key in allowed_env and not _under(Path(allowed_env[key]).resolve(), agent_workspace):
                    raise ValueError(f"Native {service} public path escapes agent workspace: {key}")
            clean_env = {
                "PATH": self.path, "HOME": "/home/agent", "TMPDIR": "/tmp",
                "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "LOCA_QUIET": "1",
                "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1", "UV_OFFLINE": "1",
                "npm_config_offline": "true", **allowed_env,
            }
            specs[name] = {
                "service": service,
                "executable": executable,
                "args": [str(a) for a in args],
                "env": clean_env,
                "cwd": str(agent_workspace),
            }
        code_mounts = [
            (self.code_root / relative, self.loca_root / relative)
            for relative in ("mcp_convert", "gem/tools/mcp_server/python_execute",
                             "gem/tools/mcp_server/claim_done")
        ]
        return specs, databases, code_mounts
