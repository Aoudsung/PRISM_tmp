#!/usr/bin/env python3
"""Live local web dashboard for monitoring remote F-GPO Colab experiments.

Reads remote telemetry read-only via SSH over `colab ssh --proxy-mode` (default,
avoids contending with the Jupyter kernel) with automatic fallback to `colab exec`,
using Python 3 standard library only (zero extra packages).
"""
from __future__ import annotations

import argparse
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from typing import Any


REMOTE_PROBE_SCRIPT = r'''
import json, os, pathlib, subprocess, time

CFG_PATH = os.environ["FGPO_CFG_PATH"]
OUT_DIR = os.environ["FGPO_OUT_DIR"]
LOG_PATH = os.environ["FGPO_LOG_PATH"]
SUITE = os.environ["FGPO_SUITE"]
BENCHMARK = os.environ["FGPO_BENCHMARK"]
METHOD = os.environ["FGPO_METHOD"]
SEEDS = [int(x) for x in os.environ["FGPO_SEEDS"].split(",") if x.strip()]

def sh(cmd):
    return subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True).stdout.strip()

data = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": sh("hostname"),
    "uptime": sh("uptime"),
    "gpu_csv": sh("nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits"),
    "cpu_count": os.cpu_count(),
    "loadavg": list(os.getloadavg()),
    "mem_info": sh("free -m"),
    "disk_df": sh("df -B1 /content | tail -n 1"),
    "procs": sh("ps -eo pid,ppid,pcpu,pmem,etime,args | grep -E 'run_colab_seq|run_fgpo_study|fgpo\\.envs' | grep -v grep"),
    "log_tail": sh(f"tail -n 80 '{LOG_PATH}' 2>/dev/null || true"),
    "seeds": {},
}

cfg_path = pathlib.Path(CFG_PATH)
if cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text())
    mc = (cfg.get("method_configs") or {}).get(METHOD) or {}
    data["config"] = {
        "protocol": cfg.get("protocol"),
        "rounds": mc.get("rounds") or cfg.get("rounds") or 60,
        "groups": cfg.get("groups") or 16,
        "trajectories": cfg.get("trajectories") or 8,
        "output_tokens": (mc.get("output_tokens_by_benchmark") or {}).get(BENCHMARK) or cfg.get("output_tokens") or 256,
        "checkpoint_interval_rounds": cfg.get("checkpoint_interval_rounds"),
        "host_memory_fraction": cfg.get("host_memory_fraction"),
        "host_runtime_reserve_gib": cfg.get("host_runtime_reserve_gib"),
        "retain_training_artifacts": cfg.get("retain_training_artifacts"),
        "model_path": cfg.get("model_path"),
        "eval_splits": (cfg.get("evaluation_splits") or {}).get(BENCHMARK, ["valid_unseen"]),
        "official_task_counts": cfg.get("official_task_counts") or {},
    }

def read_costs_incremental(cj: pathlib.Path, seed: int, rounds_done: int):
    cache_path = pathlib.Path(f"/tmp/fgpo_mon_cache_{METHOD}_{BENCHMARK}_{seed}.json")
    st = cj.stat()
    cache = None
    if cache_path.is_file():
        try:
            c = json.loads(cache_path.read_text())
            if c.get("ino") == st.st_ino and c.get("offset", 0) <= st.st_size:
                cache = c
        except Exception:
            cache = None
    if cache is None:
        cache = {
            "ino": st.st_ino,
            "offset": 0,
            "total_events": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "forward_tokens": 0,
            "env_steps": 0,
            "session_elapsed_seconds": None,
            "role_counts": {},
            "role_seconds": {},
            "last_group_id": None,
            "last_phase_role": None,
            "groups_by_round": {},
            "recent": [],
        }

    if st.st_size > cache["offset"]:
        with cj.open("rb") as fh:
            fh.seek(cache["offset"])
            chunk = fh.read()
        last_nl = chunk.rfind(b"\n")
        if last_nl >= 0:
            complete_bytes = chunk[: last_nl + 1]
            cache["offset"] += len(complete_bytes)
            for raw_ln in complete_bytes.decode("utf-8", errors="replace").splitlines():
                ln = raw_ln.strip()
                if not ln:
                    continue
                try:
                    ev = json.loads(ln)
                except Exception:
                    continue
                cache["total_events"] += 1
                role = ev.get("role", "unknown")
                cache["role_counts"][role] = cache["role_counts"].get(role, 0) + 1
                cache["role_seconds"][role] = cache["role_seconds"].get(role, 0.0) + float(ev.get("seconds") or 0.0)
                cache["input_tokens"] += int(ev.get("input_tokens") or 0)
                cache["output_tokens"] += int(ev.get("output_tokens") or 0)
                cache["forward_tokens"] += int(ev.get("forward_tokens") or 0)
                cache["env_steps"] += int(ev.get("env_steps") or 0)
                if ev.get("session_elapsed_seconds") is not None:
                    cache["session_elapsed_seconds"] = ev.get("session_elapsed_seconds")
                det = ev.get("details") or {}
                gid = det.get("group_id")
                if gid:
                    cache["last_group_id"] = gid
                if role in (
                    "rollout.group", "rollout.step_native", "rollout.decode",
                    "reference.single_round_activation", "credit.peer_values",
                    "ppo.backward", "ppo.adamw", "evaluation.rollout",
                ):
                    cache["last_phase_role"] = role
                if role == "rollout.group" and isinstance(gid, str):
                    parts = gid.split(":")
                    if len(parts) >= 4:
                        rk = parts[2]
                        cache["groups_by_round"][rk] = cache["groups_by_round"].get(rk, 0) + 1
                cache["recent"].append({
                    "event_index": ev.get("event_index"),
                    "kind": ev.get("kind"),
                    "role": role,
                    "seconds": ev.get("seconds"),
                    "env_steps": ev.get("env_steps"),
                    "input_tokens": ev.get("input_tokens"),
                    "output_tokens": ev.get("output_tokens"),
                    "group_id": gid,
                    "outcome": det.get("outcome"),
                    "elapsed": ev.get("session_elapsed_seconds"),
                })
                if len(cache["recent"]) > 20:
                    cache["recent"].pop(0)
            try:
                tmp = cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(cache, ensure_ascii=False))
                tmp.replace(cache_path)
            except Exception:
                pass

    cur_round_groups_done = cache["groups_by_round"].get(str(rounds_done), 0)
    summary = {
        "total_events": cache["total_events"],
        "input_tokens": cache["input_tokens"],
        "output_tokens": cache["output_tokens"],
        "forward_tokens": cache["forward_tokens"],
        "env_steps": cache["env_steps"],
        "session_elapsed_seconds": cache["session_elapsed_seconds"],
        "role_counts": cache["role_counts"],
        "role_seconds": cache["role_seconds"],
        "last_group_id": cache["last_group_id"],
        "last_phase_role": cache["last_phase_role"],
        "current_round_groups_done": cur_round_groups_done,
    }
    return summary, cache["recent"]

base = pathlib.Path(OUT_DIR) / SUITE / BENCHMARK / METHOD
for seed in SEEDS:
    d = base / f"seed-{seed}"
    if not d.exists():
        data["seeds"][str(seed)] = {"exists": False}
        continue
    s = {"exists": True, "du": sh(f"du -sh '{d}'/* 2>/dev/null || true")}
    tj = d / "training.jsonl"
    s["training_rounds"] = (
        [json.loads(ln) for ln in tj.read_text(errors="replace").splitlines() if ln.strip()]
        if tj.is_file() else []
    )
    groups_dir = d / "groups"
    if groups_dir.is_dir():
        cur_round = len(s["training_rounds"])
        s["current_round_group_files"] = len(list(groups_dir.glob(f"round-{cur_round}-group-*.pt")))
    else:
        s["current_round_group_files"] = 0

    cj = d / "costs.jsonl"
    if cj.is_file():
        s["costs_summary"], s["recent_cost_events"] = read_costs_incremental(cj, seed, len(s["training_rounds"]))

    ckpt_cur = d / "checkpoint" / "CURRENT"
    if ckpt_cur.is_file():
        s["checkpoint_current"] = json.loads(ckpt_cur.read_text())
    ts = d / "training_summary.json"
    if ts.is_file():
        s["training_summary"] = json.loads(ts.read_text())
    cs = d / "cell_summary.json"
    if cs.is_file():
        s["cell_summary"] = json.loads(cs.read_text())

    eval_splits = (data.get("config") or {}).get("eval_splits") or ["valid_unseen"]
    s["evaluations"] = {}
    for split in eval_splits:
        ev_info = {}
        ev_jsonl = d / f"evaluation-{split}.jsonl"
        if ev_jsonl.is_file():
            erows = [json.loads(ln) for ln in ev_jsonl.read_text(errors="replace").splitlines() if ln.strip()]
            ev_info["completed_count"] = len(erows)
            ev_info["success_count"] = sum(1 for r in erows if (r.get("official_evaluation") or {}).get("success"))
            ev_info["recent"] = [
                {
                    "task_id": r.get("task_id"),
                    "success": (r.get("official_evaluation") or {}).get("success"),
                    "steps": r.get("steps"),
                    "input_tokens": r.get("input_tokens"),
                    "output_tokens": r.get("output_tokens"),
                }
                for r in erows[-15:]
            ]
        ev_sum = d / split / "evaluation_summary.json"
        if ev_sum.is_file():
            ev_info["summary"] = json.loads(ev_sum.read_text())
        if ev_info:
            s["evaluations"][split] = ev_info
    data["seeds"][str(seed)] = s

print("__FGPO_JSON_START__")
print(json.dumps(data, ensure_ascii=False))
print("__FGPO_JSON_END__")
'''


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"
    if not env.get("HTTPS_PROXY") and not env.get("https_proxy"):
        try:
            with socket.create_connection(("127.0.0.1", 7897), 0.2):
                env["HTTPS_PROXY"] = "http://127.0.0.1:7897"
                env["HTTP_PROXY"] = "http://127.0.0.1:7897"
        except OSError:
            pass
    return env


def _parse_gpu_csv(raw: str) -> list[dict[str, Any]]:
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        def _num(val: str) -> float | None:
            try:
                return float(val)
            except ValueError:
                return None
        gpus.append({
            "index": parts[0],
            "name": parts[1],
            "temp_c": _num(parts[2]),
            "util_gpu_pct": _num(parts[3]),
            "util_mem_pct": _num(parts[4]),
            "mem_used_mib": _num(parts[5]),
            "mem_total_mib": _num(parts[6]),
            "power_draw_w": _num(parts[7]),
            "power_limit_w": _num(parts[8]),
        })
    return gpus


def _parse_mem_info(raw: str) -> dict[str, int | None]:
    for line in raw.splitlines():
        if line.startswith("Mem:"):
            cols = line.split()
            if len(cols) >= 7:
                return {
                    "total_mib": int(cols[1]),
                    "used_mib": int(cols[2]),
                    "free_mib": int(cols[3]),
                    "buff_cache_mib": int(cols[5]),
                    "available_mib": int(cols[6]),
                }
    return {}


def _parse_disk_df(raw: str) -> dict[str, Any]:
    cols = raw.split()
    if len(cols) >= 6:
        total_b = int(cols[1])
        used_b = int(cols[2])
        avail_b = int(cols[3])
        return {
            "filesystem": cols[0],
            "total_gib": round(total_b / (1024 ** 3), 2),
            "used_gib": round(used_b / (1024 ** 3), 2),
            "avail_gib": round(avail_b / (1024 ** 3), 2),
            "use_pct": cols[4],
            "mount": cols[5],
        }
    return {}


def _parse_procs(raw: str) -> list[dict[str, str]]:
    rows = []
    for line in raw.splitlines():
        parts = line.strip().split(None, 5)
        if len(parts) == 6:
            rows.append({
                "pid": parts[0],
                "ppid": parts[1],
                "cpu_pct": parts[2],
                "mem_pct": parts[3],
                "elapsed": parts[4],
                "cmd": parts[5],
            })
    return rows


def _parse_colab_usage(raw: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in raw.splitlines():
        if "Current balance:" in line:
            info["balance"] = line.split("Current balance:", 1)[1].strip()
        elif "Usage rate:" in line:
            info["rate"] = line.split("Usage rate:", 1)[1].strip()
        elif "Active assignments:" in line:
            info["active_assignments"] = line.split("Active assignments:", 1)[1].strip()
    return info


def _parse_colab_sessions(raw: str, target_session: str) -> dict[str, Any]:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("[colab]")]
    matched = next((ln for ln in lines if f"[{target_session}]" in ln), "")
    return {"raw_lines": lines, "target_line": matched, "active": bool(matched)}


class ColabPoller:
    """Thread-safe poller that queries Colab and maintains telemetry history."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.colab_bin = os.path.expanduser(args.colab_bin)
        self.env = _build_env()
        self.lock = threading.Lock()
        self.poll_lock = threading.Lock()
        self.history: deque[dict[str, Any]] = deque(maxlen=120)
        self.last_meta_poll_ts = 0.0
        self.state: dict[str, Any] = {
            "meta": {
                "session": args.session,
                "output_dir": args.output_dir,
                "log_path": args.log_path,
                "config_path": args.config_path,
                "suite": args.suite,
                "benchmark": args.benchmark,
                "method": args.method,
                "seeds": [int(s) for s in args.seeds.split(",") if s.strip()],
                "poll_interval_sec": args.interval,
                "transport": args.transport,
            },
            "status": "initializing",
            "polling": False,
            "last_ok_at": None,
            "last_attempt_at": None,
            "last_error": None,
            "last_transport": None,
            "colab": {},
            "telemetry": None,
            "history": [],
        }

    def _run_cli(self, cli_args: list[str], *, stdin_text: str | None = None, timeout: int = 45) -> str:
        proc = subprocess.run(
            [self.colab_bin, *cli_args],
            input=stdin_text,
            text=True,
            capture_output=True,
            env=self.env,
            timeout=timeout,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise RuntimeError(f"colab {' '.join(cli_args)}: {msg}")
        return proc.stdout

    def _run_remote_ssh(self, script: str, timeout: int) -> str:
        proxy_cmd = f"{self.colab_bin} ssh --proxy-mode -s {self.args.session}"
        proc = subprocess.run(
            [
                "ssh",
                "-T",
                "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=2",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", f"ProxyCommand={proxy_cmd}",
                "root@colab-runtime",
                "python3 -",
            ],
            input=script,
            text=True,
            capture_output=True,
            env=self.env,
            timeout=timeout,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or f"ssh exit {proc.returncode}").strip()
            raise RuntimeError(f"ssh proxy-mode: {msg}")
        return proc.stdout

    def _run_remote_exec(self, script: str, timeout: int) -> str:
        return self._run_cli(["exec", "-s", self.args.session], stdin_text=script, timeout=timeout)

    def _probe_remote(self, script: str) -> tuple[str, str]:
        mode = self.args.transport
        timeout = self.args.timeout
        if mode == "ssh":
            return self._run_remote_ssh(script, timeout), "ssh"
        if mode == "exec":
            return self._run_remote_exec(script, timeout), "exec"
        try:
            return self._run_remote_ssh(script, min(timeout, 45)), "ssh"
        except Exception as ssh_err:
            try:
                return self._run_remote_exec(script, timeout), "exec(fallback)"
            except Exception as exec_err:
                raise RuntimeError(f"SSH 失败 ({ssh_err}); colab exec 回退失败 ({exec_err})")

    def poll_once(self, *, force_meta: bool = False) -> dict[str, Any]:
        if not self.poll_lock.acquire(blocking=False):
            with self.lock:
                return dict(self.state)
        try:
            with self.lock:
                self.state["polling"] = True
                self.state["last_attempt_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            now = time.time()
            colab_info = self.state.get("colab") or {}
            if force_meta or not colab_info or (now - self.last_meta_poll_ts) >= 300:
                try:
                    sess_raw = self._run_cli(["sessions"], timeout=30)
                    usage_raw = self._run_cli(["usage"], timeout=30)
                    colab_info = {
                        "sessions": _parse_colab_sessions(sess_raw, self.args.session),
                        "usage": _parse_colab_usage(usage_raw),
                    }
                    self.last_meta_poll_ts = now
                except Exception as meta_err:
                    if not colab_info:
                        raise meta_err

            sessions_info = colab_info.get("sessions") or {"active": True}
            preamble = (
                "import os\n"
                f"os.environ['FGPO_CFG_PATH'] = {self.args.config_path!r}\n"
                f"os.environ['FGPO_OUT_DIR'] = {self.args.output_dir!r}\n"
                f"os.environ['FGPO_LOG_PATH'] = {self.args.log_path!r}\n"
                f"os.environ['FGPO_SUITE'] = {self.args.suite!r}\n"
                f"os.environ['FGPO_BENCHMARK'] = {self.args.benchmark!r}\n"
                f"os.environ['FGPO_METHOD'] = {self.args.method!r}\n"
                f"os.environ['FGPO_SEEDS'] = {self.args.seeds!r}\n"
            )

            telemetry = None
            used_transport = None
            if sessions_info.get("active", True):
                raw_out, used_transport = self._probe_remote(preamble + REMOTE_PROBE_SCRIPT)
                m = re.search(r"__FGPO_JSON_START__\s*(.*?)\s*__FGPO_JSON_END__", raw_out, re.S)
                if not m:
                    raise RuntimeError("Missing JSON marker from remote probe output")
                telemetry = json.loads(m.group(1))
                telemetry["gpus"] = _parse_gpu_csv(telemetry.get("gpu_csv", ""))
                telemetry["mem"] = _parse_mem_info(telemetry.get("mem_info", ""))
                telemetry["disk"] = _parse_disk_df(telemetry.get("disk_df", ""))
                telemetry["proc_list"] = _parse_procs(telemetry.get("procs", ""))

                gpu0 = telemetry["gpus"][0] if telemetry["gpus"] else {}
                active_seed_data = None
                for sk in [str(x) for x in self.state["meta"]["seeds"]]:
                    sd = (telemetry.get("seeds") or {}).get(sk) or {}
                    if sd.get("exists") and not sd.get("cell_summary"):
                        active_seed_data = sd
                        break
                if active_seed_data is None:
                    for sk in reversed([str(x) for x in self.state["meta"]["seeds"]]):
                        sd = (telemetry.get("seeds") or {}).get(sk) or {}
                        if sd.get("exists"):
                            active_seed_data = sd
                            break
                csum = (active_seed_data or {}).get("costs_summary") or {}
                self.history.append({
                    "ts": time.strftime("%H:%M:%S"),
                    "gpu_util": gpu0.get("util_gpu_pct"),
                    "mem_util": gpu0.get("util_mem_pct"),
                    "vram_used_gib": round((gpu0.get("mem_used_mib") or 0) / 1024, 2),
                    "power_w": gpu0.get("power_draw_w"),
                    "temp_c": gpu0.get("temp_c"),
                    "env_steps": csum.get("env_steps"),
                    "output_tokens": csum.get("output_tokens"),
                    "total_events": csum.get("total_events"),
                })

            with self.lock:
                self.state["status"] = "online" if sessions_info.get("active", True) else "session_inactive"
                self.state["polling"] = False
                self.state["last_ok_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.state["last_error"] = None
                self.state["last_transport"] = used_transport
                self.state["colab"] = colab_info
                if telemetry is not None:
                    self.state["telemetry"] = telemetry
                self.state["history"] = list(self.history)
                return dict(self.state)
        except Exception as exc:
            with self.lock:
                self.state["polling"] = False
                self.state["status"] = "stale" if self.state.get("telemetry") else "error"
                self.state["last_error"] = str(exc)
                return dict(self.state)
        finally:
            self.poll_lock.release()

    def get_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)


DASHBOARD_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Colab 远端实验实时监控台 · F-GPO r3</title>
  <style>
    :root {
      --bg: #0b0f17;
      --card: #121826;
      --card-2: #171f30;
      --border: #253047;
      --text: #e6edf7;
      --muted: #8b9bb4;
      --accent: #3b82f6;
      --c1: #3b82f6;
      --c2: #10b981;
      --c3: #f59e0b;
      --c4: #a855f7;
      --good: #10b981;
      --warn: #f59e0b;
      --bad: #ef4444;
      --code-bg: #090d14;
    }
    html.light {
      --bg: #f5f7fb;
      --card: #ffffff;
      --card-2: #f0f4fa;
      --border: #dce3ee;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #2563eb;
      --c1: #2563eb;
      --c2: #059669;
      --c3: #d97706;
      --c4: #7c3aed;
      --good: #059669;
      --warn: #d97706;
      --bad: #dc2626;
      --code-bg: #f8fafc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 13px;
      line-height: 1.5;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-variant-numeric: tabular-nums; }
    .container { max-width: 1380px; margin: 0 auto; padding: 18px 20px 40px; }
    .topbar {
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
      gap: 12px; padding-bottom: 14px; border-bottom: 1px solid var(--border); margin-bottom: 16px;
    }
    .title-wrap h1 { margin: 0; font-size: 18px; font-weight: 650; display: flex; align-items: center; gap: 10px; }
    .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .btn, select {
      background: var(--card); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 6px 11px; font-size: 12px; cursor: pointer;
    }
    .btn:hover { border-color: var(--accent); }
    .btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 600; }
    .badge {
      display: inline-flex; align-items: center; gap: 6px; padding: 2px 9px;
      border-radius: 999px; font-size: 11px; font-weight: 600; border: 1px solid var(--border);
      background: var(--card-2);
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
    .dot.good { background: var(--good); box-shadow: 0 0 8px var(--good); }
    .dot.warn { background: var(--warn); }
    .dot.bad { background: var(--bad); }
    .tabs { display: flex; gap: 6px; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .tab {
      background: transparent; color: var(--muted); border: 1px solid transparent;
      border-radius: 8px; padding: 6px 14px; font-size: 13px; font-weight: 500; cursor: pointer;
    }
    .tab.active { background: var(--card); color: var(--text); border-color: var(--border); }
    .grid-4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    @media (max-width: 1024px) { .grid-4 { grid-template-columns: repeat(2, 1fr); } .grid-3, .grid-2 { grid-template-columns: 1fr; } }
    .card {
      background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
    }
    .card-h { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
    .card-t { font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
    .kpi { font-size: 24px; font-weight: 700; line-height: 1.2; margin: 2px 0 4px; }
    .kpi-sub { font-size: 12px; color: var(--muted); }
    .bar-bg { width: 100%; height: 8px; background: var(--card-2); border-radius: 999px; overflow: hidden; margin-top: 8px; }
    .bar-fill { height: 100%; border-radius: 999px; transition: width .3s ease; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
    th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
    pre.logbox {
      background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 10px 12px; max-height: 320px; overflow: auto; margin: 0; font-size: 11.5px; line-height: 1.45;
      white-space: pre-wrap; word-break: break-word;
    }
    .hidden { display: none; }
    .err-banner {
      background: rgba(245, 158, 11, .12); border: 1px solid var(--warn); color: var(--text);
      padding: 10px 14px; border-radius: 10px; margin-bottom: 14px; font-size: 12px;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 11.5px; color: var(--muted); margin-bottom: 6px; }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
    svg.chart { width: 100%; display: block; }
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <div class="title-wrap">
        <h1>
          <span>Colab 远端实验实时监控台</span>
          <span id="status-badge" class="badge"><span class="dot warn"></span><span>初始化中</span></span>
        </h1>
        <div class="sub mono" id="subtitle">Session: - | Protocol: - | Last Poll: -</div>
      </div>
      <div class="controls">
        <label class="sub">自动刷新:
          <select id="auto-select">
            <option value="0">手动</option>
            <option value="15000">15s</option>
            <option value="30000" selected>30s</option>
            <option value="60000">60s</option>
          </select>
        </label>
        <button class="btn btn-primary" id="refresh-btn">立即探测 Colab</button>
        <button class="btn" id="theme-btn">切换亮/暗</button>
      </div>
    </div>

    <div id="err-box" class="err-banner hidden"></div>

    <div class="tabs">
      <button class="tab active" data-tab="overview">总览与服务器</button>
      <button class="tab" data-tab="experiment">实验进度与结果</button>
      <button class="tab" data-tab="costs">成本分解与事件流</button>
      <button class="tab" data-tab="logs">进程与远端日志</button>
    </div>

    <div id="tab-overview">
      <div class="grid-4" id="kpi-row"></div>
      <div class="grid-3" id="seed-cards"></div>
      <div class="grid-2">
        <div class="card">
          <div class="card-h"><span class="card-t">GPU 实时监控历史 (最近采样)</span><span class="kpi-sub mono" id="gpu-hist-meta">0 samples</span></div>
          <div class="legend">
            <span class="legend-item"><span class="swatch" style="background:var(--c1)"></span>GPU 利用率 %</span>
            <span class="legend-item"><span class="swatch" style="background:var(--c2)"></span>显存占用 %</span>
          </div>
          <svg id="gpu-chart" class="chart" viewBox="0 0 560 190"></svg>
        </div>
        <div class="card">
          <div class="card-h"><span class="card-t">主机资源与磁盘容量 (/content)</span><span class="kpi-sub mono" id="host-meta">-</span></div>
          <div id="host-bars"></div>
        </div>
      </div>
    </div>

    <div id="tab-experiment" class="hidden">
      <div class="card" style="margin-bottom:14px;">
        <div class="card-h">
          <span class="card-t">种子选择与实验契约</span>
          <div id="seed-pills" class="controls"></div>
        </div>
        <div id="seed-detail-summary"></div>
      </div>
      <div class="grid-2">
        <div class="card">
          <div class="card-h"><span class="card-t">训练轮次 Rollout 成功率 (SR %)</span><span class="kpi-sub mono" id="sr-meta">-</span></div>
          <svg id="sr-chart" class="chart" viewBox="0 0 560 220"></svg>
        </div>
        <div class="card">
          <div class="card-h"><span class="card-t">PPO Loss 与 Reference KL</span><span class="kpi-sub mono" id="loss-meta">-</span></div>
          <div class="legend">
            <span class="legend-item"><span class="swatch" style="background:var(--c3)"></span>Loss</span>
            <span class="legend-item"><span class="swatch" style="background:var(--c4)"></span>Reference KL</span>
          </div>
          <svg id="loss-chart" class="chart" viewBox="0 0 560 220"></svg>
        </div>
      </div>
      <div class="card">
        <div class="card-h"><span class="card-t">轮次明细与官方评估结果 (valid_unseen 134 局)</span></div>
        <div id="rounds-table-wrap" style="overflow:auto; max-height:360px;"></div>
      </div>
    </div>

    <div id="tab-costs" class="hidden">
      <div class="grid-2">
        <div class="card">
          <div class="card-h"><span class="card-t">Costs Ledger 算子调用分布</span><span class="kpi-sub mono" id="cost-total-meta">-</span></div>
          <div id="cost-bars"></div>
        </div>
        <div class="card">
          <div class="card-h"><span class="card-t">最近 20 条真实交互与模型事件 (costs.jsonl)</span></div>
          <div id="recent-events-wrap" style="overflow:auto; max-height:420px;"></div>
        </div>
      </div>
    </div>

    <div id="tab-logs" class="hidden">
      <div class="card" style="margin-bottom:14px;">
        <div class="card-h"><span class="card-t">远端活跃进程树</span></div>
        <div id="procs-wrap" style="overflow:auto;"></div>
      </div>
      <div class="card">
        <div class="card-h"><span class="card-t">远端主日志尾部 (/content/logs/fgpo_956ceab_alf.log)</span></div>
        <pre id="log-tail" class="logbox mono">等待拉取...</pre>
      </div>
    </div>
  </div>

  <script>
    let currentTab = "overview";
    let selectedSeed = "0";
    let lastData = null;
    let timer = null;

    const fmtInt = (n) => (n == null ? "-" : Number(n).toLocaleString("en-US"));
    const fmtNum = (n, d = 2) => (n == null ? "-" : Number(n).toFixed(d));
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        currentTab = btn.dataset.tab;
        document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
        ["overview", "experiment", "costs", "logs"].forEach((t) => {
          document.getElementById("panel-" + t).classList.toggle("hidden", t !== currentTab);
        });
      });
    });

    document.getElementById("theme-btn").addEventListener("click", () => {
      document.documentElement.classList.toggle("light");
    });

    document.getElementById("refresh-btn").addEventListener("click", async () => {
      const btn = document.getElementById("refresh-btn");
      btn.disabled = true;
      btn.textContent = "探测中...";
      try {
        const res = await fetch("/api/refresh", { method: "POST" });
        const data = await res.json();
        render(data);
      } finally {
        btn.disabled = false;
        btn.textContent = "立即探测 Colab";
      }
    });

    const autoSel = document.getElementById("auto-select");
    function resetTimer() {
      if (timer) clearInterval(timer);
      const ms = Number(autoSel.value);
      if (ms > 0) timer = setInterval(fetchState, ms);
    }
    autoSel.addEventListener("change", resetTimer);

    async function fetchState() {
      try {
        const res = await fetch("/api/state");
        const data = await res.json();
        render(data);
      } catch (e) {
        console.error(e);
      }
    }

    function detectPhase(seedData, cfg) {
      if (!seedData || !seedData.exists) return { label: "排队等待", cls: "warn", detail: "尚未启动" };
      if (seedData.cell_summary && seedData.cell_summary.status === "complete") {
        const sr = ((seedData.cell_summary.evaluation || {}).valid_unseen || {}).SR;
        return { label: "已完成", cls: "good", detail: sr != null ? `valid_unseen SR = ${fmtNum(sr, 2)}%` : "Complete" };
      }
      const evMap = seedData.evaluations || {};
      const evUnseen = evMap.valid_unseen;
      const totalEval = (cfg.official_task_counts || {}).valid_unseen || 134;
      if (evUnseen && evUnseen.completed_count != null) {
        return { label: "官方评估中", cls: "good", detail: `${evUnseen.completed_count}/${totalEval} tasks` };
      }
      const roundsDone = (seedData.training_rounds || []).length;
      const totalRounds = cfg.rounds || 60;
      const csum = seedData.costs_summary || {};
      const gDone = Math.max(csum.current_round_groups_done || 0, seedData.current_round_group_files || 0);
      const totalGroups = cfg.groups || 16;
      const role = csum.last_phase_role || "";
      let stage = "采样 Rollout";
      if (role === "reference.single_round_activation") stage = "Reference 前向";
      else if (role === "credit.peer_values") stage = "同深度同伴归因";
      else if (role === "ppo.backward" || role === "ppo.adamw") stage = "PPO 参数更新";
      return {
        label: `Round ${roundsDone}/${totalRounds}`,
        cls: "good",
        detail: `${stage} · Group ${gDone}/${totalGroups}`,
      };
    }

    function renderLineSvg(svgId, seriesList, labels, yMaxFixed) {
      const svg = document.getElementById(svgId);
      const W = 560, H = 200, padL = 40, padR = 16, padT = 14, padB = 26;
      const plotW = W - padL - padR, plotH = H - padT - padB;
      const allVals = seriesList.flatMap((s) => s.values.filter((v) => v != null));
      if (!allVals.length) {
        svg.innerHTML = `<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="var(--muted)" font-size="12">暂无已完成轮次 / 采样点，等待数据写入...</text>`;
        return;
      }
      const yMin = 0;
      const yMax = yMaxFixed != null ? yMaxFixed : Math.max(1e-6, ...allVals) * 1.15;
      let html = "";
      for (let i = 0; i <= 4; i++) {
        const y = padT + (plotH * i) / 4;
        const val = yMax - ((yMax - yMin) * i) / 4;
        html += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--border)" stroke-dasharray="3,3" />`;
        html += `<text x="${padL - 6}" y="${y + 4}" text-anchor="end" fill="var(--muted)" font-size="10">${val >= 10 ? val.toFixed(0) : val.toFixed(2)}</text>`;
      }
      seriesList.forEach((s) => {
        const pts = [];
        s.values.forEach((v, idx) => {
          if (v == null) return;
          const x = s.values.length === 1 ? padL + plotW / 2 : padL + (plotW * idx) / (s.values.length - 1);
          const y = padT + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;
          pts.push([x, y, v, labels[idx]]);
        });
        if (pts.length >= 2) {
          const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
          html += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" />`;
        }
        pts.forEach((p) => {
          html += `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.5" fill="${s.color}"><title>${esc(s.name)} (${esc(p[3])}): ${p[2]}</title></circle>`;
        });
      });
      if (labels.length >= 1) {
        const first = labels[0], last = labels[labels.length - 1];
        html += `<text x="${padL}" y="${H - 6}" fill="var(--muted)" font-size="10">${esc(first)}</text>`;
        if (labels.length > 1) {
          html += `<text x="${W - padR}" y="${H - 6}" text-anchor="end" fill="var(--muted)" font-size="10">${esc(last)}</text>`;
        }
      }
      svg.innerHTML = html;
    }

    function render(data) {
      lastData = data;
      const meta = data.meta || {};
      const t = data.telemetry || {};
      const colab = data.colab || {};
      const usage = colab.usage || {};
      const sess = colab.sessions || {};
      const cfg = t.config || {};

      const badge = document.getElementById("status-badge");
      if (data.status === "online") {
        badge.innerHTML = `<span class="dot good"></span><span>在线运行中 (${esc(data.last_transport || "-")})</span>`;
      } else if (data.status === "stale") {
        badge.innerHTML = `<span class="dot warn"></span><span>显示上次缓存 (探测重试中)</span>`;
      } else if (data.status === "session_inactive") {
        badge.innerHTML = `<span class="dot warn"></span><span>会话未激活</span>`;
      } else if (data.status === "error") {
        badge.innerHTML = `<span class="dot bad"></span><span>连接异常</span>`;
      } else {
        badge.innerHTML = `<span class="dot warn"></span><span>${esc(data.status)}</span>`;
      }

      document.getElementById("subtitle").textContent =
        `Session: ${meta.session} | Protocol: ${cfg.protocol || "-"} | Last OK: ${data.last_ok_at || "-"} | Remote UTC: ${t.timestamp_utc || "-"}`;

      const errBox = document.getElementById("err-box");
      if (data.last_error) {
        errBox.classList.remove("hidden");
        errBox.textContent = "最近探测异常（已保留最后一次成功快照）: " + data.last_error;
      } else {
        errBox.classList.add("hidden");
      }

      const gpu = (t.gpus || [])[0] || {};
      const disk = t.disk || {};
      const mem = t.mem || {};
      const vramPct = gpu.mem_total_mib ? (gpu.mem_used_mib / gpu.mem_total_mib) * 100 : 0;

      const seedsObj = t.seeds || {};
      const seedKeys = (meta.seeds || [0, 1, 2]).map(String);
      let activeKey = seedKeys.find((k) => (seedsObj[k] || {}).exists && !(seedsObj[k] || {}).cell_summary) || seedKeys[0];
      const activeSeed = seedsObj[activeKey] || {};
      const activeCsum = activeSeed.costs_summary || {};
      const activePhase = detectPhase(activeSeed, cfg);

      document.getElementById("kpi-row").innerHTML = `
        <div class="card">
          <div class="card-t">GPU 算力与温度</div>
          <div class="kpi mono">${gpu.util_gpu_pct ?? "-"}% <span style="font-size:14px;color:var(--muted)">${gpu.temp_c ?? "-"}°C</span></div>
          <div class="kpi-sub">${esc(gpu.name || "Colab GPU")} · ${fmtNum(gpu.power_draw_w, 0)}/${fmtNum(gpu.power_limit_w, 0)} W</div>
          <div class="bar-bg"><div class="bar-fill" style="width:${gpu.util_gpu_pct || 0}%;background:var(--c1)"></div></div>
        </div>
        <div class="card">
          <div class="card-t">显存占用 (VRAM)</div>
          <div class="kpi mono">${fmtNum((gpu.mem_used_mib || 0)/1024, 1)} <span style="font-size:14px;color:var(--muted)">/ ${fmtNum((gpu.mem_total_mib || 0)/1024, 1)} GiB</span></div>
          <div class="kpi-sub mono">${fmtInt(gpu.mem_used_mib)} / ${fmtInt(gpu.mem_total_mib)} MiB (${fmtNum(vramPct, 1)}%)</div>
          <div class="bar-bg"><div class="bar-fill" style="width:${vramPct}%;background:var(--c2)"></div></div>
        </div>
        <div class="card">
          <div class="card-t">Colab 算力单元</div>
          <div class="kpi mono">${esc(usage.balance || "-")}</div>
          <div class="kpi-sub mono">消耗速率: ${esc(usage.rate || "-")} · 实例: ${esc(usage.active_assignments || "-")}</div>
          <div class="kpi-sub mono">${esc(sess.target_line || "无匹配会话")}</div>
        </div>
        <div class="card">
          <div class="card-t">当前活跃单元 (Seed ${esc(activeKey)})</div>
          <div class="kpi mono">${esc(activePhase.label)}</div>
          <div class="kpi-sub mono">${esc(activePhase.detail)} · ${fmtInt(activeCsum.env_steps)} env steps</div>
          <div class="kpi-sub mono">已用时 ${fmtNum((activeCsum.session_elapsed_seconds || 0)/60, 1)} min · ${fmtInt(activeCsum.total_events)} 事件</div>
        </div>
      `;

      document.getElementById("seed-cards").innerHTML = seedKeys.map((sk) => {
        const sd = seedsObj[sk] || {};
        const ph = detectPhase(sd, cfg);
        const roundsDone = (sd.training_rounds || []).length;
        const totalRounds = cfg.rounds || 60;
        const csum = sd.costs_summary || {};
        const pct = sd.cell_summary ? 100 : Math.min(100, ((roundsDone + ((csum.current_round_groups_done || 0) / (cfg.groups || 16))) / totalRounds) * 100);
        return `
          <div class="card">
            <div class="card-h">
              <span class="card-t">ALFWorld · F-GPO · Seed ${esc(sk)}</span>
              <span class="badge"><span class="dot ${ph.cls}"></span><span>${esc(ph.label)}</span></span>
            </div>
            <div class="mono" style="font-size:13px;font-weight:600;">${esc(ph.detail)}</div>
            <div class="kpi-sub mono" style="margin-top:4px;">
              Tokens: ${fmtInt(csum.input_tokens)} in / ${fmtInt(csum.output_tokens)} out · Steps: ${fmtInt(csum.env_steps)}
            </div>
            <div class="bar-bg"><div class="bar-fill" style="width:${pct.toFixed(1)}%;background:var(--accent)"></div></div>
          </div>
        `;
      }).join("");

      const hist = data.history || [];
      document.getElementById("gpu-hist-meta").textContent = `${hist.length} samples`;
      renderLineSvg("gpu-chart", [
        { name: "GPU Util %", color: "var(--c1)", values: hist.map((h) => h.gpu_util) },
        { name: "VRAM %", color: "var(--c2)", values: hist.map((h) => (gpu.mem_total_mib ? Number(((h.vram_used_gib * 1024 / gpu.mem_total_mib) * 100).toFixed(1)) : null)) },
      ], hist.map((h) => h.ts), 100);

      const ramPct = mem.total_mib ? (mem.used_mib / mem.total_mib) * 100 : 0;
      const diskPct = disk.total_gib ? (disk.used_gib / disk.total_gib) * 100 : 0;
      document.getElementById("host-meta").textContent = `${t.cpu_count || "-"} vCPU · Load ${(t.loadavg || []).map((x) => x.toFixed(2)).join(" / ")}`;
      document.getElementById("host-bars").innerHTML = `
        <div style="margin-bottom:14px;">
          <div class="card-h" style="margin-bottom:4px;"><span>系统内存 (RAM)</span><span class="mono">${fmtNum((mem.used_mib||0)/1024,1)} / ${fmtNum((mem.total_mib||0)/1024,1)} GiB (可用 ${fmtNum((mem.available_mib||0)/1024,1)} GiB)</span></div>
          <div class="bar-bg"><div class="bar-fill" style="width:${ramPct}%;background:var(--c4)"></div></div>
        </div>
        <div style="margin-bottom:14px;">
          <div class="card-h" style="margin-bottom:4px;"><span>/content 磁盘空间</span><span class="mono">${disk.used_gib ?? "-"} / ${disk.total_gib ?? "-"} GiB (剩余 ${disk.avail_gib ?? "-"} GiB)</span></div>
          <div class="bar-bg"><div class="bar-fill" style="width:${diskPct}%;background:${diskPct > 85 ? 'var(--bad)' : 'var(--c3)'}"></div></div>
        </div>
        <div class="kpi-sub mono">Uptime: ${esc(t.uptime || "-")}</div>
        <div class="kpi-sub mono" style="margin-top:4px;">Checkpoint 间隔: 每 ${cfg.checkpoint_interval_rounds ?? "-"} 轮 · Retain artifacts: ${cfg.retain_training_artifacts}</div>
        <pre class="logbox mono" style="margin-top:10px;max-height:110px;">${esc(activeSeed.du || "(暂无目录大小信息)")}</pre>
      `;

      document.getElementById("seed-pills").innerHTML = seedKeys.map((sk) =>
        `<button class="btn ${sk === selectedSeed ? "btn-primary" : ""}" onclick="window._selSeed('${sk}')">Seed ${sk}</button>`
      ).join("");
      window._selSeed = (sk) => { selectedSeed = sk; if (lastData) render(lastData); };

      const sd = seedsObj[selectedSeed] || {};
      const rounds = sd.training_rounds || [];
      const ph = detectPhase(sd, cfg);
      const sdCsum = sd.costs_summary || {};
      document.getElementById("seed-detail-summary").innerHTML = `
        <div class="mono" style="font-size:12px;">
          状态: <b>${esc(ph.label)}</b> (${esc(ph.detail)}) |
          Checkpoint: ${esc((sd.checkpoint_current || {}).snapshot || "无")} |
          Last Group: ${esc(sdCsum.last_group_id || "-")} |
          总事件: ${fmtInt(sdCsum.total_events)} |
          总 Env Steps: ${fmtInt(sdCsum.env_steps)}
        </div>
      `;

      const rLabels = rounds.map((r) => `R${r.round}`);
      document.getElementById("sr-meta").textContent = `${rounds.length} / ${cfg.rounds || 60} rounds`;
      renderLineSvg("sr-chart", [
        { name: "Rollout SR %", color: "var(--c2)", values: rounds.map((r) => r.trajectories ? Number(((100 * r.successes) / r.trajectories).toFixed(2)) : 0) },
      ], rLabels, 100);

      document.getElementById("loss-meta").textContent = `${rounds.length} rounds`;
      renderLineSvg("loss-chart", [
        { name: "Loss", color: "var(--c3)", values: rounds.map((r) => (r.metrics || {}).loss) },
        { name: "Ref KL", color: "var(--c4)", values: rounds.map((r) => (r.metrics || {}).reference_kl) },
      ], rLabels, null);

      let tbl = "";
      if (sd.cell_summary && sd.cell_summary.evaluation) {
        const ev = sd.cell_summary.evaluation.valid_unseen || {};
        tbl += `<div class="badge" style="margin-bottom:10px;"><span class="dot good"></span><span class="mono">官方 valid_unseen: SR=${fmtNum(ev.SR, 2)}% (${ev.success_count}/${ev.evaluated_count}), mean_steps=${fmtNum(ev.mean_steps, 1)}</span></div>`;
      }
      if (rounds.length) {
        tbl += `<table><thead><tr><th>Round</th><th>Success/Traj</th><th>SR %</th><th>Opt Steps</th><th>Loss</th><th>KL</th><th>Entropy</th><th>Grad Norm</th><th>Zero Groups</th></tr></thead><tbody>`;
        rounds.slice().reverse().forEach((r) => {
          const m = r.metrics || {};
          const dg = r.diagnostics || {};
          const sr = r.trajectories ? (100 * r.successes) / r.trajectories : 0;
          tbl += `<tr class="mono"><td>${r.round}</td><td>${r.successes}/${r.trajectories}</td><td>${sr.toFixed(1)}%</td><td>${r.optimizer_steps}</td><td>${fmtNum(m.loss, 4)}</td><td>${fmtNum(m.reference_kl, 4)}</td><td>${fmtNum(m.entropy, 4)}</td><td>${fmtNum(m.gradient_norm, 3)}</td><td>${dg.all_zero_groups ?? "-"}</td></tr>`;
        });
        tbl += `</tbody></table>`;
      } else {
        tbl += `<div class="kpi-sub">Seed ${esc(selectedSeed)} 尚未完成第 0 轮（当前处于 ${esc(ph.detail)}）。完成后每轮指标、Loss、KL 与官方 valid_unseen 结果将自动展示在此处。</div>`;
      }
      document.getElementById("rounds-table-wrap").innerHTML = tbl;

      const roleCounts = sdCsum.role_counts || {};
      const roleSecs = sdCsum.role_seconds || {};
      const rolesSorted = Object.entries(roleCounts).sort((a, b) => b[1] - a[1]);
      const maxRoleCount = Math.max(1, ...rolesSorted.map((x) => x[1]));
      document.getElementById("cost-total-meta").textContent = `${fmtInt(sdCsum.total_events)} events · ${fmtNum(sdCsum.session_elapsed_seconds, 1)}s`;
      document.getElementById("cost-bars").innerHTML = rolesSorted.length ? rolesSorted.map(([role, cnt]) => {
        const w = (cnt / maxRoleCount) * 100;
        const sec = roleSecs[role] || 0;
        return `
          <div style="margin-bottom:9px;">
            <div class="card-h" style="margin-bottom:3px;"><span class="mono">${esc(role)}</span><span class="mono">${fmtInt(cnt)} 次 · ${fmtNum(sec, 2)}s</span></div>
            <div class="bar-bg" style="height:6px;margin-top:0;"><div class="bar-fill" style="width:${w}%;background:var(--c1)"></div></div>
          </div>
        `;
      }).join("") : `<div class="kpi-sub">暂无 costs.jsonl 事件</div>`;

      const recEvents = sd.recent_cost_events || [];
      document.getElementById("recent-events-wrap").innerHTML = recEvents.length ? `
        <table><thead><tr><th>#</th><th>Role</th><th>Group</th><th>Tokens (in/out)</th><th>Steps</th><th>Seconds</th><th>Outcome</th></tr></thead><tbody>
        ${recEvents.slice().reverse().map((e) => `<tr class="mono"><td>${e.event_index}</td><td>${esc(e.role)}</td><td>${esc(e.group_id || "-")}</td><td>${e.input_tokens || 0}/${e.output_tokens || 0}</td><td>${e.env_steps || 0}</td><td>${fmtNum(e.seconds, 3)}s</td><td>${esc(e.outcome || "-")}</td></tr>`).join("")}
        </tbody></table>` : `<div class="kpi-sub">暂无事件</div>`;

      const procs = t.proc_list || [];
      document.getElementById("procs-wrap").innerHTML = procs.length ? `
        <table><thead><tr><th>PID</th><th>PPID</th><th>CPU%</th><th>MEM%</th><th>Elapsed</th><th>Command</th></tr></thead><tbody>
        ${procs.map((p) => `<tr class="mono"><td>${esc(p.pid)}</td><td>${esc(p.ppid)}</td><td>${esc(p.cpu_pct)}</td><td>${esc(p.mem_pct)}</td><td>${esc(p.elapsed)}</td><td style="word-break:break-all;">${esc(p.cmd)}</td></tr>`).join("")}
        </tbody></table>` : `<div class="kpi-sub">未检测到活跃训练/环境进程</div>`;
      document.getElementById("log-tail").textContent = t.log_tail || "(日志为空)";
    }

    fetchState();
    resetTimer();
  </script>
</body>
</html>
'''


class MonitorHandler(BaseHTTPRequestHandler):
    poller: ColabPoller

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._send_json(self.poller.get_snapshot())
        else:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/refresh":
            self._send_json(self.poller.poll_once(force_meta=True))
        else:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)


def _background_loop(poller: ColabPoller, interval: int, stop_event: threading.Event) -> None:
    poller.poll_once(force_meta=True)
    if interval <= 0:
        return
    while not stop_event.wait(interval):
        poller.poll_once()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live local monitor UI for F-GPO Colab experiments.")
    parser.add_argument("--session", default="fgpo-r3-956ceab-alf")
    parser.add_argument("--output-dir", default="/content/outputs/fgpo-r3-956ceab-alf")
    parser.add_argument("--log-path", default="/content/logs/fgpo_956ceab_alf.log")
    parser.add_argument("--config-path", default="/content/fgpo-r3/configs/fgpo-r3-colab-956ceab-alf.json")
    parser.add_argument("--suite", default="comparison")
    parser.add_argument("--benchmark", default="alfworld")
    parser.add_argument("--method", default="fgpo")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=int, default=60, help="Auto-poll interval in seconds (0 = manual only)")
    parser.add_argument("--transport", choices=("auto", "ssh", "exec"), default="auto")
    parser.add_argument("--timeout", type=int, default=90, help="Remote probe timeout in seconds")
    parser.add_argument("--colab-bin", default="~/.local/bin/colab")
    args = parser.parse_args()

    poller = ColabPoller(args)
    stop_event = threading.Event()
    bg = threading.Thread(target=_background_loop, args=(poller, args.interval, stop_event), daemon=True)
    bg.start()

    class _Handler(MonitorHandler):
        pass
    _Handler.poller = poller

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"F-GPO Colab Monitor UI running at http://{args.host}:{args.port} "
        f"(session={args.session}, transport={args.transport}, interval={args.interval}s)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
