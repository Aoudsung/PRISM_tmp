#!/usr/bin/env python3
"""Unified live local web dashboard for monitoring remote F-GPO V4 experiments
across all 3 servers in parallel (`Colab`, `Host A`, `Host B`).

- Aligned to the 18-cell V4 formal schedule (`compute_schedule_18_cells.csv`),
  covering ALFWorld, WebShop, and AppWorld across 5 execution lanes plus
  ALFWorld `valid_seen` replay and archived Host A GiGPO baseline cells.
- Strictly read-only remote telemetry collection using Python 3 stdlib only.
- Zero hardcoded credentials, IPs, ports, usernames, or tokens: parses
  `--ssh-doc` (`SSH_Document.md`) in memory at runtime for Host A / Host B and
  scrubs all sensitive strings before storing or serving `/api/state`.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import fcntl
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pty
import re
import select
import socket
import subprocess
import termios
import threading
import time
from typing import Any
import urllib.parse



LANE_DEFINITIONS: list[dict[str, Any]] = [
    {"lane_id": "colab_gpu_0", "server_id": "colab", "lane_title": "Colab · PRO 6000 (GPU 0)", "short_lane": "Colab GPU 0", "role_slot": "Colab G4 / PRO 6000", "gpu_indices": [0]},
    {"lane_id": "host_a_l40_3_4", "server_id": "host_a", "lane_title": "Host A · L40 3+4 (Queue A1)", "short_lane": "L40 3+4", "role_slot": "Queue A1 (3+4)", "gpu_indices": [3, 4]},
    {"lane_id": "host_a_l40_1_7", "server_id": "host_a", "lane_title": "Host A · L40 1+7 (Queue A2)", "short_lane": "L40 1+7", "role_slot": "Queue A2 (1+7)", "gpu_indices": [1, 7]},
    {"lane_id": "host_b_gpu_0", "server_id": "host_b", "lane_title": "Host B · GPU 0 (Queue B0)", "short_lane": "GPU 0", "role_slot": "Queue B0 (GPU 0 · 80 GiB cap)", "gpu_indices": [0]},
    {"lane_id": "host_b_gpu_2", "server_id": "host_b", "lane_title": "Host B · GPU 2 (Queue B2)", "short_lane": "GPU 2", "role_slot": "Queue B2 (GPU 2 · 80 GiB cap)", "gpu_indices": [2]},
]


# ---------------------------------------------------------------------------
# Shared Read-Only Remote Probe Helpers (Python 3.8+ universal stdlib)
# ---------------------------------------------------------------------------

REMOTE_COMMON_HELPERS = r'''
import glob, hashlib, json, os, pathlib, subprocess, time

MY_PID = os.getpid()

PHASE_ROLES = (
    "rollout.group", "rollout.step_native", "rollout.decode",
    "rollout.segment_replay", "rollout.merge_cohorts",
    "native.step", "native.reset",
    "gigpo.actor.decode", "gigpo.actor.decode.head", "gigpo.actor.batch_prefill",
    "g2po.actor.decode", "g2po.actor.decode.head", "g2po.actor.batch_prefill",
    "baseline.advantages",
    "reference.single_round_activation", "reference.packed_segment", "reference.local_observations",
    "credit.peer_values",
    "ppo.current_segment", "ppo.forward", "ppo.backward", "ppo.adamw",
    "evaluation.rollout", "rollout.official_evaluation", "native.official_scoring",
)

def sh(cmd):
    try:
        return subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, timeout=25).stdout.strip()
    except Exception:
        return ""

def parse_num(v):
    try:
        return float(v)
    except Exception:
        return None

def collect_gpus():
    cmd_g = [
        "nvidia-smi",
        "--query-gpu=index,gpu_bus_id,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    cmd_p = [
        "nvidia-smi",
        "--query-compute-apps=gpu_bus_id,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    raw_g, raw_p = "", ""
    pg = None
    pp = None
    try:
        pg = subprocess.Popen(cmd_g, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pp = subprocess.Popen(cmd_p, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out_g, _ = pg.communicate(timeout=20)
        out_p, _ = pp.communicate(timeout=20)
        raw_g = (out_g or "").strip()
        raw_p = (out_p or "").strip()
    except Exception:
        pass
    finally:
        for proc in (pg, pp):
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
    bus_pids = {}
    for ln in raw_p.splitlines():
        pts = [x.strip() for x in ln.split(",")]
        if len(pts) >= 3 and pts[1].isdigit():
            bus = pts[0].lower()
            pid_i = int(pts[1])
            mem_f = parse_num(pts[2]) or 0.0
            bus_pids.setdefault(bus, []).append({"pid": pid_i, "used_mib": mem_f})
    gpus = []
    for ln in raw_g.splitlines():
        pts = [x.strip() for x in ln.split(",")]
        if len(pts) < 10:
            continue
        idx = int(pts[0]) if pts[0].isdigit() else pts[0]
        bus = pts[1].lower()
        apps = bus_pids.get(bus, [])
        gpus.append({
            "index": idx,
            "bus_id": pts[1],
            "name": pts[2],
            "temp_c": parse_num(pts[3]),
            "util_gpu_pct": parse_num(pts[4]),
            "util_mem_pct": parse_num(pts[5]),
            "mem_used_mib": parse_num(pts[6]),
            "mem_total_mib": parse_num(pts[7]),
            "power_draw_w": parse_num(pts[8]),
            "power_limit_w": parse_num(pts[9]),
            "pids": [a["pid"] for a in apps],
            "pid_mem_mib": {str(a["pid"]): a["used_mib"] for a in apps},
        })
    return gpus

def annotate_gpu_fgpo_vram(gpus, procs):
    fgpo_pids = {str(p["pid"]) for p in procs}
    for g in gpus:
        pm = g.get("pid_mem_mib") or {}
        fgpo_m = round(sum(v for k, v in pm.items() if k in fgpo_pids), 1)
        ext_m = round(sum(v for k, v in pm.items() if k not in fgpo_pids), 1)
        g["experiment_mem_used_mib"] = fgpo_m
        g["fgpo_mem_used_mib"] = fgpo_m
        g["external_mem_used_mib"] = ext_m

def collect_mem_info():
    raw = sh("free -m")
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

def collect_disks(mounts):
    out = []
    seen = set()
    for m in mounts:
        if not os.path.exists(m):
            continue
        raw = sh("df -B1 " + repr(m) + " 2>/dev/null | tail -n 1")
        cols = raw.split()
        if len(cols) >= 6:
            mnt = cols[5]
            if mnt in seen and m != "/content":
                continue
            seen.add(mnt)
            tot = int(cols[1])
            usd = int(cols[2])
            avl = int(cols[3])
            pct_num = round(100.0 * usd / tot, 2) if tot > 0 else 0.0
            out.append({
                "filesystem": cols[0],
                "total_gib": round(tot / (1024 ** 3), 2),
                "used_gib": round(usd / (1024 ** 3), 2),
                "avail_gib": round(avl / (1024 ** 3), 2),
                "use_pct": cols[4],
                "used_pct_num": pct_num,
                "mount": mnt,
            })
    return out

def read_cpu_affinity(pid):
    try:
        txt = pathlib.Path(f"/proc/{pid}/status").read_text(errors="replace")
        for ln in txt.splitlines():
            if ln.startswith("Cpus_allowed_list:"):
                return ln.split(":", 1)[1].strip()
    except Exception:
        pass
    return "-"

def parse_runner_cli(cmd):
    toks = cmd.split()
    info = {"target_resume": "--resume" in toks}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "--output-dir" and i + 1 < len(toks):
            info["target_output_dir"] = toks[i + 1]
            i += 2
            continue
        if t == "--suite" and i + 1 < len(toks):
            info["target_suite"] = toks[i + 1]
            i += 2
            continue
        if t in ("--benchmarks", "--benchmark") and i + 1 < len(toks):
            info["target_benchmark"] = toks[i + 1]
            i += 2
            continue
        if t in ("--methods", "--method") and i + 1 < len(toks):
            info["target_method"] = toks[i + 1]
            i += 2
            continue
        if t in ("--seeds", "--seed") and i + 1 < len(toks):
            seeds = []
            j = i + 1
            while j < len(toks) and not toks[j].startswith("--"):
                for part in toks[j].split(","):
                    if part.strip().isdigit():
                        seeds.append(int(part.strip()))
                j += 1
            info["target_seeds"] = seeds
            i = j
            continue
        i += 1
    return info

def collect_fgpo_processes(default_lane=""):
    raw = sh("ps -eo pid=,ppid=,pcpu=,pmem=,rss=,etime=,args= 2>/dev/null")
    pat_tokens = (
        "fgpo.runner", "fgpo.envs", "capped_study.py", "launch_lane.sh",
        "launch_gpu2_v4.sh", "launch-webshop.sh", "webshop-3-4",
        "run-queue-", "run_colab_seq", "run_fgpo_study", "fgpo_gcs_checkpoint",
        ".fgpo-colab-supervisor.sh", "run_fgpo_host_b_webshop_queue.sh", "launch-host-a-textcraft.sh",
    )
    all_ps = []
    matched_pids = set()
    for ln in raw.splitlines():
        parts = ln.strip().split(None, 6)
        if len(parts) < 7 or not parts[0].isdigit():
            continue
        pid_i = int(parts[0])
        ppid_i = int(parts[1]) if parts[1].isdigit() else 0
        if pid_i == MY_PID or ppid_i == MY_PID:
            continue
        cmd = parts[6]
        if "grep " in cmd or "ps -eo" in cmd:
            continue
        rec = (pid_i, ppid_i, parts[2], parts[3], parts[4], parts[5], cmd)
        all_ps.append(rec)
        if any(tok in cmd for tok in pat_tokens):
            matched_pids.add(pid_i)

    # Also capture direct child waiters (e.g., `tail --pid=...` under `launch_gpu2_v4.sh`)
    for rec in all_ps:
        if rec[1] in matched_pids and ("tail " in rec[6] and "--pid=" in rec[6]):
            matched_pids.add(rec[0])

    rows = []
    for pid_i, ppid_i, pcpu, pmem, rss_s, etime, cmd in all_ps:
        if pid_i not in matched_pids:
            continue
        rss_mib = round(int(rss_s) / 1024.0, 1) if rss_s.isdigit() else None
        role = "process"
        if "launch_gpu2_v4.sh" in cmd:
            role = "handoff_supervisor"
        elif "tail " in cmd and "--pid=" in cmd:
            role = "handoff_waiter"
        elif "fgpo.envs" in cmd:
            role = "fgpo.envs"
        elif "capped_study.py" in cmd:
            role = "capped_study / fgpo.runner"
        elif "fgpo.runner" in cmd:
            role = "fgpo.runner"
        elif any(name in cmd for name in ("run-queue-", "launch-webshop", "webshop-3-4",
                                          "run_fgpo_host_b_webshop_queue.sh", "launch-host-a-textcraft.sh")):
            role = "queue_controller"
        elif "launch_lane.sh" in cmd or "supervisor" in cmd or "run_colab_seq" in cmd:
            role = "supervisor"

        lane = default_lane
        if "3-4" in cmd or "--gpu 3 4" in cmd or "launch-webshop" in cmd or "study-webshop" in cmd:
            lane = "L40 3+4"
        elif "1-7" in cmd or "--gpu 1 7" in cmd:
            lane = "L40 1+7"
        elif "launch_lane.sh 2" in cmd or "launch_gpu2_v4" in cmd or "--gpu 2" in cmd or "run_fgpo_host_b_webshop_queue.sh 2" in cmd:
            lane = "GPU 2"
        elif "launch_lane.sh 0" in cmd or "run_fgpo_host_b_webshop_queue.sh 0" in cmd or ("--gpu 0" in cmd and not default_lane):
            lane = "GPU 0"

        item = {
            "pid": str(pid_i),
            "ppid": str(ppid_i),
            "cpu_pct": pcpu,
            "mem_pct": pmem,
            "rss_mib": rss_mib,
            "affinity": read_cpu_affinity(pid_i),
            "elapsed": etime,
            "role": role,
            "lane": lane,
            "cmd": cmd,
        }
        if "runner" in role:
            item.update(parse_runner_cli(cmd))
        rows.append(item)

    by_pid = {r["pid"]: r for r in rows}
    for _ in range(3):
        for r in rows:
            if not r["lane"] and r["ppid"] in by_pid and by_pid[r["ppid"]]["lane"]:
                r["lane"] = by_pid[r["ppid"]]["lane"]
    return rows

def read_costs_incremental(cj, rounds_done):
    key_hash = hashlib.sha1(str(cj).encode("utf-8")).hexdigest()[:16]
    cache_path = pathlib.Path(f"/tmp/fgpo_mon_v4_{key_hash}.json")
    st = cj.stat()
    cache = None
    if cache_path.is_file():
        try:
            c = json.loads(cache_path.read_text())
            if (c.get("ino") == st.st_ino and c.get("dev") == st.st_dev
                    and "successful_adamw_count" in c and 0 <= c.get("offset", 0) <= st.st_size):
                cache = c
        except Exception:
            cache = None
    if cache is None:
        cache = {
            "dev": st.st_dev,
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
            "successful_adamw_count": 0,
            "first_successful_adamw": None,
            "last_group_id": None,
            "last_phase_role": None,
            "last_raw_role": None,
            "groups_by_round": {},
            "recent": [],
        }

    if st.st_size > cache["offset"]:
        consumed = 0
        with cj.open("rb") as fh:
            fh.seek(cache["offset"])
            rem = b""
            while True:
                blk = fh.read(4 * 1024 * 1024)
                if not blk:
                    break
                buf = rem + blk
                last_nl = buf.rfind(b"\n")
                if last_nl < 0:
                    rem = buf
                    continue
                complete = buf[:last_nl + 1]
                rem = buf[last_nl + 1:]
                consumed += len(complete)
                for raw_ln in complete.decode("utf-8", errors="replace").splitlines():
                    ln = raw_ln.strip()
                    if not ln:
                        continue
                    try:
                        ev = json.loads(ln)
                    except Exception:
                        continue
                    cache["total_events"] += 1
                    role = ev.get("role", "unknown")
                    cache["last_raw_role"] = role
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
                    if role == "ppo.adamw" and det.get("status") == "OK":
                        cache["successful_adamw_count"] += 1
                        if cache["first_successful_adamw"] is None:
                            cache["first_successful_adamw"] = {
                                "event_index": ev.get("event_index"),
                                "round": det.get("round"),
                                "session_elapsed_seconds": ev.get("session_elapsed_seconds"),
                            }
                    if gid:
                        cache["last_group_id"] = gid
                    if role in PHASE_ROLES:
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
        if consumed > 0:
            cache["offset"] += consumed
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
        "successful_adamw_count": cache["successful_adamw_count"],
        "first_successful_adamw": cache["first_successful_adamw"],
        "last_group_id": cache["last_group_id"],
        "last_phase_role": cache["last_phase_role"] or cache["last_raw_role"],
        "last_raw_role": cache["last_raw_role"],
        "current_round_groups_done": cur_round_groups_done,
    }
    return summary, cache["recent"]

def read_cell_dir(d, default_splits, include_du=False):
    if not d.exists():
        return {"exists": False}
    s = {"exists": True}
    if include_du:
        s["du"] = sh(f"du -sh '{d}'/* 2>/dev/null || true")
    tj = d / "training.jsonl"
    rounds = []
    if tj.is_file():
        for ln in tj.read_text(errors="replace").splitlines():
            if ln.strip():
                try:
                    r = json.loads(ln)
                    diag = r.get("diagnostics") or {}
                    rounds.append({
                        "round": r.get("round"),
                        "iter": r.get("iter"),
                        "trajectories": r.get("trajectories"),
                        "successes": r.get("successes"),
                        "optimizer_steps": r.get("optimizer_steps"),
                        "metrics": r.get("metrics"),
                        "diagnostics": {
                            "all_zero_groups": diag.get("all_zero_groups"),
                            "mixed_groups": diag.get("mixed_groups"),
                        },
                    })
                except Exception:
                    pass
    s["training_rounds"] = rounds
    cur_round = len(rounds)
    groups_dir = d / "groups"
    if groups_dir.is_dir():
        s["current_round_group_files"] = len(list(groups_dir.glob(f"round-{cur_round}-group-*.pt")))
    else:
        s["current_round_group_files"] = 0

    cj = d / "costs.jsonl"
    csum_file = d / "costs_summary.json"
    if cj.is_file():
        s["costs_summary"], s["recent_cost_events"] = read_costs_incremental(cj, cur_round)
    elif csum_file.is_file():
        try:
            saved_cs = json.loads(csum_file.read_text())
        except Exception:
            saved_cs = {}
        s["costs_summary"] = {
            "total_events": saved_cs.get("total_events", 0),
            "input_tokens": saved_cs.get("input_tokens", 0),
            "output_tokens": saved_cs.get("output_tokens", 0),
            "forward_tokens": saved_cs.get("forward_tokens", 0),
            "env_steps": saved_cs.get("env_steps", 0),
            "session_elapsed_seconds": saved_cs.get("session_elapsed_seconds"),
            "role_counts": saved_cs.get("role_counts", {}),
            "role_seconds": saved_cs.get("role_seconds", {}),
            "last_group_id": saved_cs.get("last_group_id"),
            "last_phase_role": saved_cs.get("last_phase_role"),
            "current_round_groups_done": (saved_cs.get("groups_by_round") or {}).get(str(cur_round), 0),
        }
        s["recent_cost_events"] = saved_cs.get("recent", [])
    h_manifest = d / "history_bundle.manifest.json"
    if h_manifest.is_file():
        try:
            hb = json.loads(h_manifest.read_text())
            s["history_bundle"] = {k: v for k, v in hb.items() if k != "members"} if isinstance(hb, dict) else hb
        except Exception:
            pass
    ckpt_cur = d / "checkpoint" / "CURRENT"
    if ckpt_cur.is_file():
        try:
            s["checkpoint_current"] = json.loads(ckpt_cur.read_text())
        except Exception:
            pass
    s["gcs_tx_locked"] = (d / ".gcs-transaction.lock").exists() or (d / "checkpoint" / ".gcs-transaction.lock").exists()
    s["actor_cache_exists"] = (d / "actor-cache").exists()
    ts = d / "training_summary.json"
    if ts.is_file():
        try:
            ts_obj = json.loads(ts.read_text())
            s["training_summary"] = {k: v for k, v in ts_obj.items() if k not in ("rounds", "history")} if isinstance(ts_obj, dict) else ts_obj
        except Exception:
            pass
    cs = d / "cell_summary.json"
    if cs.is_file():
        try:
            cs_obj = json.loads(cs.read_text())
            s["cell_summary"] = {k: v for k, v in cs_obj.items() if k not in ("rounds", "history")} if isinstance(cs_obj, dict) else cs_obj
        except Exception:
            pass

    splits = list(default_splits or [])
    for ep in d.glob("evaluation-*.jsonl"):
        sp = ep.name[len("evaluation-"):-len(".jsonl")]
        if sp and sp not in splits:
            splits.append(sp)
    for ep in d.glob("*/evaluation_summary.json"):
        sp = ep.parent.name
        if sp and sp not in splits:
            splits.append(sp)

    s["evaluations"] = {}
    for split in splits:
        ev_info = {}
        ev_jsonl = d / f"evaluation-{split}.jsonl"
        if ev_jsonl.is_file():
            erows = []
            for ln in ev_jsonl.read_text(errors="replace").splitlines():
                if ln.strip():
                    try:
                        erows.append(json.loads(ln))
                    except Exception:
                        pass
            ev_info["completed_count"] = len(erows)
            ev_info["success_count"] = sum(1 for r in erows if (r.get("official_evaluation") or {}).get("success"))
            tgc_vals = [float((r.get("official_evaluation") or {}).get("task_goal_completion")) for r in erows if (r.get("official_evaluation") or {}).get("task_goal_completion") is not None]
            if tgc_vals:
                ev_info["tgc_pct"] = round(100.0 * sum(tgc_vals) / len(tgc_vals), 2)
            score_vals = [float((r.get("official_evaluation") or {}).get("score")) for r in erows if (r.get("official_evaluation") or {}).get("score") is not None]
            if score_vals:
                ev_info["score_pct"] = round(100.0 * sum(score_vals) / len(score_vals), 2) if max(score_vals) <= 1.0 else round(sum(score_vals) / len(score_vals), 2)
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
            try:
                ev_info["summary"] = json.loads(ev_sum.read_text())
            except Exception:
                pass
        if ev_info:
            s["evaluations"][split] = ev_info
    return s
'''

COLAB_PROBE_BODY = r'''
CFG_PATH = os.environ["FGPO_CFG_PATH"]
OUT_DIR = os.environ["FGPO_OUT_DIR"]
LOG_PATH = os.environ["FGPO_LOG_PATH"]
SUITE = os.environ["FGPO_SUITE"]
BENCHMARK = os.environ["FGPO_BENCHMARK"]
METHOD = os.environ["FGPO_METHOD"]
SEEDS = [int(x) for x in os.environ["FGPO_SEEDS"].split(",") if x.strip()]
CELL_SPECS = json.loads(os.environ["FGPO_CELL_SPECS"])

def collect_disk_breakdown(out_dir):
    content_kb = sh("du -sk /content 2>/dev/null | awk '{print $1}'")
    content_bytes = int(content_kb) * 1024 if content_kb.isdigit() else None
    pinned_seen = {}
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        mf_dir = p / "map_files"
        try:
            entries = list(mf_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                tgt = os.readlink(entry)
                if "(deleted)" in tgt and not tgt.startswith("/dev/"):
                    st = os.stat(entry)
                    pinned_seen[(st.st_dev, st.st_ino)] = st.st_size
            except OSError:
                continue
    lib_so = pathlib.Path("/content/venvs/alfworld/lib/python3.11/site-packages/fast_downward/libdownward.so")
    lib_so_bytes = lib_so.stat().st_size if lib_so.is_file() else None
    bundle_count = 0
    bundle_bytes = 0
    reclaimed_bytes = 0
    out_p = pathlib.Path(out_dir)
    if out_p.is_dir():
        for mp in out_p.glob("*/*/*/seed-*/history_bundle.manifest.json"):
            try:
                bp = mp.parent / "history_bundle.tar.zst"
                if not bp.is_file():
                    continue
                md = json.loads(mp.read_text())
                bundle_count += 1
                bundle_bytes += int(bp.stat().st_size)
                reclaimed_bytes += int(md.get("reclaimed_local_bytes") or md.get("uncompressed_member_bytes") or 0)
            except Exception:
                pass
    return {
        "content_visible_gib": round(content_bytes / (1024 ** 3), 2) if content_bytes is not None else None,
        "pinned_deleted_mmap_gib": round(sum(pinned_seen.values()) / (1024 ** 3), 2),
        "pinned_deleted_count": len(pinned_seen),
        "libdownward_mib": round(lib_so_bytes / (1024 ** 2), 2) if lib_so_bytes is not None else None,
        "history_bundles_count": bundle_count,
        "history_bundles_mib": round(bundle_bytes / (1024 ** 2), 2),
        "history_reclaimed_mib": round(reclaimed_bytes / (1024 ** 2), 2),
    }

procs = collect_fgpo_processes("Colab GPU 0")
gpus = collect_gpus()
for g in gpus:
    g["role_slot"] = "Colab G4 / PRO 6000"
    g["active_in_experiment"] = True
annotate_gpu_fgpo_vram(gpus, procs)
disks = collect_disks(["/content"])

data = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": sh("hostname"),
    "uptime": sh("uptime"),
    "cpu_count": os.cpu_count(),
    "numa_nodes": {"node0": f"0-{(os.cpu_count() or 48) - 1}"},
    "loadavg": list(os.getloadavg()),
    "mem": collect_mem_info(),
    "disks": disks,
    "disk": disks[0] if disks else {},
    "disk_breakdown": collect_disk_breakdown(OUT_DIR),
    "gpus": gpus,
    "excluded_gpus": [],
    "proc_list": procs,
    "seeds": {},
    "cells": {},
    "log_tails": {},
}

cfg_path = pathlib.Path(CFG_PATH)
eval_splits = ["valid_unseen"]
if cfg_path.is_file():
    try:
        cfg = json.loads(cfg_path.read_text())
        mc = (cfg.get("method_configs") or {}).get(METHOD) or {}
        eval_splits = (cfg.get("evaluation_splits") or {}).get(BENCHMARK, ["valid_unseen"])
        data["config"] = {
            "protocol": cfg.get("protocol"),
            "checkpoint_backend": cfg.get("checkpoint_backend", "local"),
            "rounds": mc.get("rounds") or cfg.get("rounds") or 60,
            "groups": cfg.get("groups") or 16,
            "trajectories": cfg.get("trajectories") or 8,
            "output_tokens": (mc.get("output_tokens_by_benchmark") or {}).get(BENCHMARK) or cfg.get("output_tokens") or 512,
            "checkpoint_interval_rounds": cfg.get("checkpoint_interval_rounds"),
            "retain_training_artifacts": cfg.get("retain_training_artifacts"),
            "eval_splits": eval_splits,
            "official_task_counts": cfg.get("official_task_counts") or {},
        }
    except Exception:
        pass

out_root = pathlib.Path(OUT_DIR)
for spec in CELL_SPECS:
    ck = f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
    cdir = out_root / spec["suite"] / spec["benchmark"] / spec["method"] / f"seed-{spec['seed']}"
    inc_du = (spec["suite"] == SUITE and spec["benchmark"] == BENCHMARK and spec["method"] == METHOD)
    data["cells"][ck] = read_cell_dir(cdir, ["valid_unseen"], include_du=inc_du)

for seed in SEEDS:
    ck = f"{SUITE}/{BENCHMARK}/{METHOD}/seed-{seed}"
    if ck in data["cells"]:
        data["seeds"][str(seed)] = data["cells"][ck]
    else:
        cdir = out_root / SUITE / BENCHMARK / METHOD / f"seed-{seed}"
        data["seeds"][str(seed)] = read_cell_dir(cdir, eval_splits, include_du=True)
        data["cells"][ck] = data["seeds"][str(seed)]

resolved_log = LOG_PATH
if not resolved_log or resolved_log == "auto":
    active_spec = None
    for spec in CELL_SPECS:
        ck = f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
        cd = data["cells"].get(ck) or {}
        if cd.get("exists") and not cd.get("cell_summary"):
            active_spec = spec
            break
    if active_spec:
        resolved_log = f"/content/logs/fgpo_{active_spec['suite']}_{active_spec['method']}_{active_spec['benchmark']}_s{active_spec['seed']}.log"
    else:
        resolved_log = f"/content/logs/fgpo_{SUITE}_{METHOD}_{BENCHMARK}_s0.log"

seed_tail = sh(f"tail -n 50 '{resolved_log}' 2>/dev/null || true")
sup_log = "/content/logs/fgpo_comparison_fgpo_supervisor.log"
sup_tail = sh(f"tail -n 35 '{sup_log}' 2>/dev/null || true")
data["log_path"] = resolved_log
data["log_tail"] = seed_tail or sup_tail
data["log_tails"]["Colab GPU 0 · Active Seed"] = {"path": resolved_log, "text": seed_tail}
if sup_tail:
    data["log_tails"]["Colab GPU 0 · GCS Supervisor"] = {"path": sup_log, "text": sup_tail}

print("__FGPO_JSON_START__")
print(json.dumps(data, ensure_ascii=False))
print("__FGPO_JSON_END__")
'''

HOST_A_PROBE_BODY = r'''
DEPLOY_HINT = os.environ.get("FGPO_DEPLOY_DIR", "auto")
CELL_SPECS = json.loads(os.environ["FGPO_CELL_SPECS"])

procs = collect_fgpo_processes("")

deploy_dir = None
if DEPLOY_HINT and DEPLOY_HINT != "auto" and os.path.isdir(DEPLOY_HINT):
    deploy_dir = pathlib.Path(DEPLOY_HINT)
else:
    for p in procs:
        out_arg = p.get("target_output_dir") or ""
        if out_arg:
            cand = pathlib.Path(out_arg)
            if cand.name.startswith("study") and cand.parent.is_dir():
                deploy_dir = cand.parent
                break
    if deploy_dir is None:
        cands = sorted(glob.glob("/apps/users/*/fgpo-v4-formal-host-a-*"), key=lambda x: os.path.getmtime(x), reverse=True)
        if cands:
            deploy_dir = pathlib.Path(cands[0])

all_gpus = collect_gpus()
annotate_gpu_fgpo_vram(all_gpus, procs)
usable_gpus = []
excluded_gpus = []
for g in all_gpus:
    idx = g["index"]
    if idx in (0, 2):
        g["role_slot"] = "Excluded (0, 2)"
        g["active_in_experiment"] = False
        excluded_gpus.append(g)
    else:
        if idx in (3, 4):
            g["role_slot"] = "Queue A1 (3+4)"
            g["sampling_layout"] = "gpu4_backbone_and_head"
            g["active_in_experiment"] = True
        elif idx in (1, 7):
            g["role_slot"] = "Queue A2 (1+7)"
            g["sampling_layout"] = "tuned_dual_l40"
            g["active_in_experiment"] = True
        else:
            g["role_slot"] = "Idle/Shared"
            g["active_in_experiment"] = False
        usable_gpus.append(g)

numa_nodes = {}
for np in sorted(glob.glob("/sys/devices/system/node/node*/cpulist")):
    try:
        numa_nodes[pathlib.Path(np).parent.name] = pathlib.Path(np).read_text().strip()
    except Exception:
        pass

data = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "deployment_dir": str(deploy_dir) if deploy_dir else None,
    "hostname": sh("hostname"),
    "uptime": sh("uptime"),
    "cpu_count": os.cpu_count(),
    "numa_nodes": numa_nodes,
    "loadavg": list(os.getloadavg()),
    "mem": collect_mem_info(),
    "disks": collect_disks(["/apps", "/data"]),
    "gpus": usable_gpus,
    "excluded_gpus": excluded_gpus,
    "proc_list": procs,
    "cells": {},
    "archived_cells": {},
    "log_tails": {},
    "config": {"protocol": "fgpo-r3-published-v4", "groups": 16, "trajectories": 8},
}

# Build candidate study roots (live --output-dir + study-webshop + study)
study_roots = []
for p in procs:
    od = p.get("target_output_dir")
    if od and os.path.isdir(od) and pathlib.Path(od) not in study_roots:
        study_roots.append(pathlib.Path(od))
if deploy_dir and deploy_dir.is_dir():
    study_roots.extend(p for p in sorted(deploy_dir.glob("study-textcraft*")) if p.is_dir() and p not in study_roots)
    for sub in ("study-webshop", "study"):
        sp = deploy_dir / sub
        if sp.is_dir() and sp not in study_roots:
            study_roots.append(sp)
    for cfg_cand in [deploy_dir / "configs" / "fgpo-r3-host-a-l40-1-7.json", deploy_dir / "configs" / "fgpo-r3-host-a-l40-3-4.json", deploy_dir / "configs" / "fgpo-r3-host-a.json"]:
        if cfg_cand.is_file():
            try:
                cjson = json.loads(cfg_cand.read_text())
                data["config"]["protocol"] = cjson.get("protocol", "fgpo-r3-published-v4")
                data["config"]["groups"] = cjson.get("groups") or 16
                data["config"]["trajectories"] = cjson.get("trajectories") or 8
                break
            except Exception:
                pass

for spec in CELL_SPECS:
    ck = f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
    bm = spec["benchmark"]
    def_splits = spec["official_eval"].split("+")
    ordered_roots = list(study_roots)
    if bm == "webshop":
        ordered_roots.sort(key=lambda r: 0 if "webshop" in r.name else 1)
    else:
        ordered_roots.sort(key=lambda r: 1 if "webshop" in r.name else 0)
    chosen_dir = None
    for sr in ordered_roots:
        cand = sr / spec["suite"] / bm / spec["method"] / f"seed-{spec['seed']}"
        if cand.exists():
            chosen_dir = cand
            break
    if chosen_dir is None and ordered_roots:
        chosen_dir = ordered_roots[0] / spec["suite"] / bm / spec["method"] / f"seed-{spec['seed']}"
    data["cells"][ck] = read_cell_dir(chosen_dir, def_splits, include_du=False) if chosen_dir is not None else {"exists": False}

# Archived Host A 7B GiGPO cells in <deploy>/study/comparison/alfworld/gigpo/seed-*
if deploy_dir and (deploy_dir / "study").is_dir():
    arch_base = deploy_dir / "study" / "comparison" / "alfworld" / "gigpo"
    if arch_base.is_dir():
        for sd_p in sorted(arch_base.glob("seed-*")):
            ack = f"comparison/alfworld/gigpo/{sd_p.name}"
            acell = read_cell_dir(sd_p, ["valid_unseen"], include_du=False)
            if acell.get("exists"):
                data["archived_cells"][ack] = acell

if deploy_dir and deploy_dir.is_dir():
    logs_dir = deploy_dir / "logs"
    # Lane L40 3+4 logs
    has_textcraft = any(spec["benchmark"] == "textcraft" for spec in CELL_SPECS)
    l34_candidates = [] if has_textcraft else [logs_dir / "webshop-3-4-v4-formal.log", logs_dir / "queue-3-4-v4.log", logs_dir / "queue-3-4.log"]
    l34_file = next((p for p in l34_candidates if p.is_file()), None)
    l34_tail = sh(f"tail -n 22 '{l34_file}' 2>/dev/null || true") if l34_file else ""
    l34_ev = []
    for spec in CELL_SPECS:
        if spec.get("physical_gpu_ids") != "3+4":
            continue
        ck = f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
        cd = data["cells"].get(ck) or {}
        if cd.get("exists") and not cd.get("cell_summary"):
            for r in (cd.get("training_rounds") or [])[-3:]:
                m = r.get("metrics") or {}
                l34_ev.append(f"[training R{r.get('round')}] succ={r.get('successes')}/{r.get('trajectories')} opt={r.get('optimizer_steps')} loss={m.get('loss')} kl={m.get('reference_kl')}")
            for ev in (cd.get("recent_cost_events") or [])[-8:]:
                l34_ev.append(f"[costs #{ev.get('event_index')} t={ev.get('elapsed')}s] role={ev.get('role')} sec={ev.get('seconds')} steps={ev.get('env_steps') or 0} group={ev.get('group_id') or '-'}")
            break
    data["log_tails"]["Host A · L40 3+4"] = {
        "path": str(l34_file or logs_dir),
        "text": "\n".join([x for x in [l34_tail, "\n".join(l34_ev)] if x]) or "(暂无日志输出)",
    }

    # Lane L40 1+7 logs
    l17_candidates = [] if has_textcraft else [logs_dir / "queue-1-7-v4.log", logs_dir / "queue-1-7.log"]
    l17_qfile = next((p for p in l17_candidates if p.is_file()), None)
    l17_qtail = sh(f"tail -n 18 '{l17_qfile}' 2>/dev/null || true") if l17_qfile else ""
    l17_cell_log = None
    l17_ev = []
    for spec in CELL_SPECS:
        if spec.get("physical_gpu_ids") != "1+7":
            continue
        ck = f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
        cd = data["cells"].get(ck) or {}
        if cd.get("exists") and not cd.get("cell_summary"):
            cand_log = logs_dir / f"{spec['suite']}.{spec['benchmark']}.{spec['method']}.seed-{spec['seed']}.log"
            if cand_log.is_file():
                l17_cell_log = cand_log
            for ev in (cd.get("recent_cost_events") or [])[-8:]:
                l17_ev.append(f"[costs #{ev.get('event_index')} t={ev.get('elapsed')}s] role={ev.get('role')} sec={ev.get('seconds')} steps={ev.get('env_steps') or 0} group={ev.get('group_id') or '-'}")
            break
    l17_ctail = sh(f"tail -n 18 '{l17_cell_log}' 2>/dev/null || true") if l17_cell_log else ""
    data["log_tails"]["Host A · L40 1+7"] = {
        "path": str(l17_cell_log or l17_qfile or logs_dir),
        "text": "\n".join([x for x in [l17_qtail, l17_ctail, "\n".join(l17_ev)] if x]) or "(暂无日志输出)",
    }

print("__FGPO_JSON_START__")
print(json.dumps(data, ensure_ascii=False))
print("__FGPO_JSON_END__")
'''

HOST_B_PROBE_BODY = r'''
DEPLOY_HINT = os.environ.get("FGPO_DEPLOY_DIR", "auto")
CELL_SPECS = json.loads(os.environ["FGPO_CELL_SPECS"])

procs = collect_fgpo_processes("")

deploy_dir = None
if DEPLOY_HINT and DEPLOY_HINT != "auto" and os.path.isdir(DEPLOY_HINT):
    deploy_dir = pathlib.Path(DEPLOY_HINT)
else:
    for p in procs:
        out_arg = p.get("target_output_dir") or ""
        if out_arg:
            cand = pathlib.Path(out_arg)
            if cand.name == "output" and cand.parent.is_dir():
                deploy_dir = cand.parent
                break
    if deploy_dir is None:
        home_ws = str(pathlib.Path.home() / "workspace" / "fgpo-v4-formal-host-b-*")
        cands = sorted(glob.glob(home_ws) + glob.glob("/home/*/workspace/fgpo-v4-formal-host-b-*"), key=lambda x: os.path.getmtime(x), reverse=True)
        if cands:
            deploy_dir = pathlib.Path(cands[0])

all_gpus = collect_gpus()
annotate_gpu_fgpo_vram(all_gpus, procs)
usable_gpus = []
excluded_gpus = []
for g in all_gpus:
    idx = g["index"]
    if idx == 0:
        g["role_slot"] = "Queue B0 (GPU 0 · 80 GiB cap)"
        g["active_in_experiment"] = True
        g["mem_cap_gib"] = 80
        g["allocator_cap_gib"] = 79
        usable_gpus.append(g)
    elif idx == 2:
        g["role_slot"] = "Queue B2 (GPU 2 · 80 GiB cap)"
        g["active_in_experiment"] = True
        g["mem_cap_gib"] = 80
        g["allocator_cap_gib"] = 79
        usable_gpus.append(g)
    else:
        g["role_slot"] = "Unassigned (Excluded)"
        g["active_in_experiment"] = False
        excluded_gpus.append(g)

numa_nodes = {}
for np in sorted(glob.glob("/sys/devices/system/node/node*/cpulist")):
    try:
        numa_nodes[pathlib.Path(np).parent.name] = pathlib.Path(np).read_text().strip()
    except Exception:
        pass

handoff_sup = next((p for p in procs if p.get("role") == "handoff_supervisor"), None)
handoff_wait = next((p for p in procs if p.get("role") == "handoff_waiter"), None)
gpu2_runner = next((p for p in procs if p.get("lane") == "GPU 2" and "runner" in (p.get("role") or "")), None)
handoff_pending = bool(handoff_sup and handoff_wait and gpu2_runner and gpu2_runner.get("target_method") == "fgpo")

data = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "deployment_dir": str(deploy_dir) if deploy_dir else None,
    "hostname": sh("hostname"),
    "uptime": sh("uptime"),
    "cpu_count": os.cpu_count(),
    "numa_nodes": numa_nodes,
    "loadavg": list(os.getloadavg()),
    "mem": collect_mem_info(),
    "disks": collect_disks(["/", "/home"]),
    "gpus": usable_gpus,
    "excluded_gpus": excluded_gpus,
    "proc_list": procs,
    "gpu2_handoff": {
        "handoff_pending": handoff_pending,
        "handoff_supervisor_pid": handoff_sup.get("pid") if handoff_sup else None,
        "handoff_waiter_pid": handoff_wait.get("pid") if handoff_wait else None,
    },
    "cells": {},
    "log_tails": {},
    "config": {"protocol": "fgpo-r3-published-v4", "groups": 16, "trajectories": 8},
}

if deploy_dir and deploy_dir.is_dir():
    cfg_cand = deploy_dir / "deploy" / "configs" / "host-b-formal.json"
    if cfg_cand.is_file():
        try:
            cjson = json.loads(cfg_cand.read_text())
            data["config"]["protocol"] = cjson.get("protocol", "fgpo-r3-published-v4")
            data["config"]["groups"] = cjson.get("groups") or 16
            data["config"]["trajectories"] = cjson.get("trajectories") or 8
        except Exception:
            pass
    out_dir = deploy_dir / "output"
    for spec in CELL_SPECS:
        ck = f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
        study_root = deploy_dir / spec["output_subdir"] if spec.get("output_subdir") else out_dir
        cdir = study_root / ck
        data["cells"][ck] = read_cell_dir(cdir, spec["official_eval"].split("+"), include_du=False)

    launch_dir = out_dir / "launch"
    old_cell = data["cells"].get("comparison/appworld/fgpo/seed-1") or {}
    old_snapshot = (old_cell.get("checkpoint_current") or {}).get("snapshot")
    old_lane_log = launch_dir / "gpu2.lane.log"
    prior_exit = False
    if old_lane_log.is_file():
        prior_exit = any("exit physical_gpu=2 method=fgpo seed=1 status=130" in row
                         for row in old_lane_log.read_text(errors="replace").splitlines())
    if old_snapshot and prior_exit:
        runner = gpu2_runner or {}
        data["gpu2_handoff"].update(
            fgpo_saved_snapshot=old_snapshot,
            fgpo_resume_pid=runner.get("pid") if (runner.get("target_method") == "fgpo"
                and runner.get("target_seeds") == [1] and runner.get("target_resume")) else None,
        )
    fgpo_summary = old_cell.get("cell_summary") or {}
    if fgpo_summary.get("status") == "complete":
        g2po_cell = data["cells"].get("comparison/appworld/g2po/seed-0") or {}
        runner = gpu2_runner or {}
        data["gpu2_handoff"].update(
            g2po_runner_pid=runner.get("pid") if (runner.get("target_method") == "g2po"
                and runner.get("target_seeds") == [0]) else None,
            first_successful_update=(g2po_cell.get("costs_summary") or {}).get("first_successful_adamw"),
        )
    for lane_tag, prefix, gpu_id in [("Host B · GPU 0", "gpu0", "0"), ("Host B · GPU 2", "gpu2", "2")]:
        matches = sorted(launch_dir.glob(f"{prefix}*.log"), key=lambda p: p.stat().st_mtime, reverse=True) if launch_dir.is_dir() else []
        lane_specs = [spec for spec in CELL_SPECS if spec.get("physical_gpu_ids") == gpu_id]
        current_logs = [deploy_dir / spec["output_subdir"] / "launch" / name
                        for spec in lane_specs if spec.get("output_subdir")
                        for name in (f"{spec['benchmark']}-{spec['method']}-seed-{spec['seed']}.log",
                                     f"{spec['benchmark']}-seed-{spec['seed']}.log" if spec["method"] == "fgpo"
                                     else f"{spec['benchmark']}-gpu-{gpu_id}.queue.log")]
        if current_logs:
            matches = sorted((p for p in current_logs if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
        lpath = matches[0] if matches else None
        ltail = sh(f"tail -n 20 '{lpath}' 2>/dev/null || true") if lpath else ""
        extra_lines = []
        if gpu_id == "2" and handoff_pending:
            extra_lines.append(
                f"[launch_gpu2_v4.sh] pid={handoff_sup.get('pid')} -> waiter={handoff_wait.get('pid')} awaiting fgpo/seed-1 exit=130 for fgpo/seed-1 --resume; g2po/seed-0 follows F-GPO completion"
            )
        active_ck = None
        lane_runner = next((p for p in procs if p.get("lane") == "GPU " + gpu_id and
                            "runner" in (p.get("role") or "")), None)
        if lane_runner and lane_runner.get("target_seeds"):
            ts = lane_runner["target_seeds"][0]
            active_ck = f"{lane_runner.get('target_suite', 'comparison')}/{lane_runner.get('target_benchmark')}/{lane_runner.get('target_method')}/seed-{ts}"
        for spec in lane_specs:
            ck = f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
            if active_ck and ck != active_ck:
                continue
            cd = data["cells"].get(ck) or {}
            if cd.get("exists") and not cd.get("cell_summary"):
                for ev in (cd.get("recent_cost_events") or [])[-10:]:
                    extra_lines.append(
                        f"[costs #{ev.get('event_index')} t={ev.get('elapsed')}s] role={ev.get('role')} sec={ev.get('seconds')} steps={ev.get('env_steps') or 0} group={ev.get('group_id') or '-'}"
                    )
                break
        combined = "\n".join([x for x in [ltail, "\n".join(extra_lines)] if x])
        data["log_tails"][lane_tag] = {
            "path": str(lpath or launch_dir),
            "text": combined or "(暂无日志输出)",
        }

print("__FGPO_JSON_START__")
print(json.dumps(data, ensure_ascii=False))
print("__FGPO_JSON_END__")
'''


# ---------------------------------------------------------------------------
# Security, SSH Document Parsing, Schedule Loading, and PTY SSH Execution
# ---------------------------------------------------------------------------

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


def _parse_ssh_doc(ssh_doc_path: Path) -> dict[str, dict[str, Any]]:
    """Parse Host A & Host B SSH connection details in memory without logging."""
    creds: dict[str, dict[str, Any]] = {}
    if not ssh_doc_path.is_file():
        return creds
    text = ssh_doc_path.read_text(encoding="utf-8", errors="replace")
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_pw = re.match(r"SSH\s*Password\s*:\s*(\S+)", line, re.IGNORECASE)
        if m_pw:
            if current_key is not None:
                creds[current_key]["password"] = m_pw.group(1)
            continue
        if line.lower().startswith("ssh ") and "@" in line:
            if current_key is not None:
                m_target = re.search(r"([^\s@]+)@([^\s:]+)", line)
                m_port = re.search(r"-p\s+(\d+)", line)
                if m_target:
                    creds[current_key]["user"] = m_target.group(1)
                    creds[current_key]["host"] = m_target.group(2)
                if m_port:
                    creds[current_key]["port"] = int(m_port.group(1))
            continue
        m_host = re.search(r"\bHost\s+([AB])\b", line, re.IGNORECASE)
        if m_host:
            current_key = f"host_{m_host.group(1).lower()}"
            creds.setdefault(current_key, {"port": 22})
            continue
    return {k: v for k, v in creds.items() if v.get("user") and v.get("host")}


def _build_secret_replacements(ssh_creds: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for srv_key, info in ssh_creds.items():
        pw = str(info.get("password") or "")
        host = str(info.get("host") or "")
        user = str(info.get("user") or "")
        port = str(info.get("port") or "")
        if pw:
            pairs.append((pw, "[REDACTED]"))
        if user and host:
            pairs.append((f"{user}@{host}", f"<ssh-user>@<{srv_key}>"))
        if host and port:
            pairs.append((f"{host}:{port}", f"<{srv_key}>:<port>"))
        if host:
            pairs.append((host, f"<{srv_key}>"))
        if port and port != "22":
            pairs.append((f"-p {port}", "-p <port>"))
            pairs.append((f"-p{port}", "-p<port>"))
            pairs.append((f"port {port}", "port <port>"))
            pairs.append((f"Port {port}", "Port <port>"))
            pairs.append((f"port={port}", "port=<port>"))
        if user:
            pairs.append((f"/users/{user}/", "/users/<user>/"))
            pairs.append((f"/home/{user}/", "/home/<user>/"))
            pairs.append((user, "<user>"))
    pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
    return pairs


_TOKEN_PATTERNS = [
    (re.compile(r"ya29\.[A-Za-z0-9._\-]+"), "[REDACTED_OAUTH_TOKEN]"),
    (re.compile(r"(colab-runtime-proxy-token\s*[=:]\s*)[A-Za-z0-9._\-]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(X-Colab-Runtime-Proxy-Token\s*:\s*)[A-Za-z0-9._\-]+", re.I), r"\1[REDACTED]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{16,}", re.I), r"\1[REDACTED]"),
    (re.compile(r"(https?://[A-Za-z0-9.\-]*colab\.googleusercontent\.com[^\s\"']*)"), "[REDACTED_COLAB_PROXY_URL]"),
]


def _sanitize_secrets(obj: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(obj, str):
        s = obj
        for needle, repl in replacements:
            if needle and needle in s:
                s = s.replace(needle, repl)
        for rx, repl in _TOKEN_PATTERNS:
            s = rx.sub(repl, s)
        return s
    if isinstance(obj, list):
        return [_sanitize_secrets(x, replacements) for x in obj]
    if isinstance(obj, dict):
        return {_sanitize_secrets(k, replacements): _sanitize_secrets(v, replacements) for k, v in obj.items()}
    return obj


def _map_row_to_lane(host_or_card: str, physical_gpu_ids: str) -> tuple[str, str, str]:
    hc = host_or_card.strip().lower()
    if hc in ("colab", "pro6000"):
        return "colab", "colab_gpu_0", "Colab · PRO 6000 (GPU 0)"
    if hc in ("host a", "host_a", "host_a_l40"):
        if physical_gpu_ids == "3+4":
            return "host_a", "host_a_l40_3_4", "Host A · L40 3+4 (Queue A1)"
        return "host_a", "host_a_l40_1_7", "Host A · L40 1+7 (Queue A2)"
    if physical_gpu_ids == "0":
        return "host_b", "host_b_gpu_0", "Host B · GPU 0 (Queue B0)"
    return "host_b", "host_b_gpu_2", "Host B · GPU 2 (Queue B2)"


def _default_endpoint_tasks(benchmark: str) -> int:
    if benchmark == "alfworld":
        return 134
    if benchmark == "webshop":
        return 128
    return 585 if benchmark == "appworld" else None


def _load_schedule_18(csv_path: Path) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    if csv_path.is_file():
        try:
            with csv_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    bm = row["benchmark"].strip()
                    tot_r = int(row.get("rounds") or row.get("total_rounds") or 60)
                    raw_rows.append({
                        "host_or_card": (row.get("host") or row.get("host_or_card") or "").strip(),
                        "physical_gpu_ids": row["physical_gpu_ids"].strip(),
                        "device_count": int(row.get("device_count") or row.get("physical_gpu_count") or 1),
                        "lane_order": int(row["lane_order"]) if row.get("lane_order") not in (None, "") else None,
                        "suite": row["suite"].strip(),
                        "benchmark": bm,
                        "method": row["method"].strip(),
                        "seed": int(row["seed"]),
                        "total_rounds": tot_r,
                        "total_training_trajectories": int(row.get("training_trajectories") or row.get("total_training_trajectories") or (tot_r * 128)),
                        "horizon": int(row["horizon"]) if row.get("horizon") not in (None, "") else (50 if bm == "alfworld" else (15 if bm == "webshop" else 100)),
                        "official_eval": (row.get("official_eval") or ("valid_unseen" if bm == "alfworld" else ("graphgpo_released_validation" if bm == "webshop" else "test_normal+test_challenge"))).strip(),
                        "official_endpoint_tasks": int(row["official_endpoint_tasks"]) if row.get("official_endpoint_tasks") not in (None, "") else _default_endpoint_tasks(bm),
                        "status_snapshot": (row.get("status_snapshot") or "").strip(),
                        "sampling_layout": (row.get("sampling_layout") or "").strip(),
                        "dependency": (row.get("dependency") or "").strip(),
                        "output_subdir": (row.get("output_subdir") or "").strip(),
                    })
        except Exception:
            raw_rows = []
    if not raw_rows:
        raise ValueError(f"Experiment schedule has no readable rows: {csv_path}")

    lane_counters: dict[str, int] = {}
    enriched: list[dict[str, Any]] = []
    for idx, r in enumerate(raw_rows, start=1):
        srv_id, lane_id, lane_title = _map_row_to_lane(r["host_or_card"], r["physical_gpu_ids"])
        lo = r["lane_order"]
        if lo is None:
            lo = lane_counters.get(lane_id, 1)
        lane_counters[lane_id] = max(lane_counters.get(lane_id, 0), lo + 1)
        ck = f"{r['suite']}/{r['benchmark']}/{r['method']}/seed-{r['seed']}"
        item = dict(r)
        item["lane_order"] = lo
        item["physical_gpu_count"] = item["device_count"]
        item["index"] = idx
        item["server_id"] = srv_id
        item["lane_id"] = lane_id
        item["lane_title"] = lane_title
        item["cell_key"] = ck
        item["global_cell_id"] = f"{srv_id}:{ck}"
        enriched.append(item)
    return enriched


_load_schedule_21 = _load_schedule_18


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


def _run_ssh_pty_python(
    conn: dict[str, Any],
    script: str,
    timeout: int,
    batch_cache: dict[str, bool],
    server_id: str,
) -> tuple[str, str]:
    """Run `python3 -` read-only over SSH using BatchMode first, then in-memory PTY password auth."""
    user = conn["user"]
    host = conn["host"]
    port = str(conn.get("port") or 22)
    password = conn.get("password") or ""

    ssh_common = [
        "ssh",
        "-T",
        "-o", "ConnectTimeout=12",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", port,
    ]

    if batch_cache.get(server_id) is not False:
        batch_cmd = [*ssh_common, "-o", "BatchMode=yes", f"{user}@{host}", "python3 -"]
        try:
            proc_b = subprocess.run(
                batch_cmd,
                input=script,
                text=True,
                capture_output=True,
                timeout=min(timeout, 25),
            )
            if proc_b.returncode == 0 and "__FGPO_JSON_START__" in proc_b.stdout:
                batch_cache[server_id] = True
                return proc_b.stdout, "ssh-key"
            if not password:
                msg = (proc_b.stderr or proc_b.stdout or f"ssh exit {proc_b.returncode}").strip()
                raise RuntimeError(f"SSH BatchMode failed: {msg}")
            batch_cache[server_id] = False
        except subprocess.TimeoutExpired:
            if not password:
                raise RuntimeError("SSH BatchMode timed out")

    if not password:
        raise RuntimeError(f"Missing SSH password for {server_id} in SSH_Document.md")

    ssh_pw_cmd = [
        *ssh_common,
        "-o", "BatchMode=no",
        "-o", "NumberOfPasswordPrompts=1",
        "-o", "PreferredAuthentications=keyboard-interactive,password,publickey",
        f"{user}@{host}",
        "python3 -",
    ]

    clean_env = os.environ.copy()
    clean_env.pop("SSH_ASKPASS", None)
    clean_env.pop("SSH_ASKPASS_REQUIRE", None)
    clean_env.pop("DISPLAY", None)

    master_fd, slave_fd = pty.openpty()
    slave_closed = False

    def _attach_ctty() -> None:
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.close(slave_fd)

    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            ssh_pw_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(slave_fd,),
            preexec_fn=_attach_ctty,
            env=clean_env,
        )
        os.close(slave_fd)
        slave_closed = True

        out_chunks: list[bytes] = []
        err_chunks: list[bytes] = []

        def _feed_stdin() -> None:
            assert proc is not None and proc.stdin is not None
            try:
                proc.stdin.write(script.encode("utf-8"))
                proc.stdin.flush()
            except OSError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        def _drain(pipe: Any, sink: list[bytes]) -> None:
            try:
                data = pipe.read()
                if data:
                    sink.append(data)
            except OSError:
                pass
            finally:
                try:
                    pipe.close()
                except OSError:
                    pass

        t_in = threading.Thread(target=_feed_stdin, daemon=True)
        t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
        t_in.start()
        t_out.start()
        t_err.start()

        deadline = time.monotonic() + timeout
        pty_buf = b""
        pw_sent = False
        master_eof = False

        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                proc.kill()
                raise RuntimeError(f"SSH probe timed out after {timeout}s")
            if not master_eof:
                rlist, _, _ = select.select([master_fd], [], [], min(0.2, rem))
                if rlist:
                    try:
                        chunk = os.read(master_fd, 1024)
                        if not chunk:
                            master_eof = True
                        else:
                            pty_buf = (pty_buf + chunk)[-4096:]
                            if (not pw_sent) and (b"assword:" in pty_buf or b"asscode:" in pty_buf):
                                os.write(master_fd, (password + "\n").encode("utf-8"))
                                pw_sent = True
                                pty_buf = b""
                    except OSError:
                        master_eof = True
            else:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)

            if proc.poll() is not None and master_eof:
                break

        t_in.join(timeout=2.0)
        t_out.join(timeout=5.0)
        t_err.join(timeout=5.0)
        rc = proc.wait(timeout=2.0)
        stdout_text = b"".join(out_chunks).decode("utf-8", errors="replace")
        stderr_text = b"".join(err_chunks).decode("utf-8", errors="replace")
        if rc != 0:
            pty_msg = pty_buf.decode("utf-8", errors="replace").strip()
            msg = (stderr_text or pty_msg or stdout_text or f"ssh exit {rc}").strip()
            raise RuntimeError(f"SSH exit {rc}: {msg}")
        return stdout_text, "ssh-pty"
    finally:
        if not slave_closed:
            try:
                os.close(slave_fd)
            except OSError:
                pass
        try:
            os.close(master_fd)
        except OSError:
            pass
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Unified Multi-Server ClusterPoller
# ---------------------------------------------------------------------------

class ClusterPoller:
    """Thread-safe poller that queries Colab, Host A, and Host B concurrently."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = Path(__file__).resolve().parents[1]
        self.colab_bin = os.path.expanduser(args.colab_bin)
        self.env = _build_env()
        self.ssh_doc_path = Path(os.path.expanduser(args.ssh_doc)).resolve()
        self.schedule_csv_path = Path(os.path.expanduser(args.schedule_csv)).resolve()
        self.schedule_18 = _load_schedule_18(self.schedule_csv_path)

        self.ssh_creds = _parse_ssh_doc(self.ssh_doc_path)
        self.secret_replacements = _build_secret_replacements(self.ssh_creds)
        self._batch_mode_cache: dict[str, bool] = {}

        self.enabled: dict[str, bool] = {
            "colab": not getattr(args, "disable_colab", False),
            "host_a": not getattr(args, "disable_host_a", False),
            "host_b": not getattr(args, "disable_host_b", False),
        }

        self.server_titles: dict[str, str] = {
            "colab": f"Colab · RTX PRO 6000 (Session {args.session})",
            "host_a": "Host A · mgt01 (6× NVIDIA L40)",
            "host_b": "Host B · Dual RTX 6000D (GPUs 0 & 2)",
        }

        self.state_locks: dict[str, threading.Lock] = {k: threading.Lock() for k in ("colab", "host_a", "host_b")}
        self.poll_locks: dict[str, threading.Lock] = {k: threading.Lock() for k in ("colab", "host_a", "host_b")}
        self.histories: dict[str, deque[dict[str, Any]]] = {k: deque(maxlen=120) for k in ("colab", "host_a", "host_b")}
        self.last_colab_meta_ts = 0.0

        self.server_states: dict[str, dict[str, Any]] = {}
        for sid in ("colab", "host_a", "host_b"):
            self.server_states[sid] = {
                "server_id": sid,
                "display_name": self.server_titles[sid],
                "enabled": self.enabled[sid],
                "status": "initializing" if self.enabled[sid] else "disabled",
                "polling": False,
                "last_ok_at": None,
                "last_attempt_at": None,
                "latency_ms": None,
                "last_error": None,
                "transport": None,
                "colab": {} if sid == "colab" else None,
                "telemetry": None,
                "history": [],
            }

        self.meta = {
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
            "host_a_dir": args.host_a_dir,
            "host_b_dir": args.host_b_dir,
        }

    def _reload_ssh_creds_if_needed(self) -> None:
        fresh = _parse_ssh_doc(self.ssh_doc_path)
        if fresh:
            self.ssh_creds = fresh
            self.secret_replacements = _build_secret_replacements(fresh)

    def _load_local_valid_seen(self) -> dict[int, dict[str, Any]]:
        """Read local ALFWorld valid_seen replay summaries cleanly on snapshot."""
        out: dict[int, dict[str, Any]] = {}
        vs_root = self.repo_root / "outputs" / "exp-plan-20261118" / "alfworld-valid-seen"
        if not vs_root.is_dir():
            return out
        for sp in sorted(vs_root.glob("seed-*/evaluation_summary.json")):
            try:
                raw = json.loads(sp.read_text(encoding="utf-8"))
                seed_i = int(raw.get("seed"))
                by_t = raw.get("by_task_type") or {}
                all_blk = by_t.get("All") or {}
                sr_val = all_blk.get("SR")
                out[seed_i] = _sanitize_secrets({
                    "split": "valid_seen",
                    "status": raw.get("status", "complete"),
                    "model_version": raw.get("model_version"),
                    "evaluated_count": raw.get("evaluated_episode_count") or all_blk.get("episode_count"),
                    "unique_task_count": raw.get("unique_task_count"),
                    "success_count": all_blk.get("success_count"),
                    "SR": round(float(sr_val), 2) if sr_val is not None else None,
                    "wall_seconds": raw.get("wall_seconds"),
                    "by_task_type": by_t,
                }, self.secret_replacements)
            except Exception:
                pass
        return out

    def _load_archived_colab_cells(self) -> dict[str, dict[str, Any]]:
        archive = (self.repo_root / "outputs" / "fgpo-colab-956ceab-ag-20261116"
                   / "gdrive_staging")
        cells = {}
        for scheduled in self.schedule_18:
            if scheduled["server_id"] != "colab":
                continue
            key = scheduled["cell_key"]
            directory = archive / key
            try:
                summary = json.loads((directory / "cell_summary.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if summary.get("status") != "complete" or summary.get("finalization_stage") != "finalized":
                continue
            try:
                rounds = [json.loads(line) for line in
                          (directory / "training.jsonl").read_text(encoding="utf-8").splitlines()
                          if line.strip()]
            except (OSError, ValueError):
                rounds = []
            cells[key] = _sanitize_secrets({
                "exists": True,
                "cell_summary": summary,
                "training_rounds": rounds,
                "evaluations": {
                    split: {"summary": result, "completed_count": result.get("evaluated_count"),
                            "success_count": result.get("success_count")}
                    for split, result in (summary.get("evaluation") or {}).items()
                },
            }, self.secret_replacements)
        return cells

    def _extract_probe_json(self, raw_out: str) -> dict[str, Any]:
        m = re.search(r"__FGPO_JSON_START__\s*(.*?)\s*__FGPO_JSON_END__", raw_out, re.S)
        if not m:
            raise RuntimeError("Missing JSON marker from remote probe output")
        parsed = json.loads(m.group(1))
        return _sanitize_secrets(parsed, self.secret_replacements)

    # ---- Colab transport ----
    def _run_colab_cli(self, cli_args: list[str], *, stdin_text: str | None = None, timeout: int = 35) -> str:
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

    def _run_colab_ssh(self, script: str, timeout: int) -> str:
        proxy_cmd = f"{self.colab_bin} ssh --proxy-mode -s {self.args.session}"
        proc = subprocess.run(
            [
                "ssh", "-T",
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
            raise RuntimeError(f"colab ssh proxy-mode: {msg}")
        return proc.stdout

    def _run_colab_exec(self, script: str, timeout: int) -> str:
        common_sh = Path(__file__).resolve().with_name("colab_common.sh")
        env = dict(self.env)
        env["COLAB_CLI"] = self.colab_bin
        env["COLAB_EXEC_TIMEOUT"] = str(timeout)
        proc = subprocess.run(
            ["bash", "-c", 'source "$1" && colab_exec_run "$2" "python3 -"', "colab_exec_bridge", str(common_sh), self.args.session],
            input=script,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout + 15,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise RuntimeError(f"colab exec: {msg}")
        return proc.stdout

    def _poll_colab_once(self, *, force_meta: bool = False) -> None:
        sid = "colab"
        if not self.enabled[sid]:
            return
        if not self.poll_locks[sid].acquire(blocking=False):
            return
        t0 = time.monotonic()
        try:
            with self.state_locks[sid]:
                self.server_states[sid]["polling"] = True
                self.server_states[sid]["last_attempt_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            now = time.time()
            colab_info = self.server_states[sid].get("colab") or {}
            if force_meta or not colab_info or (now - self.last_colab_meta_ts) >= 300:
                try:
                    sess_raw = self._run_colab_cli(["sessions"], timeout=30)
                    usage_raw = self._run_colab_cli(["usage"], timeout=30)
                    colab_info = _sanitize_secrets({
                        "sessions": _parse_colab_sessions(sess_raw, self.args.session),
                        "usage": _parse_colab_usage(usage_raw),
                    }, self.secret_replacements)
                    self.last_colab_meta_ts = now
                    with self.state_locks[sid]:
                        self.server_states[sid]["colab"] = colab_info
                except Exception as meta_err:
                    if not colab_info:
                        raise meta_err

            sessions_info = colab_info.get("sessions") or {"active": True}
            colab_specs = [
                {"suite": c["suite"], "benchmark": c["benchmark"], "method": c["method"], "seed": c["seed"]}
                for c in self.schedule_18 if c["server_id"] == "colab"
            ]
            preamble = (
                "import os\n"
                f"os.environ['FGPO_CFG_PATH'] = {self.args.config_path!r}\n"
                f"os.environ['FGPO_OUT_DIR'] = {self.args.output_dir!r}\n"
                f"os.environ['FGPO_LOG_PATH'] = {self.args.log_path!r}\n"
                f"os.environ['FGPO_SUITE'] = {self.args.suite!r}\n"
                f"os.environ['FGPO_BENCHMARK'] = {self.args.benchmark!r}\n"
                f"os.environ['FGPO_METHOD'] = {self.args.method!r}\n"
                f"os.environ['FGPO_SEEDS'] = {self.args.seeds!r}\n"
                f"os.environ['FGPO_CELL_SPECS'] = {json.dumps(colab_specs)!r}\n"
            )
            full_script = preamble + REMOTE_COMMON_HELPERS + COLAB_PROBE_BODY

            telemetry = None
            used_transport = self.args.transport
            if sessions_info.get("active", True):
                if self.args.transport == "ssh":
                    raw_out = self._run_colab_ssh(full_script, self.args.timeout)
                elif self.args.transport == "exec":
                    raw_out = self._run_colab_exec(full_script, self.args.timeout)
                else:
                    raise RuntimeError(f"Unsupported Colab transport: {self.args.transport!r}")
                telemetry = self._extract_probe_json(raw_out)
                self._append_server_history(sid, telemetry)

            latency_ms = int(round((time.monotonic() - t0) * 1000))
            with self.state_locks[sid]:
                st = self.server_states[sid]
                st["status"] = "online" if sessions_info.get("active", True) else "session_inactive"
                st["polling"] = False
                st["last_ok_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                st["latency_ms"] = latency_ms
                st["last_error"] = None
                st["transport"] = used_transport
                st["colab"] = colab_info
                if telemetry is not None:
                    st["telemetry"] = telemetry
                st["history"] = list(self.histories[sid])
        except Exception as exc:
            err_clean = _sanitize_secrets(str(exc), self.secret_replacements)
            with self.state_locks[sid]:
                st = self.server_states[sid]
                st["polling"] = False
                st["latency_ms"] = int(round((time.monotonic() - t0) * 1000))
                st["status"] = "stale" if st.get("telemetry") else "error"
                st["last_error"] = err_clean
        finally:
            self.poll_locks[sid].release()

    def _poll_ssh_server_once(self, sid: str) -> None:
        if not self.enabled[sid]:
            return
        if not self.poll_locks[sid].acquire(blocking=False):
            return
        t0 = time.monotonic()
        try:
            with self.state_locks[sid]:
                self.server_states[sid]["polling"] = True
                self.server_states[sid]["last_attempt_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            self._reload_ssh_creds_if_needed()
            conn = self.ssh_creds.get(sid)
            if not conn:
                raise RuntimeError(f"Connection entry for {sid} not found in {self.ssh_doc_path.name}")

            specs = [
                {
                    "suite": c["suite"],
                    "benchmark": c["benchmark"],
                    "method": c["method"],
                    "seed": c["seed"],
                    "physical_gpu_ids": c["physical_gpu_ids"],
                    "official_eval": c["official_eval"],
                    "output_subdir": c.get("output_subdir"),
                }
                for c in self.schedule_18 if c["server_id"] == sid
            ]
            deploy_hint = self.args.host_a_dir if sid == "host_a" else self.args.host_b_dir
            body = HOST_A_PROBE_BODY if sid == "host_a" else HOST_B_PROBE_BODY
            preamble = (
                "import os\n"
                f"os.environ['FGPO_DEPLOY_DIR'] = {deploy_hint!r}\n"
                f"os.environ['FGPO_CELL_SPECS'] = {json.dumps(specs)!r}\n"
            )
            raw_out, used_transport = _run_ssh_pty_python(
                conn,
                preamble + REMOTE_COMMON_HELPERS + body,
                self.args.timeout,
                self._batch_mode_cache,
                sid,
            )
            telemetry = self._extract_probe_json(raw_out)
            self._append_server_history(sid, telemetry)

            latency_ms = int(round((time.monotonic() - t0) * 1000))
            with self.state_locks[sid]:
                st = self.server_states[sid]
                st["status"] = "online"
                st["polling"] = False
                st["last_ok_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                st["latency_ms"] = latency_ms
                st["last_error"] = None
                st["transport"] = used_transport
                st["telemetry"] = telemetry
                st["history"] = list(self.histories[sid])
        except Exception as exc:
            err_clean = _sanitize_secrets(str(exc), self.secret_replacements)
            with self.state_locks[sid]:
                st = self.server_states[sid]
                st["polling"] = False
                st["latency_ms"] = int(round((time.monotonic() - t0) * 1000))
                st["status"] = "stale" if st.get("telemetry") else "error"
                st["last_error"] = err_clean
        finally:
            self.poll_locks[sid].release()

    def _append_server_history(self, sid: str, telemetry: dict[str, Any]) -> None:
        gpus = [g for g in (telemetry.get("gpus") or []) if g.get("active_in_experiment")] or (telemetry.get("gpus") or [])
        if gpus:
            avg_util = round(sum(float(g.get("util_gpu_pct") or 0.0) for g in gpus) / len(gpus), 1)
            used_mib = sum(float(g.get("mem_used_mib") or 0.0) for g in gpus)
            tot_mib = sum(float(g.get("mem_total_mib") or 0.0) for g in gpus)
            vram_pct = round(100.0 * used_mib / tot_mib, 1) if tot_mib > 0 else 0.0
            max_temp = max((g.get("temp_c") or 0) for g in gpus)
            sum_power = round(sum(float(g.get("power_draw_w") or 0.0) for g in gpus), 1)
        else:
            avg_util, used_mib, tot_mib, vram_pct, max_temp, sum_power = None, 0.0, 0.0, None, None, None

        cells = telemetry.get("cells") or {}
        env_steps_sum = 0
        out_tok_sum = 0
        ev_sum = 0
        for cd in cells.values():
            cs = (cd or {}).get("costs_summary") or {}
            env_steps_sum += int(cs.get("env_steps") or 0)
            out_tok_sum += int(cs.get("output_tokens") or 0)
            ev_sum += int(cs.get("total_events") or 0)

        self.histories[sid].append({
            "ts": time.strftime("%H:%M:%S"),
            "gpu_util": avg_util,
            "vram_pct": vram_pct,
            "vram_used_gib": round(used_mib / 1024.0, 2),
            "vram_total_gib": round(tot_mib / 1024.0, 2),
            "power_w": sum_power,
            "temp_c": max_temp,
            "env_steps": env_steps_sum,
            "output_tokens": out_tok_sum,
            "total_events": ev_sum,
        })

    def poll_once(self, *, target_server: str = "all", force_meta: bool = False) -> dict[str, Any]:
        targets = ["colab", "host_a", "host_b"] if target_server == "all" else [target_server]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = []
            for sid in targets:
                if sid == "colab" and self.enabled.get("colab"):
                    futs.append(pool.submit(self._poll_colab_once, force_meta=force_meta))
                elif sid in ("host_a", "host_b") and self.enabled.get(sid):
                    futs.append(pool.submit(self._poll_ssh_server_once, sid))
            for f in futs:
                f.result()
        return self.get_snapshot()

    def _summarize_cell_live(
        self,
        sched: dict[str, Any],
        raw_cell: dict[str, Any] | None,
        *,
        active_runner_ck: str | None,
        lane_has_runner: bool,
        lane_has_queue: bool,
        handoff_pending_on_lane: bool,
        valid_seen_map: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        merged = dict(sched)
        cd = raw_cell or {}
        exists = bool(cd.get("exists"))
        merged["exists"] = exists
        rounds = cd.get("training_rounds") or []
        rounds_done = len(rounds)
        csum = cd.get("costs_summary") or {}
        g_done = max(int(csum.get("current_round_groups_done") or 0), int(cd.get("current_round_group_files") or 0))
        cell_sum = cd.get("cell_summary") or {}
        evals = dict(cd.get("evaluations") or {})

        # Attach local ALFWorld valid_seen replay summary onto Colab fgpo seeds
        eval_valid_seen_sr_pct = None
        if sched["server_id"] == "colab" and sched["suite"] == "comparison" and sched["benchmark"] == "alfworld" and sched["method"] == "fgpo":
            vs_entry = valid_seen_map.get(int(sched["seed"]))
            if vs_entry:
                evals["valid_seen"] = {"summary": vs_entry, "completed_count": vs_entry.get("evaluated_count"), "success_count": vs_entry.get("success_count")}
                eval_valid_seen_sr_pct = vs_entry.get("SR")

        is_complete = cell_sum.get("status") == "complete" or bool(cell_sum.get("evaluation"))
        eval_started = any((v.get("completed_count") or 0) > 0 for k, v in evals.items() if k != "valid_seen")
        has_progress = exists and (rounds_done > 0 or int(csum.get("total_events") or 0) > 0 or eval_started)

        if is_complete:
            live_status = "completed"
        elif lane_has_runner and sched["cell_key"] == active_runner_ck:
            live_status = "evaluating" if eval_started else "running"
        elif has_progress:
            live_status = "paused"
        else:
            live_status = "queued" if lane_has_queue else "planned"

        handoff_pending = bool(handoff_pending_on_lane and live_status == "running")
        staged_handoff = bool(handoff_pending_on_lane and live_status == "queued" and sched["lane_order"] == 1)

        last_r = rounds[-1] if rounds else None
        last_m = (last_r or {}).get("metrics") or {}
        latest_sr = None
        latest_succ = None
        latest_traj = None
        if last_r and last_r.get("trajectories"):
            latest_succ = int(last_r.get("successes") or 0)
            latest_traj = int(last_r.get("trajectories") or 128)
            latest_sr = round(100.0 * latest_succ / latest_traj, 2)

        eval_sr_pct = None
        eval_metric_name = "SR"
        eval_sgc_pct = None
        eval_score_pct = None
        if cell_sum.get("evaluation"):
            ev_map = cell_sum["evaluation"]
            if "valid_unseen" in ev_map and ev_map["valid_unseen"].get("SR") is not None:
                eval_sr_pct = round(float(ev_map["valid_unseen"]["SR"]), 2)
                eval_metric_name = "SR"
            elif "test_normal" in ev_map:
                tn = ev_map["test_normal"] or {}
                if tn.get("SR") is not None:
                    eval_sr_pct = round(float(tn["SR"]), 2)
                    eval_metric_name = "SR"
                elif tn.get("TGC") is not None:
                    eval_sr_pct = round(float(tn["TGC"]), 2)
                    eval_metric_name = "TGC"
                if tn.get("SGC") is not None:
                    eval_sgc_pct = round(float(tn["SGC"]), 2)
            else:
                for sp_name in ("test", "valid", "graphgpo_released_validation"):
                    if sp_name in ev_map and isinstance(ev_map[sp_name], dict):
                        wb = ev_map[sp_name]
                        if wb.get("SR") is not None:
                            eval_sr_pct = round(float(wb["SR"]), 2)
                            eval_metric_name = "SR"
                        sc_v = wb.get("Score") if wb.get("Score") is not None else wb.get("score")
                        if sc_v is not None:
                            eval_score_pct = round(float(sc_v), 2)
                            if eval_sr_pct is None:
                                eval_sr_pct = eval_score_pct
                                eval_metric_name = "Score"
                        break

        merged.update({
            "live_status": live_status,
            "handoff_pending": handoff_pending,
            "staged_handoff": staged_handoff,
            "rounds_done": rounds_done,
            "current_round_groups_done": g_done if live_status != "completed" else 16,
            "groups_per_round": 16,
            "latest_sr_pct": latest_sr,
            "latest_successes": latest_succ,
            "latest_trajectories": latest_traj,
            "optimizer_steps": max(int((last_r or {}).get("optimizer_steps") or 0),
                                   int(csum.get("successful_adamw_count") or 0)),
            "latest_loss": last_m.get("loss"),
            "latest_kl": last_m.get("reference_kl"),
            "latest_entropy": last_m.get("entropy"),
            "latest_grad_norm": last_m.get("gradient_norm"),
            "env_steps": int(csum.get("env_steps") or 0),
            "input_tokens": int(csum.get("input_tokens") or 0),
            "output_tokens": int(csum.get("output_tokens") or 0),
            "forward_tokens": int(csum.get("forward_tokens") or 0),
            "total_events": int(csum.get("total_events") or 0),
            "session_elapsed_seconds": csum.get("session_elapsed_seconds"),
            "last_group_id": csum.get("last_group_id"),
            "last_phase_role": csum.get("last_phase_role"),
            "checkpoint_snapshot": (cd.get("checkpoint_current") or {}).get("snapshot"),
            "eval_sr_pct": eval_sr_pct,
            "eval_metric_name": eval_metric_name,
            "eval_sgc_pct": eval_sgc_pct,
            "eval_score_pct": eval_score_pct,
            "eval_valid_seen_sr_pct": eval_valid_seen_sr_pct,
            "evaluations": evals,
        })
        return merged

    def get_snapshot(self) -> dict[str, Any]:
        servers_snap: dict[str, Any] = {}
        for sid in ("colab", "host_a", "host_b"):
            with self.state_locks[sid]:
                servers_snap[sid] = copy.deepcopy(self.server_states[sid])

        valid_seen_map = self._load_local_valid_seen()
        archived_colab = self._load_archived_colab_cells()
        if archived_colab:
            colab_tel = servers_snap["colab"].get("telemetry") or {}
            colab_cells = colab_tel.setdefault("cells", {})
            for key, archive in archived_colab.items():
                live = colab_cells.get(key) or {}
                if not live.get("cell_summary") or not live.get("training_rounds"):
                    colab_cells[key] = {**live, **archive}
            servers_snap["colab"]["telemetry"] = colab_tel
        else:
            colab_tel = servers_snap["colab"].get("telemetry")
        if isinstance(colab_tel, dict) and isinstance(colab_tel.get("cells"), dict):
            for seed_i, vs_entry in valid_seen_map.items():
                ck0 = f"comparison/alfworld/fgpo/seed-{seed_i}"
                if ck0 in colab_tel["cells"] and colab_tel["cells"][ck0].get("exists"):
                    colab_tel["cells"][ck0].setdefault("evaluations", {})["valid_seen"] = {
                        "summary": vs_entry,
                        "completed_count": vs_entry.get("evaluated_count"),
                        "success_count": vs_entry.get("success_count"),
                    }

        # Determine active runner target cell per lane
        lane_active_runner_ck: dict[str, str | None] = {}
        lane_has_runner_map: dict[str, bool] = {}
        lane_has_queue_map: dict[str, bool] = {}
        lane_handoff_map: dict[str, bool] = {}
        for ldef in LANE_DEFINITIONS:
            lid = ldef["lane_id"]
            sid = ldef["server_id"]
            srv = servers_snap[sid]
            tel = (srv.get("telemetry") or {}) if srv.get("status") == "online" else {}
            procs = [p for p in (tel.get("proc_list") or []) if p.get("lane") == ldef["short_lane"]]
            rproc = next((p for p in procs if "runner" in (p.get("role") or "")), None)
            lane_has_runner_map[lid] = rproc is not None
            lane_has_queue_map[lid] = any(
                "run_fgpo_host_b_webshop_queue.sh" in (p.get("cmd") or "")
                or "launch-host-a-textcraft.sh" in (p.get("cmd") or "")
                for p in procs if p.get("role") == "queue_controller"
            )
            lane_handoff_map[lid] = bool((tel.get("gpu2_handoff") or {}).get("handoff_pending")) if lid == "host_b_gpu_2" else False
            chosen_ck = None
            if rproc and rproc.get("target_method"):
                t_suite = rproc.get("target_suite") or "comparison"
                t_bench = rproc.get("target_benchmark")
                t_meth = rproc.get("target_method")
                t_seeds = rproc.get("target_seeds") or []
                for sd in t_seeds:
                    cand_ck = f"{t_suite}/{t_bench}/{t_meth}/seed-{sd}"
                    raw_c = (tel.get("cells") or {}).get(cand_ck) or {}
                    cs_obj = raw_c.get("cell_summary") or {}
                    if not (cs_obj.get("status") == "complete" or cs_obj.get("evaluation")):
                        chosen_ck = cand_ck
                        break
            lane_active_runner_ck[lid] = chosen_ck

        # Build live-merged 18-cell schedule
        merged_18: list[dict[str, Any]] = []
        for sc in self.schedule_18:
            lid = sc["lane_id"]
            srv_tel = (servers_snap.get(sc["server_id"]) or {}).get("telemetry") or {}
            raw_cell = (srv_tel.get("cells") or {}).get(sc["cell_key"])
            merged_18.append(self._summarize_cell_live(
                sc,
                raw_cell,
                active_runner_ck=lane_active_runner_ck.get(lid),
                lane_has_runner=lane_has_runner_map.get(lid, False),
                lane_has_queue=lane_has_queue_map.get(lid, False),
                handoff_pending_on_lane=lane_handoff_map.get(lid, False),
                valid_seen_map=valid_seen_map,
            ))

        # Build 5 execution lanes
        lanes_out: list[dict[str, Any]] = []
        for ldef in LANE_DEFINITIONS:
            lid = ldef["lane_id"]
            sid = ldef["server_id"]
            srv = servers_snap[sid]
            tel = (srv.get("telemetry") or {}) if srv.get("status") == "online" else {}
            lane_cells = sorted([c for c in merged_18 if c["lane_id"] == lid], key=lambda x: x["lane_order"])
            active_cell = next((c for c in lane_cells if c["live_status"] in ("running", "evaluating")), None)
            if active_cell is None:
                active_cell = next((c for c in lane_cells if c["live_status"] == "paused"), None)
            if active_cell is None:
                active_cell = next((c for c in lane_cells if c["live_status"] in ("queued", "planned")), lane_cells[-1] if lane_cells else None)
            completed_cells = [c for c in lane_cells if c["live_status"] == "completed"]
            paused_cells = [c for c in lane_cells if c["live_status"] == "paused"]
            queued_cells = [c for c in lane_cells if c["live_status"] == "queued" and c is not active_cell]
            planned_cells = [c for c in lane_cells if c["live_status"] == "planned"]

            lane_gpus = [g for g in (tel.get("gpus") or []) if g.get("index") in ldef["gpu_indices"]]
            procs = [p for p in (tel.get("proc_list") or []) if p.get("lane") == ldef["short_lane"]]
            runner_proc = next((p for p in procs if "runner" in (p.get("role") or "")), None)
            sup_proc = next((p for p in procs if p.get("role") in ("supervisor", "queue_controller")), None)
            env_workers = sum(1 for p in procs if p.get("role") == "fgpo.envs")

            lanes_out.append({
                "lane_id": lid,
                "server_id": sid,
                "server_status": srv.get("status"),
                "lane_title": ldef["lane_title"],
                "short_lane": ldef["short_lane"],
                "role_slot": ldef["role_slot"],
                "gpu_indices": ldef["gpu_indices"],
                "gpus": lane_gpus,
                "handoff_pending": lane_handoff_map.get(lid, False),
                "gpu2_handoff": tel.get("gpu2_handoff") if lid == "host_b_gpu_2" else None,
                "runner_pid": runner_proc.get("pid") if runner_proc else None,
                "supervisor_pid": sup_proc.get("pid") if sup_proc else None,
                "cpu_affinity": (runner_proc or sup_proc or {}).get("affinity") or "-",
                "env_worker_count": env_workers,
                "active_cell": active_cell,
                "completed_cells": completed_cells,
                "paused_cells": paused_cells,
                "queued_cells": queued_cells,
                "planned_cells": planned_cells,
            })

        # Cluster-wide summary
        enabled_list = [s for s in servers_snap.values() if s.get("enabled")]
        online_count = sum(1 for s in enabled_list if s.get("status") == "online")
        usable_pool_gpus = 0
        active_gpus: list[dict[str, Any]] = []
        total_pids = 0
        for s in enabled_list:
            tel = s.get("telemetry") or {}
            glist = tel.get("gpus") or []
            usable_pool_gpus += len(glist)
            if s.get("status") == "online":
                active_gpus.extend(g for g in glist if g.get("active_in_experiment") and g.get("fgpo_mem_used_mib"))
                total_pids += len(tel.get("proc_list") or [])

        vram_used_mib = sum(float(g.get("mem_used_mib") or 0.0) for g in active_gpus)
        vram_tot_mib = sum(float(g.get("mem_total_mib") or 0.0) for g in active_gpus)
        fgpo_vram_mib = sum(float(g.get("fgpo_mem_used_mib") or 0.0) for g in active_gpus)
        ext_vram_mib = sum(float(g.get("external_mem_used_mib") or 0.0) for g in active_gpus)
        mean_gpu_util = round(sum(float(g.get("util_gpu_pct") or 0.0) for g in active_gpus) / len(active_gpus), 1) if active_gpus else None

        env_steps_cum = sum(int(c.get("env_steps") or 0) for c in merged_18)
        in_tok_cum = sum(int(c.get("input_tokens") or 0) for c in merged_18)
        out_tok_cum = sum(int(c.get("output_tokens") or 0) for c in merged_18)

        host_a_arch = ((servers_snap["host_a"].get("telemetry") or {}).get("archived_cells") or {})
        colab_usage = ((servers_snap["colab"].get("colab") or {}).get("usage") or {})

        cluster_summary = {
            "servers_enabled": len(enabled_list),
            "servers_online": online_count,
            "lanes_total": len(lanes_out),
            "lanes_running": sum(1 for ln in lanes_out if (ln.get("active_cell") or {}).get("live_status") in ("running", "evaluating")),
            "usable_pool_gpus": usable_pool_gpus,
            "active_experiment_gpus": len(active_gpus),
            "mean_active_gpu_util_pct": mean_gpu_util,
            "active_vram_used_mib": int(round(vram_used_mib)),
            "active_vram_total_mib": int(round(vram_tot_mib)),
            "active_vram_used_gib": round(vram_used_mib / 1024.0, 2),
            "active_vram_total_gib": round(vram_tot_mib / 1024.0, 2),
            "fgpo_vram_used_mib": int(round(fgpo_vram_mib)),
            "external_vram_used_mib": int(round(ext_vram_mib)),
            "cells_total": len(merged_18),
            "cells_completed": sum(1 for c in merged_18 if c["live_status"] == "completed"),
            "cells_running": sum(1 for c in merged_18 if c["live_status"] in ("running", "evaluating")),
            "cells_paused": sum(1 for c in merged_18 if c["live_status"] == "paused"),
            "cells_queued": sum(1 for c in merged_18 if c["live_status"] == "queued"),
            "cells_planned": sum(1 for c in merged_18 if c["live_status"] == "planned"),
            "archived_cells_count": len(host_a_arch),
            "env_steps_cum": env_steps_cum,
            "input_tokens_cum": in_tok_cum,
            "output_tokens_cum": out_tok_cum,
            "active_pids_count": total_pids,
            "colab_balance": colab_usage.get("balance"),
            "colab_rate": colab_usage.get("rate"),
        }

        colab_st = servers_snap["colab"]
        payload = {
            "protocol": "fgpo-r3-published-v4",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "servers": servers_snap,
            "lanes": lanes_out,
            "schedule": merged_18,
            "schedule_18": merged_18,
            "schedule_21": merged_18,
            "cluster_summary": cluster_summary,
            # Top-level Colab backward compatibility aliases
            "meta": self.meta,
            "status": colab_st.get("status"),
            "polling": any(s.get("polling") for s in servers_snap.values()),
            "last_ok_at": colab_st.get("last_ok_at"),
            "last_attempt_at": colab_st.get("last_attempt_at"),
            "last_error": colab_st.get("last_error"),
            "last_transport": colab_st.get("transport"),
            "colab": colab_st.get("colab") or {},
            "telemetry": colab_st.get("telemetry"),
            "history": colab_st.get("history") or [],
        }
        return _sanitize_secrets(payload, self.secret_replacements)


ColabPoller = ClusterPoller


# ---------------------------------------------------------------------------
# Unified Dashboard HTML (Self-contained CSS + Vanilla JS + Interactive SVGs)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>F-GPO 实验监控与排程</title>
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
      --c5: #ec4899;
      --c6: #06b6d4;
      --good: #10b981;
      --warn: #f59e0b;
      --bad: #ef4444;
      --code-bg: #090d14;
      --viz-fgpo: #3987e5;
      --viz-gigpo: #d95926;
      --viz-eval: #199e70;
      --viz-hosta: #a855f7;
      --viz-grid: #253047;
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
      --c5: #db2777;
      --c6: #0891b2;
      --good: #059669;
      --warn: #d97706;
      --bad: #dc2626;
      --code-bg: #f8fafc;
      --viz-fgpo: #2a78d6;
      --viz-gigpo: #eb6834;
      --viz-eval: #1baf7a;
      --viz-hosta: #7c3aed;
      --viz-grid: #e2e8f0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
      font-size: 13px; line-height: 1.5;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-variant-numeric: tabular-nums; }
    .container { max-width: 1440px; margin: 0 auto; padding: 16px 20px 44px; }
    .topbar {
      display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between;
      gap: 12px; padding-bottom: 14px; border-bottom: 1px solid var(--border); margin-bottom: 14px;
    }
    .title-wrap h1 { margin: 0; font-size: 18px; font-weight: 650; display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
    .sub { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .srv-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .srv-pill {
      display: inline-flex; align-items: center; gap: 7px; padding: 4px 10px;
      background: var(--card); border: 1px solid var(--border); border-radius: 999px; font-size: 12px;
    }
    .srv-pill button {
      background: transparent; border: none; color: var(--muted); cursor: pointer; padding: 0 2px; font-size: 12px;
    }
    .srv-pill button:hover { color: var(--accent); }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .btn, select, input[type="search"] {
      background: var(--card); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 6px 11px; font-size: 12px;
    }
    .btn, select { cursor: pointer; }
    .btn:hover { border-color: var(--accent); }
    .btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 600; }
    .btn-sm { padding: 3px 8px; font-size: 11.5px; border-radius: 6px; }
    .badge {
      display: inline-flex; align-items: center; gap: 6px; padding: 2px 9px;
      border-radius: 999px; font-size: 11.5px; font-weight: 600; border: 1px solid var(--border); background: var(--card-2);
    }
    .pill-tag {
      display: inline-block; padding: 1px 7px; border-radius: 5px; font-size: 11px;
      border: 1px solid var(--border); background: var(--card); color: var(--muted);
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
    .dot.good { background: var(--good); box-shadow: 0 0 7px var(--good); }
    .dot.warn { background: var(--warn); }
    .dot.bad { background: var(--bad); }
    .tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .tab {
      background: transparent; color: var(--muted); border: 1px solid transparent;
      border-radius: 8px; padding: 6px 14px; font-size: 13px; font-weight: 500; cursor: pointer;
    }
    .tab.active { background: var(--card); color: var(--text); border-color: var(--border); font-weight: 600; }
    .grid-5 { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    @media (max-width: 1200px) { .grid-5 { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 960px) { .grid-5, .grid-4, .grid-3, .grid-2 { grid-template-columns: 1fr; } }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
    .card-sub { background: var(--card-2); border: 1px solid var(--border); border-radius: 9px; padding: 10px 12px; }
    .card-h { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
    .card-t { font-size: 12px; color: var(--muted); font-weight: 600; letter-spacing: .02em; }
    .kpi { font-size: 22px; font-weight: 700; line-height: 1.2; margin: 2px 0 4px; }
    .kpi-sub { font-size: 12px; color: var(--muted); }
    .bar-bg { position: relative; width: 100%; height: 8px; background: var(--card-2); border-radius: 999px; overflow: hidden; margin-top: 6px; }
    .bar-fill { height: 100%; border-radius: 999px; transition: width .25s ease; }
    .grp-seg-row { display: flex; gap: 2px; margin-top: 5px; }
    .grp-seg { flex: 1; height: 7px; border-radius: 2px; background: var(--card-2); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
    th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; position: sticky; top: 0; background: var(--card); }
    pre.logbox {
      background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 10px 12px; max-height: 380px; overflow: auto; margin: 0; font-size: 12px; line-height: 1.45;
      white-space: pre-wrap; word-break: break-word;
    }
    .hidden { display: none !important; }
    .err-banner {
      background: rgba(245, 158, 11, .12); border: 1px solid var(--warn); color: var(--text);
      padding: 9px 13px; border-radius: 10px; margin-bottom: 12px; font-size: 12px;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--muted); margin-bottom: 8px; }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
    .line-key { width: 16px; height: 2.5px; border-radius: 2px; display: inline-block; }
    .glyph-key { font-size: 11px; letter-spacing: 1px; line-height: 1; }
    .viz-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 10px; }
    @media (max-width: 900px) { .viz-kpis { grid-template-columns: repeat(2, 1fr); } }
    .viz-kpi-tile { background: var(--card-2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 11px; }
    .viz-kpi-label { font-size: 11px; color: var(--muted); }
    .viz-kpi-val { font-size: 18px; font-weight: 650; margin-top: 2px; color: var(--text); }
    .viz-kpi-sub { font-size: 11px; color: var(--muted); margin-top: 1px; }
    .viz-wrap { position: relative; width: 100%; }
    .viz-tooltip {
      position: absolute; top: 14px; pointer-events: none; z-index: 30;
      min-width: 225px; max-width: 310px;
      background: var(--card); border: 1px solid var(--border); border-radius: 8px;
      padding: 8px 10px; box-shadow: 0 8px 24px rgba(0,0,0,.28); font-size: 12px; color: var(--text);
    }
    .float-tip {
      position: fixed; pointer-events: none; z-index: 60;
      min-width: 230px; max-width: 350px;
      background: var(--card); border: 1px solid var(--border); border-radius: 8px;
      padding: 8px 10px; box-shadow: 0 8px 24px rgba(0,0,0,.3); font-size: 12px; color: var(--text);
    }
    .viz-tip-head { font-weight: 650; margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
    .viz-tip-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 3px 0; }
    .viz-tip-key { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
    .viz-tip-val { font-weight: 650; color: var(--text); font-variant-numeric: tabular-nums; }
    .viz-note { font-size: 12px; color: var(--muted); margin-top: 8px; line-height: 1.5; }
    .btn-link {
      background: transparent; border: 1px solid var(--border); color: var(--text);
      border-radius: 6px; padding: 3px 8px; font-size: 11.5px; cursor: pointer;
    }
    .btn-link:hover { border-color: var(--accent); color: var(--accent); }
    svg.chart { width: 100%; display: block; }
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <div class="title-wrap">
        <h1>
          <span>F-GPO V4 三机远端实验实时监控台</span>
          <span id="cluster-badge" class="badge"><span class="dot warn"></span><span>初始化探测中...</span></span>
        </h1>
        <div class="sub mono" id="subtitle">训练与评测进度</div>
        <div class="srv-pills" id="server-status-pills"></div>
      </div>
      <div class="controls">
        <label class="sub">范围:
          <select id="scope-select">
            <option value="all" selected>三机全景 (All 3)</option>
            <option value="colab">仅 Colab</option>
            <option value="host_a">仅 Host A</option>
            <option value="host_b">仅 Host B</option>
          </select>
        </label>
        <label class="sub">自动刷新:
          <select id="auto-select">
            <option value="0">手动</option>
            <option value="15000">15s</option>
            <option value="30000" selected>30s</option>
            <option value="60000">60s</option>
          </select>
        </label>
        <button class="btn btn-primary" id="refresh-btn">立即探测三机</button>
        <button class="btn" id="theme-btn">切换亮/暗</button>
      </div>
    </div>

    <div id="err-box" class="err-banner hidden"></div>

    <div class="tabs">
      <button class="tab active" data-tab="overview">三机总览与通道</button>
      <button class="tab" data-tab="experiment">训练曲线与终评</button>
      <button class="tab" data-tab="schedule">当前排程</button>
      <button class="tab" data-tab="costs">成本分解与事件流</button>
      <button class="tab" data-tab="logs">进程与远端日志</button>
    </div>

    <!-- TAB 1: OVERVIEW -->
    <div id="tab-overview">
      <div class="grid-5" id="cluster-kpi-row"></div>
      <div class="card" style="margin-bottom:14px;">
        <div class="card-h">
          <span class="card-t">当前执行通道</span>
          <span class="kpi-sub">Host A · TextCraft　|　Host B · WebShop　|　Colab · alfworld</span>
        </div>
        <div class="grid-3" id="lanes-grid" style="margin-bottom:0;"></div>
      </div>
      <div id="servers-hw-stack" style="display:flex;flex-direction:column;gap:14px;"></div>
    </div>

    <!-- TAB 2: EXPERIMENT & GIGPO COMPARISON -->
    <div id="tab-experiment" class="hidden">
      <div class="card" style="margin-bottom:14px;">
        <div class="card-h">
          <span class="card-t">实验单元</span>
          <div class="controls">
            <label class="sub">单元:
              <select id="exp-cell-select"></select>
            </label>
          </div>
        </div>
        <div id="exp-quick-pills" class="controls" style="margin-bottom:8px;"></div>
        <div id="seed-detail-summary"></div>
      </div>

      <div class="card" id="alfworld-gigpo-comparison" style="margin-bottom:14px;">
        <div class="card-h" style="flex-wrap:wrap; gap:10px;">
          <div>
            <span class="card-t">GiGPO 论文训练曲线映射对比 · Success Rate vs Iteration</span>
            <div class="kpi-sub" id="gigpo-compare-subtitle">按同采样预算映射: 1 Iteration = 1 Round = 128 轨迹</div>
          </div>
          <div class="controls" id="gigpo-filter-row">
            <label class="sub">视野:
              <select id="gigpo-range-sel">
                <option value="zoom60" selected>前 60 轮放大 (Iter 1–60)</option>
                <option value="full150">GiGPO 全程 (Iter 0–150)</option>
              </select>
            </label>
            <label class="sub">主对比系列:
              <select id="gigpo-series-sel">
                <option value="selected" selected>当前选中单元</option>
                <option value="mean">同方法多 Seed 均值 + 各 Seed</option>
              </select>
            </label>
            <label class="sub" style="display:inline-flex;align-items:center;gap:5px;cursor:pointer;">
              <input type="checkbox" id="gigpo-band-chk" checked /> GiGPO 1.5B 阴影带
            </label>
            <label class="sub" style="display:inline-flex;align-items:center;gap:5px;cursor:pointer;">
              <input type="checkbox" id="gigpo-hosta-chk" checked /> 叠加 Host A 实测 7B GiGPO (已停归档)
            </label>
          </div>
        </div>
        <div class="legend" style="margin-bottom:8px;">
          <span class="legend-item"><span class="line-key" style="background:var(--viz-fgpo)"></span>当前选中实测单元 (128 轨迹/round)</span>
          <span class="legend-item"><span class="line-key" style="background:var(--viz-gigpo)"></span>GiGPO 论文均值 ± 带 (图源数字化 · Qwen2.5-1.5B, 150 iters)</span>
          <span class="legend-item"><span class="glyph-key" style="color:var(--viz-gigpo)">■ ▲ ●</span>GiGPO 论文里程碑 (Iter 10 / 75 / 140)</span>
          <span class="legend-item"><span class="line-key" style="background:var(--viz-hosta)"></span>Host A 7B GiGPO 归档基线</span>
          <span class="legend-item"><span class="line-key" style="background:var(--viz-eval)"></span>单 Seed 轨迹 / 官方终局 Eval</span>
        </div>
        <div class="viz-kpis" id="gigpo-compare-kpis"></div>
        <div class="viz-wrap">
          <svg id="gigpo-compare-chart" class="chart" viewBox="0 0 960 330" role="img" aria-label="F-GPO 与 GiGPO 成功率训练曲线对比图"></svg>
          <div id="gigpo-tooltip" class="viz-tooltip hidden"></div>
        </div>
        <div class="viz-note">
          对齐口径：每 round 与 GiGPO 每 iteration 均为 16 组 × 8 轨迹 = 128 条轨迹，严格按累计训练样本映射为 <span class="mono">Iter = Round + 1</span>（Round 0 → Iter 1）。紫色点线可选叠加 Host A 已停归档的 <span class="mono">comparison/alfworld/gigpo</span> 7B 基线轮次。
        </div>
        <div id="gigpo-milestone-table-wrap" style="overflow:auto; max-height:220px; margin-top:10px;"></div>
      </div>

      <div class="grid-3">
        <div class="card">
          <div class="card-h"><span class="card-t">训练轮次 Rollout 成功率 (SR %, 单轴)</span><span class="kpi-sub mono" id="sr-meta">-</span></div>
          <svg id="sr-chart" class="chart" viewBox="0 0 460 200"></svg>
        </div>
        <div class="card">
          <div class="card-h"><span class="card-t">PPO 策略损失 (Policy Loss, 单轴)</span><span class="kpi-sub mono" id="loss-meta">-</span></div>
          <svg id="loss-chart" class="chart" viewBox="0 0 460 200"></svg>
        </div>
        <div class="card">
          <div class="card-h"><span class="card-t">参考策略散度 (Reference KL, 单轴)</span><span class="kpi-sub mono" id="kl-meta">-</span></div>
          <svg id="kl-chart" class="chart" viewBox="0 0 460 200"></svg>
        </div>
      </div>

      <div class="card">
        <div class="card-h"><span class="card-t">轮次明细与官方终局评估结果 (valid_unseen / valid_seen / WebShop / AppWorld TGC &amp; SGC)</span></div>
        <div id="rounds-table-wrap" style="overflow:auto; max-height:380px;"></div>
      </div>
    </div>

    <!-- TAB 3: 18-CELL SCHEDULE -->
    <div id="tab-schedule" class="hidden">
      <div class="card" style="margin-bottom:14px;">
        <div class="card-h">
          <div>
            <span class="card-t">逐卡训练排程</span>
            <div class="kpi-sub">块宽对应训练轮数，填充反映已完成轮次。后续计划以逐种子表注明的状态为准。</div>
          </div>
        </div>
        <div class="legend" id="gantt-legend"></div>
        <svg id="gantt-svg" class="chart" viewBox="0 0 1040 236"></svg>
      </div>

      <div class="card" style="margin-bottom:14px;">
        <div class="card-h">
          <span class="card-t">逐种子状态</span>
          <div class="controls">
            <select id="sched-srv-filter">
              <option value="all">全部服务器</option>
              <option value="colab">Colab</option>
              <option value="host_a">Host A</option>
              <option value="host_b">Host B</option>
            </select>
            <select id="sched-bench-filter">
              <option value="all">全部 Benchmark</option>
              <option value="alfworld">alfworld</option>
              <option value="webshop">WebShop</option>
              <option value="textcraft">TextCraft</option>
            </select>
          </div>
        </div>
        <div id="sched-table-wrap" style="overflow:auto;"></div>
      </div>

      <div class="card">
        <div class="card-h"><span class="card-t">后续实验安排</span></div>
        <table><thead><tr><th>服务器</th><th>顺序</th><th>当前状态</th></tr></thead><tbody>
          <tr><td>Host A · TextCraft</td><td>3+4：F-GPO → Obs-Only → No-Support-Weight<br>1+7：GraphGPO → G2PO → No-Transfer</td><td>每个方法按 seed 0、1、2 接续。每种子 50 轮、79 题终评。BiPACE 已发表基线按作者口径引用。</td></tr>
          <tr><td>Host B · WebShop</td><td>GPU 0：F-GPO 0、2 → Obs-Only 0、2 → No-Transfer 2<br>GPU 2：F-GPO 1 → Obs-Only 1 → No-Transfer 1、0</td><td>每种子 60 轮、128 题终评。消融等同种子 F-GPO 终评。7B 公开基线已有 Score 和 SR。</td></tr>
          <tr><td>Colab · alfworld</td><td>完整 F-GPO seed 2 → No-Transfer seed 0–2</td><td>待可用正式会话。seed 0、1 结果已归档。</td></tr>
        </tbody></table>
        <p class="kpi-sub">两台主机的 AppWorld 已退出排程。Host A 的旧 WebShop 结果和检查点作为历史记录保留。</p>
      </div>

      <div class="card">
        <div class="card-h">
          <span class="card-t">已停归档基线单元 (Host A 7B ALFWorld GiGPO · Clean SIGTERM Checkpoint Archived)</span>
          <span class="kpi-sub">历史检查点和训练日志供对比</span>
        </div>
        <div id="archived-table-wrap" style="overflow:auto;"></div>
      </div>
    </div>

    <!-- TAB 4: COSTS & EVENTS -->
    <div id="tab-costs" class="hidden">
      <div class="card" style="margin-bottom:14px;">
        <div class="card-h">
          <div>
            <span class="card-t">活跃通道非重叠 6 阶段耗时分解 (严格去重嵌套父子计时器，总和 &lt;= 墙钟 elapsed)</span>
            <div class="kpi-sub">点击任一通道条形或使用下方下拉框切换查看该单元算子调用分布与最新 20 条真实事件</div>
          </div>
          <div class="controls">
            <button class="btn btn-sm btn-primary" id="stage-mode-pct">占比视图 (100%)</button>
            <button class="btn btn-sm" id="stage-mode-sec">绝对秒数 (Seconds)</button>
            <label class="sub">明细单元:
              <select id="cost-cell-select"></select>
            </label>
          </div>
        </div>
        <div class="legend" id="stage-legend"></div>
        <svg id="stage-svg" class="chart" viewBox="0 0 920 220"></svg>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-h"><span class="card-t">原始 Costs Ledger 算子调用次数与累计秒数</span><span class="kpi-sub mono" id="cost-total-meta">-</span></div>
          <div id="cost-bars" style="max-height:420px;overflow:auto;"></div>
        </div>
        <div class="card">
          <div class="card-h"><span class="card-t">最近 20 条真实交互与模型事件 (costs.jsonl)</span></div>
          <div id="recent-events-wrap" style="overflow:auto; max-height:420px;"></div>
        </div>
      </div>
    </div>

    <!-- TAB 5: PROCESSES & LOGS -->
    <div id="tab-logs" class="hidden">
      <div class="card" style="margin-bottom:14px;">
        <div class="card-h">
          <span class="card-t">三机活跃实验进程树 (PID / PPID / NUMA 亲和性 / RSS)</span>
          <div class="controls">
            <input type="search" id="proc-search" placeholder="搜索 PID / 通道 / 角色 / 命令..." style="width:220px;" />
            <select id="proc-srv-filter">
              <option value="all">全部服务器</option>
              <option value="colab">Colab</option>
              <option value="host_a">Host A</option>
              <option value="host_b">Host B</option>
            </select>
            <select id="proc-role-filter">
              <option value="all">全部角色</option>
              <option value="supervisor">Supervisor / Queue / Handoff</option>
              <option value="runner">Study Runner</option>
              <option value="envs">fgpo.envs Worker</option>
            </select>
          </div>
        </div>
        <div id="procs-wrap" style="overflow:auto; max-height:380px;"></div>
      </div>

      <div class="card">
        <div class="card-h">
          <span class="card-t">远端通道日志尾部 (<span id="log-path-label" class="mono">-</span>)</span>
          <div class="controls" id="log-key-pills"></div>
        </div>
        <pre id="log-tail" class="logbox mono">等待拉取...</pre>
      </div>
    </div>
  </div>

  <div id="float-tip" class="float-tip hidden"></div>

  <script>
    let currentTab = "overview";
    let scopeServer = "all";
    let selectedGlobalCell = "colab:comparison/alfworld/fgpo/seed-0";
    let selectedLogKey = "";
    let stageNormMode = "pct";
    let lastData = null;
    let timer = null;

    const LANE_COLORS = {
      colab_gpu_0: "var(--c1)",
      host_a_l40_3_4: "var(--c2)",
      host_a_l40_1_7: "var(--c3)",
      host_b_gpu_0: "var(--c4)",
      host_b_gpu_2: "var(--c5)"
    };

    const METHOD_COLORS = {
      fgpo: "var(--c1)",
      fgpo_no_transfer: "var(--c2)",
      fgpo_obs_only: "var(--c3)",
      gigpo: "var(--c4)",
      graphgpo: "var(--c4)",
      fgpo_no_support_weight: "var(--viz-eval)",
      g2po: "var(--c5)"
    };

    const STAGE_META = [
      { key: "rollout_draw", label: "Rollout Draw / Coord", color: "var(--c1)" },
      { key: "decode_prefill", label: "Actor Decode & Prefill", color: "var(--c2)" },
      { key: "env_io", label: "Env Reset & Step", color: "var(--c3)" },
      { key: "ppo_optim", label: "PPO Fwd/Bwd/AdamW", color: "var(--c4)" },
      { key: "ref_credit", label: "Ref KL & Peer Credit", color: "var(--c5)" },
      { key: "ckpt_startup", label: "Startup / Load / Ckpt", color: "var(--c6)" }
    ];

    const fmtInt = (n) => (n == null ? "-" : Number(n).toLocaleString("en-US"));
    const fmtNum = (n, d = 2) => (n == null ? "-" : Number(n).toFixed(d));
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

    function fmtDur(sec) {
      if (sec == null) return "-";
      const s = Math.round(Number(sec));
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      return `${h}h ${m}m (${fmtNum(sec, 1)}s)`;
    }

    // Exact non-overlapping 6-stage timer decomposition (sum <= session_elapsed_seconds)
    function categorizeRoleSeconds(rs) {
      rs = rs || {};
      const decode_prefill = (rs["rollout.decode"] || 0) + (rs["gigpo.actor.decode"] || 0) + (rs["gigpo.actor.shared_prefill"] || 0) + (rs["gigpo.actor.batch_prefill"] || 0) + (rs["g2po.actor.decode"] || 0) + (rs["g2po.actor.shared_prefill"] || 0) + (rs["g2po.actor.batch_prefill"] || 0) + (rs["rollout.turn_append"] || 0) + (rs["rollout.shared_root"] || 0) + (rs["rollout.segment_replay"] || 0) + (rs["rollout.merge_cohorts"] || 0) + (rs["rollout.collection_replica_refresh"] || 0);
      const env_io = (rs["rollout.create"] || rs["native.create"] || 0) + (rs["rollout.reset"] || rs["native.reset"] || 0) + (rs["rollout.step_native"] || rs["native.step"] || 0) + (rs["rollout.close"] || rs["native.close"] || 0) + (rs["rollout.official_evaluation"] || rs["native.official_scoring"] || 0);
      const totalRollout = Math.max((rs["rollout.group"] || 0) + (rs["evaluation.rollout"] || 0), (rs["rollout.draw"] || 0) + (rs["rollout.create"] || rs["native.create"] || 0) + (rs["rollout.reset"] || rs["native.reset"] || 0));
      const rollout_draw = Math.max(0, totalRollout - decode_prefill - env_io);
      const ppo_optim = (rs["ppo.forward"] || 0) + (rs["ppo.backward"] || 0) + (rs["ppo.adamw"] || 0);
      const ref_credit = (rs["reference.single_round_activation"] || 0) + (rs["credit.peer_values"] || 0) + (rs["baseline.advantages"] || 0);
      const ckpt_startup = (rs["checkpoint.publish"] || 0) + (rs["checkpoint.completed_artifacts"] || 0) + (rs["checkpoint.reclaim_completed_resume_state"] || 0) + (rs["checkpoint.final_publish_logs"] || 0) + (rs["runner.cold_asset_restore"] || 0) + (rs["cuda.preallocate"] || 0) + (rs["cuda.preallocate_after_dispatch"] || 0) + ((rs["runner.cell_startup"] || 0) >= (rs["model.load"] || 0) ? (rs["runner.cell_startup"] || 0) : (rs["runner.cell_startup"] || 0) + (rs["model.load"] || 0));
      return { rollout_draw, decode_prefill, env_io, ppo_optim, ref_credit, ckpt_startup };
    }

    const floatTip = document.getElementById("float-tip");
    function showFloatTip(evt, headText, rows) {
      floatTip.textContent = "";
      const hd = document.createElement("div");
      hd.className = "viz-tip-head mono";
      hd.textContent = headText;
      floatTip.appendChild(hd);
      rows.forEach((r) => {
        const row = document.createElement("div");
        row.className = "viz-tip-row";
        const k = document.createElement("span");
        k.className = "viz-tip-key";
        if (r.color) {
          const sw = document.createElement("span");
          sw.className = "line-key";
          sw.style.background = r.color;
          k.appendChild(sw);
        }
        const kl = document.createElement("span");
        kl.textContent = r.label;
        k.appendChild(kl);
        const v = document.createElement("span");
        v.className = "viz-tip-val mono";
        v.textContent = r.value;
        row.appendChild(k);
        row.appendChild(v);
        floatTip.appendChild(row);
      });
      floatTip.classList.remove("hidden");
      const vw = window.innerWidth || 1280;
      const vh = window.innerHeight || 800;
      const lx = evt.clientX + 320 > vw ? Math.max(10, Math.round(evt.clientX - 310)) : Math.round(evt.clientX + 14);
      const ty = evt.clientY + 210 > vh ? Math.max(10, Math.round(evt.clientY - 180)) : Math.round(evt.clientY + 14);
      floatTip.style.left = lx + "px";
      floatTip.style.top = ty + "px";
    }
    function hideFloatTip() { floatTip.classList.add("hidden"); }

    function switchTab(tabName) {
      currentTab = tabName;
      hideFloatTip();
      document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabName));
      ["overview", "experiment", "schedule", "costs", "logs"].forEach((t) => {
        document.getElementById("tab-" + t).classList.toggle("hidden", t !== tabName);
      });
    }

    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        switchTab(btn.dataset.tab);
        if (lastData) render(lastData);
      });
    });

    window._inspectCell = (globalId, tabTarget) => {
      selectedGlobalCell = globalId;
      switchTab(tabTarget || "experiment");
      if (lastData) render(lastData);
    };

    window._refreshServer = async (srv) => {
      try {
        const res = await fetch("/api/refresh?server=" + encodeURIComponent(srv), { method: "POST" });
        const data = await res.json();
        render(data);
      } catch (e) {
        console.error(e);
      }
    };

    document.getElementById("theme-btn").addEventListener("click", () => {
      document.documentElement.classList.toggle("light");
      if (lastData) render(lastData);
    });

    document.getElementById("scope-select").addEventListener("change", (e) => {
      scopeServer = e.target.value;
      if (lastData) render(lastData);
    });

    ["gigpo-range-sel", "gigpo-series-sel", "gigpo-band-chk", "gigpo-hosta-chk", "sched-srv-filter", "sched-bench-filter", "proc-srv-filter", "proc-role-filter"].forEach((id) => {
      const node = document.getElementById(id);
      if (node) node.addEventListener("change", () => { if (lastData) render(lastData); });
    });

    document.getElementById("proc-search").addEventListener("input", () => { if (lastData) render(lastData); });
    document.getElementById("exp-cell-select").addEventListener("change", (e) => {
      selectedGlobalCell = e.target.value;
      if (lastData) render(lastData);
    });
    document.getElementById("cost-cell-select").addEventListener("change", (e) => {
      selectedGlobalCell = e.target.value;
      if (lastData) render(lastData);
    });

    document.getElementById("stage-mode-pct").addEventListener("click", () => {
      stageNormMode = "pct";
      document.getElementById("stage-mode-pct").classList.add("btn-primary");
      document.getElementById("stage-mode-sec").classList.remove("btn-primary");
      if (lastData) render(lastData);
    });
    document.getElementById("stage-mode-sec").addEventListener("click", () => {
      stageNormMode = "sec";
      document.getElementById("stage-mode-sec").classList.add("btn-primary");
      document.getElementById("stage-mode-pct").classList.remove("btn-primary");
      if (lastData) render(lastData);
    });

    // Digitized GiGPO 150-iteration curve (Qwen2.5-1.5B, 150 iters, 128 rollouts/iter)
    const GIGPO_CURVE = [
      [1, 0.027, 0.01, 0.053], [2, 0.063, 0.019, 0.106], [3, 0.112, 0.068, 0.155], [4, 0.133, 0.106, 0.159],
      [5, 0.129, 0.101, 0.155], [6, 0.094, 0.063, 0.13], [7, 0.068, 0.048, 0.087], [8, 0.077, 0.058, 0.155],
      [9, 0.109, 0.01, 0.222], [10, 0.109, 0.01, 0.222], [11, 0.109, 0.058, 0.14], [12, 0.109, 0.063, 0.14],
      [13, 0.098, 0.068, 0.14], [14, 0.106, 0.077, 0.135], [15, 0.115, 0.082, 0.15], [16, 0.118, 0.082, 0.155],
      [17, 0.121, 0.087, 0.159], [18, 0.097, 0.058, 0.135], [19, 0.134, 0.082, 0.184], [20, 0.163, 0.13, 0.203],
      [21, 0.155, 0.111, 0.203], [22, 0.174, 0.14, 0.208], [23, 0.181, 0.14, 0.217], [24, 0.157, 0.116, 0.198],
      [25, 0.14, 0.106, 0.174], [26, 0.165, 0.135, 0.203], [27, 0.174, 0.13, 0.208], [28, 0.185, 0.126, 0.242],
      [29, 0.236, 0.174, 0.3], [30, 0.251, 0.203, 0.295], [31, 0.25, 0.208, 0.3], [32, 0.224, 0.15, 0.309],
      [33, 0.193, 0.145, 0.246], [34, 0.213, 0.159, 0.266], [35, 0.256, 0.203, 0.304], [36, 0.292, 0.246, 0.338],
      [37, 0.323, 0.275, 0.367], [38, 0.309, 0.251, 0.372], [39, 0.328, 0.256, 0.396], [40, 0.363, 0.295, 0.444],
      [41, 0.315, 0.227, 0.401], [42, 0.299, 0.217, 0.396], [43, 0.348, 0.246, 0.44], [44, 0.376, 0.329, 0.43],
      [45, 0.373, 0.329, 0.42], [46, 0.364, 0.304, 0.415], [47, 0.346, 0.271, 0.415], [48, 0.392, 0.295, 0.478],
      [49, 0.431, 0.353, 0.493], [50, 0.404, 0.348, 0.454], [51, 0.398, 0.343, 0.459], [52, 0.436, 0.367, 0.512],
      [53, 0.417, 0.348, 0.488], [54, 0.427, 0.348, 0.512], [55, 0.482, 0.411, 0.551], [56, 0.473, 0.401, 0.536],
      [57, 0.462, 0.382, 0.541], [58, 0.432, 0.348, 0.517], [59, 0.403, 0.309, 0.483], [60, 0.508, 0.396, 0.623],
      [61, 0.625, 0.541, 0.705], [62, 0.547, 0.435, 0.657], [63, 0.482, 0.396, 0.565], [64, 0.424, 0.353, 0.483],
      [65, 0.514, 0.401, 0.628], [66, 0.597, 0.498, 0.676], [67, 0.578, 0.527, 0.623], [68, 0.608, 0.551, 0.671],
      [69, 0.666, 0.604, 0.739], [70, 0.655, 0.58, 0.725], [71, 0.601, 0.517, 0.691], [72, 0.575, 0.502, 0.647],
      [73, 0.577, 0.498, 0.657], [74, 0.522, 0.483, 0.638], [75, 0.531, 0.483, 0.647], [76, 0.524, 0.483, 0.638],
      [77, 0.623, 0.507, 0.729], [78, 0.569, 0.483, 0.681], [79, 0.577, 0.483, 0.691], [80, 0.632, 0.517, 0.744],
      [81, 0.72, 0.609, 0.836], [82, 0.754, 0.667, 0.841], [83, 0.723, 0.614, 0.826], [84, 0.653, 0.541, 0.768],
      [85, 0.596, 0.483, 0.71], [86, 0.606, 0.502, 0.686], [87, 0.672, 0.585, 0.754], [88, 0.699, 0.623, 0.758],
      [89, 0.688, 0.609, 0.763], [90, 0.709, 0.614, 0.802], [91, 0.73, 0.657, 0.807], [92, 0.684, 0.58, 0.773],
      [93, 0.625, 0.512, 0.725], [94, 0.683, 0.57, 0.797], [95, 0.736, 0.623, 0.85], [96, 0.746, 0.633, 0.86],
      [97, 0.785, 0.671, 0.894], [98, 0.837, 0.744, 0.937], [99, 0.786, 0.676, 0.889], [100, 0.818, 0.705, 0.928],
      [101, 0.872, 0.758, 0.981], [102, 0.794, 0.681, 0.908], [103, 0.747, 0.638, 0.845], [104, 0.743, 0.652, 0.826],
      [105, 0.778, 0.691, 0.86], [106, 0.736, 0.643, 0.831], [107, 0.699, 0.618, 0.787], [108, 0.734, 0.657, 0.812],
      [109, 0.754, 0.691, 0.821], [110, 0.778, 0.705, 0.85], [111, 0.816, 0.739, 0.889], [112, 0.832, 0.787, 0.879],
      [113, 0.848, 0.807, 0.903], [114, 0.875, 0.841, 0.918], [115, 0.892, 0.845, 0.937], [116, 0.85, 0.787, 0.908],
      [117, 0.789, 0.729, 0.85], [118, 0.801, 0.739, 0.87], [119, 0.823, 0.749, 0.889], [120, 0.804, 0.739, 0.865],
      [121, 0.791, 0.734, 0.855], [122, 0.82, 0.744, 0.908], [123, 0.844, 0.758, 0.932], [124, 0.847, 0.749, 0.942],
      [125, 0.876, 0.787, 0.957], [126, 0.874, 0.807, 0.942], [127, 0.863, 0.797, 0.932], [128, 0.836, 0.758, 0.913],
      [129, 0.791, 0.696, 0.884], [130, 0.835, 0.749, 0.918], [131, 0.829, 0.744, 0.918], [132, 0.811, 0.734, 0.889],
      [133, 0.807, 0.744, 0.87], [134, 0.803, 0.749, 0.86], [135, 0.827, 0.763, 0.889], [136, 0.875, 0.812, 0.932],
      [137, 0.897, 0.845, 0.961], [138, 0.904, 0.845, 0.966], [139, 0.874, 0.831, 0.952], [140, 0.874, 0.758, 0.99],
      [141, 0.874, 0.836, 0.918], [142, 0.849, 0.812, 0.899], [143, 0.84, 0.807, 0.874], [144, 0.839, 0.812, 0.865],
      [145, 0.86, 0.831, 0.894], [146, 0.879, 0.845, 0.918], [147, 0.871, 0.816, 0.923], [148, 0.843, 0.792, 0.899],
      [149, 0.867, 0.816, 0.918], [150, 0.886, 0.836, 0.937]
    ];
    const GIGPO_BY_ITER = Object.fromEntries(GIGPO_CURVE.map((p) => [p[0], { iter: p[0], mean: p[1], low: p[2], high: p[3] }]));
    const GIGPO_MILESTONES = [
      { iter: 10, mean: 0.109, shape: "square", label: "Iter 10" },
      { iter: 75, mean: 0.531, shape: "triangle", label: "Iter 75" },
      { iter: 140, mean: 0.874, shape: "circle", label: "Iter 140" }
    ];
    let gigpoVizState = null;

    function roundRate(r) {
      return r && r.trajectories ? r.successes / r.trajectories : null;
    }

    function fmtSignedDelta(delta) {
      if (delta == null) return "-";
      const sign = delta >= 0 ? "+" : "";
      return `${sign}${delta.toFixed(3)} (${sign}${(delta * 100).toFixed(1)} pp)`;
    }

    function updateGigpoTooltip(iter) {
      const tip = document.getElementById("gigpo-tooltip");
      const cross = document.getElementById("gigpo-crosshair");
      const hiGigpo = document.getElementById("gigpo-hi-gigpo");
      const hiFgpo = document.getElementById("gigpo-hi-fgpo");
      if (!tip || !gigpoVizState) return;
      const st = gigpoVizState;
      const it = Math.max(1, Math.min(st.xMax, Math.round(iter)));
      const gp = GIGPO_BY_ITER[it];
      const fg = st.primaryByIter[it] || null;
      const ha = st.hostAGigpoByIter ? st.hostAGigpoByIter[it] : null;
      const cx = st.xToPx(it);
      if (cross) {
        cross.setAttribute("x1", String(cx));
        cross.setAttribute("x2", String(cx));
        cross.style.display = "inline";
      }
      if (hiGigpo && gp) {
        hiGigpo.setAttribute("cx", String(cx));
        hiGigpo.setAttribute("cy", String(st.yToPx(gp.mean)));
        hiGigpo.style.display = "inline";
      } else if (hiGigpo) {
        hiGigpo.style.display = "none";
      }
      if (hiFgpo && fg && fg.sr != null) {
        hiFgpo.setAttribute("cx", String(cx));
        hiFgpo.setAttribute("cy", String(st.yToPx(fg.sr)));
        hiFgpo.style.display = "inline";
      } else if (hiFgpo) {
        hiFgpo.style.display = "none";
      }

      tip.textContent = "";
      const head = document.createElement("div");
      head.className = "viz-tip-head mono";
      head.textContent = `Iteration ${it} · 累计 ${(it * 128).toLocaleString("en-US")} 条轨迹`;
      tip.appendChild(head);

      const addRow = (colorVar, labelText, valText) => {
        const row = document.createElement("div");
        row.className = "viz-tip-row";
        const keyWrap = document.createElement("span");
        keyWrap.className = "viz-tip-key";
        const stroke = document.createElement("span");
        stroke.className = "line-key";
        stroke.style.background = colorVar;
        const lbl = document.createElement("span");
        lbl.textContent = labelText;
        keyWrap.appendChild(stroke);
        keyWrap.appendChild(lbl);
        const val = document.createElement("span");
        val.className = "viz-tip-val mono";
        val.textContent = valText;
        row.appendChild(keyWrap);
        row.appendChild(val);
        tip.appendChild(row);
      };

      if (fg && fg.sr != null) {
        addRow("var(--viz-fgpo)", st.primaryLabel, `${fg.sr.toFixed(3)} (${(fg.sr * 100).toFixed(1)}% · ${fg.detail || ""})`);
      } else {
        addRow("var(--viz-fgpo)", st.primaryLabel, it <= st.fgpoBudgetMax ? "尚未完成此轮" : `超出 ${st.fgpoBudgetMax} 轮预算`);
      }
      if (gp) {
        addRow("var(--viz-gigpo)", "GiGPO 1.5B 论文", `${gp.mean.toFixed(3)} [${gp.low.toFixed(3)}–${gp.high.toFixed(3)}]`);
      }
      if (ha && ha.sr != null) {
        addRow("var(--viz-hosta)", "Host A 7B GiGPO (归档)", `${ha.sr.toFixed(3)} (${(ha.sr * 100).toFixed(1)}%)`);
      }
      if (fg && fg.sr != null && gp) {
        addRow("transparent", "差值 (实测 − 论文)", fmtSignedDelta(fg.sr - gp.mean));
      }

      const pctX = (cx / st.W) * 100;
      tip.style.left = pctX > 62 ? `${Math.max(4, Math.round(pctX - 2))}%` : `${Math.min(92, Math.round(pctX + 2))}%`;
      tip.style.transform = pctX > 62 ? "translateX(-100%)" : "none";
      tip.classList.remove("hidden");
    }

    function hideGigpoTooltip() {
      const tip = document.getElementById("gigpo-tooltip");
      const cross = document.getElementById("gigpo-crosshair");
      const hiGigpo = document.getElementById("gigpo-hi-gigpo");
      const hiFgpo = document.getElementById("gigpo-hi-fgpo");
      if (tip) tip.classList.add("hidden");
      if (cross) cross.style.display = "none";
      if (hiGigpo) hiGigpo.style.display = "none";
      if (hiFgpo) hiFgpo.style.display = "none";
    }

    (function bindGigpoHoverOnce() {
      const svg = document.getElementById("gigpo-compare-chart");
      if (!svg) return;
      svg.addEventListener("pointermove", (ev) => {
        if (!gigpoVizState) return;
        const rect = svg.getBoundingClientRect();
        if (!rect.width) return;
        const svgX = ((ev.clientX - rect.left) / rect.width) * gigpoVizState.W;
        if (svgX < gigpoVizState.padL - 8 || svgX > gigpoVizState.W - gigpoVizState.padR + 8) {
          hideGigpoTooltip();
          return;
        }
        const rawIter = ((svgX - gigpoVizState.padL) / gigpoVizState.plotW) * gigpoVizState.xMax;
        updateGigpoTooltip(rawIter);
      });
      svg.addEventListener("pointerleave", hideGigpoTooltip);
    })();

    function getRawCellFromData(data, srvId, cellKey) {
      const tel = ((data.servers || {})[srvId] || {}).telemetry || {};
      return ((tel.cells || {})[cellKey]) || null;
    }

    function renderGigpoComparison(data, selMeta, selRaw) {
      const rangeMode = (document.getElementById("gigpo-range-sel") || {}).value || "zoom60";
      const seriesMode = (document.getElementById("gigpo-series-sel") || {}).value || "selected";
      const showBand = (document.getElementById("gigpo-band-chk") || {}).checked !== false;
      const showHostA = (document.getElementById("gigpo-hosta-chk") || {}).checked !== false;

      const fgpoBudgetMax = selMeta ? (selMeta.total_rounds || 60) : 60;
      const xMax = rangeMode === "full150" ? 150 : Math.max(60, fgpoBudgetMax);
      const schedList = data.schedule_18 || data.schedule_21 || [];

      const siblings = schedList.filter((c) =>
        selMeta && c.server_id === selMeta.server_id && c.suite === selMeta.suite && c.benchmark === selMeta.benchmark && c.method === selMeta.method
      );

      const bySeed = {};
      siblings.forEach((sc) => {
        const rc = getRawCellFromData(data, sc.server_id, sc.cell_key) || {};
        const pts = {};
        (rc.training_rounds || []).forEach((r) => {
          const it = Number(r.round) + 1;
          const sr = roundRate(r);
          if (sr != null) {
            pts[it] = { iter: it, round: Number(r.round), sr, detail: `${r.successes}/${r.trajectories}` };
          }
        });
        bySeed[String(sc.seed)] = { pts, evalSr: sc.eval_sr_pct != null ? sc.eval_sr_pct / 100.0 : null };
      });

      const activeSeedKey = String(selMeta ? selMeta.seed : 0);
      const primaryByIter = {};
      let primaryLabel = selMeta ? `${selMeta.method} s${selMeta.seed} (${selMeta.benchmark})` : "F-GPO s0";
      if (seriesMode === "mean" && siblings.length > 0) {
        primaryLabel = `${selMeta.method} 多 Seed 均值`;
        for (let it = 1; it <= fgpoBudgetMax; it++) {
          const vals = Object.keys(bySeed).map((sk) => bySeed[sk].pts[it]).filter(Boolean);
          if (vals.length) {
            primaryByIter[it] = {
              iter: it,
              round: it - 1,
              sr: vals.reduce((a, x) => a + x.sr, 0) / vals.length,
              detail: `${vals.length} seeds`
            };
          }
        }
      } else if (bySeed[activeSeedKey]) {
        Object.assign(primaryByIter, bySeed[activeSeedKey].pts);
      }

      // Host A 7B GiGPO overlay from archived_cells (fallback cells)
      const hostAGigpoByIter = {};
      if (showHostA) {
        const haTel = (((data.servers || {}).host_a || {}).telemetry) || {};
        const archMap = Object.keys(haTel.archived_cells || {}).length ? haTel.archived_cells : (haTel.cells || {});
        const gigpoEntries = Object.entries(archMap).filter(([k]) => k.startsWith("comparison/alfworld/gigpo/"));
        for (let it = 1; it <= 150; it++) {
          const vals = [];
          gigpoEntries.forEach(([, rc]) => {
            const rd = ((rc || {}).training_rounds || []).find((r) => Number(r.round) + 1 === it);
            const sr = roundRate(rd);
            if (sr != null) vals.push(sr);
          });
          if (vals.length) {
            hostAGigpoByIter[it] = { iter: it, sr: vals.reduce((a, b) => a + b, 0) / vals.length, n: vals.length };
          }
        }
      }

      const primaryPoints = Object.values(primaryByIter).sort((a, b) => a.iter - b.iter);
      const latestPt = primaryPoints.length ? primaryPoints[primaryPoints.length - 1] : null;
      const gpLatest = latestPt ? GIGPO_BY_ITER[latestPt.iter] : null;

      const subEl = document.getElementById("gigpo-compare-subtitle");
      if (subEl) {
        subEl.textContent = latestPt
          ? `当前对齐点 Round ${latestPt.round} → Iter ${latestPt.iter}/${fgpoBudgetMax} (${fmtInt(latestPt.iter * 128)} 轨迹) · 视野 0–${xMax} Iter`
          : `${primaryLabel} 尚未完成第 0 轮 (当前处于 R0 采样或排队中)，已展示 GiGPO 基准曲线`;
      }

      const kpiWrap = document.getElementById("gigpo-compare-kpis");
      if (kpiWrap) {
        kpiWrap.innerHTML = `
          <div class="viz-kpi-tile">
            <div class="viz-kpi-label">对齐预算位置</div>
            <div class="viz-kpi-val">${latestPt ? `Iter ${latestPt.iter} / ${fgpoBudgetMax}` : `R0 / ${fgpoBudgetMax}`}</div>
            <div class="viz-kpi-sub mono">${latestPt ? `Round ${latestPt.round} · ${fmtInt(latestPt.iter * 128)} 轨迹` : `Group ${selMeta ? selMeta.current_round_groups_done : 0}/16`}</div>
          </div>
          <div class="viz-kpi-tile">
            <div class="viz-kpi-label">${esc(primaryLabel)} SR</div>
            <div class="viz-kpi-val">${latestPt ? latestPt.sr.toFixed(3) : "-"}</div>
            <div class="viz-kpi-sub mono">${latestPt ? `${(latestPt.sr * 100).toFixed(1)}% (${esc(latestPt.detail)})` : "等待首轮写入"}</div>
          </div>
          <div class="viz-kpi-tile">
            <div class="viz-kpi-label">同采样量 GiGPO 论文 SR</div>
            <div class="viz-kpi-val">${gpLatest ? gpLatest.mean.toFixed(3) : "0.109 (I10)"}</div>
            <div class="viz-kpi-sub mono">${gpLatest ? `区间 [${gpLatest.low.toFixed(3)}–${gpLatest.high.toFixed(3)}]` : "Iter 60: 0.508 · Iter 150: 0.886"}</div>
          </div>
          <div class="viz-kpi-tile">
            <div class="viz-kpi-label">相对 GiGPO 论文差值</div>
            <div class="viz-kpi-val">${latestPt && gpLatest ? `${latestPt.sr >= gpLatest.mean ? "+" : ""}${(latestPt.sr - gpLatest.mean).toFixed(3)}` : "-"}</div>
            <div class="viz-kpi-sub mono">${latestPt && gpLatest ? fmtSignedDelta(latestPt.sr - gpLatest.mean) : "完成首轮后自动计算"}</div>
          </div>
        `;
      }

      const W = 960, H = 330, padL = 54, padR = 34, padT = 26, padB = 42;
      const plotW = W - padL - padR, plotH = H - padT - padB;
      const xToPx = (it) => Math.round(padL + (it / xMax) * plotW);
      const yToPx = (sr) => Math.round(padT + plotH - Math.max(0, Math.min(1, sr)) * plotH);
      gigpoVizState = { W, H, padL, padR, padT, padB, plotW, plotH, xMax, fgpoBudgetMax, xToPx, yToPx, primaryByIter, primaryLabel, hostAGigpoByIter };

      const xTicks = xMax > 100 ? [0, 25, 50, 75, 100, 125, 150] : [0, 10, 20, 30, 40, 50, 60];
      const yTicks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
      let svgHtml = "";

      yTicks.forEach((yv) => {
        const y = yToPx(yv);
        svgHtml += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--viz-grid)" stroke-width="1" />`;
        svgHtml += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" fill="var(--muted)" font-size="11" class="mono">${yv.toFixed(1)}</text>`;
      });
      xTicks.forEach((xv) => {
        if (xv > xMax) return;
        const x = xToPx(xv);
        svgHtml += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + plotH}" stroke="var(--viz-grid)" stroke-width="1" />`;
        svgHtml += `<text x="${x}" y="${H - 18}" text-anchor="middle" fill="var(--muted)" font-size="11" class="mono">${xv}</text>`;
      });
      svgHtml += `<line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="var(--border)" stroke-width="1" />`;
      svgHtml += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="var(--border)" stroke-width="1" />`;
      svgHtml += `<text x="${Math.round(padL + plotW / 2)}" y="${H - 3}" text-anchor="middle" fill="var(--muted)" font-size="11">Iteration / Round+1 (每步 = 128 轨迹)</text>`;

      if (xMax > fgpoBudgetMax) {
        const x60 = xToPx(fgpoBudgetMax);
        svgHtml += `<line x1="${x60}" y1="${padT}" x2="${x60}" y2="${padT + plotH}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4,4" opacity="0.7" />`;
        svgHtml += `<text x="${x60 + 6}" y="${padT + 14}" fill="var(--muted)" font-size="11">当前方法预算终点 (Iter ${fgpoBudgetMax})</text>`;
      }

      const gpVisible = GIGPO_CURVE.filter((p) => p[0] <= xMax);
      if (showBand && gpVisible.length >= 2) {
        const topPts = gpVisible.map((p) => `${xToPx(p[0])},${yToPx(p[3])}`);
        const botPts = gpVisible.slice().reverse().map((p) => `${xToPx(p[0])},${yToPx(p[2])}`);
        svgHtml += `<polygon points="${topPts.concat(botPts).join(" ")}" fill="var(--viz-gigpo)" fill-opacity="0.12" />`;
      }
      if (gpVisible.length >= 2) {
        const gpPath = gpVisible.map((p, i) => `${i === 0 ? "M" : "L"}${xToPx(p[0])},${yToPx(p[1])}`).join(" ");
        svgHtml += `<path d="${gpPath}" fill="none" stroke="var(--viz-gigpo)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />`;
      }

      GIGPO_MILESTONES.filter((ms) => ms.iter <= xMax).forEach((ms) => {
        const mx = xToPx(ms.iter);
        const my = yToPx(ms.mean);
        if (ms.shape === "square") {
          svgHtml += `<rect x="${mx - 4}" y="${my - 4}" width="8" height="8" rx="1" fill="var(--viz-gigpo)" stroke="var(--card)" stroke-width="2" />`;
        } else if (ms.shape === "triangle") {
          svgHtml += `<polygon points="${mx},${my - 6} ${mx + 5},${my + 4} ${mx - 5},${my + 4}" fill="var(--viz-gigpo)" stroke="var(--card)" stroke-width="2" />`;
        } else {
          svgHtml += `<circle cx="${mx}" cy="${my}" r="4" fill="var(--viz-gigpo)" stroke="var(--card)" stroke-width="2" />`;
        }
        const lblAnchor = ms.iter >= xMax * 0.85 ? "end" : "start";
        const lblDx = lblAnchor === "end" ? -9 : 9;
        const lblDy = ms.mean > 0.8 ? 16 : -10;
        svgHtml += `<text x="${mx + lblDx}" y="${my + lblDy}" text-anchor="${lblAnchor}" fill="var(--text)" font-size="11" class="mono">GiGPO I${ms.iter}: ${ms.mean.toFixed(3)}</text>`;
      });

      const haPts = Object.values(hostAGigpoByIter).filter((p) => p.iter <= xMax).sort((a, b) => a.iter - b.iter);
      if (haPts.length >= 2) {
        const haD = haPts.map((p, i) => `${i === 0 ? "M" : "L"}${xToPx(p.iter)},${yToPx(p.sr)}`).join(" ");
        svgHtml += `<path d="${haD}" fill="none" stroke="var(--viz-hosta)" stroke-width="2" />`;
      }
      haPts.forEach((p) => {
        svgHtml += `<circle cx="${xToPx(p.iter)}" cy="${yToPx(p.sr)}" r="4.5" fill="var(--viz-hosta)" stroke="var(--card)" stroke-width="2" />`;
      });

      if (seriesMode === "mean") {
        const startedSeeds = Object.keys(bySeed).filter((sk) => Object.keys(bySeed[sk].pts).length > 0);
        if (startedSeeds.length >= 2) {
          startedSeeds.forEach((sk) => {
            const sPts = Object.values(bySeed[sk].pts).filter((p) => p.iter <= xMax).sort((a, b) => a.iter - b.iter);
            if (sPts.length >= 2) {
              const d = sPts.map((p, i) => `${i === 0 ? "M" : "L"}${xToPx(p.iter)},${yToPx(p.sr)}`).join(" ");
              svgHtml += `<path d="${d}" fill="none" stroke="var(--viz-eval)" stroke-width="1.5" stroke-opacity="0.5" stroke-linecap="round" />`;
            }
          });
        }
      }

      const fgVis = primaryPoints.filter((p) => p.iter <= xMax);
      if (fgVis.length >= 2) {
        const fgPath = fgVis.map((p, i) => `${i === 0 ? "M" : "L"}${xToPx(p.iter)},${yToPx(p.sr)}`).join(" ");
        svgHtml += `<path d="${fgPath}" fill="none" stroke="var(--viz-fgpo)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />`;
      }
      fgVis.forEach((p) => {
        svgHtml += `<circle cx="${xToPx(p.iter)}" cy="${yToPx(p.sr)}" r="4" fill="var(--viz-fgpo)" stroke="var(--card)" stroke-width="2" />`;
      });

      if (fgVis.length) {
        const endP = fgVis[fgVis.length - 1];
        const ex = xToPx(endP.iter);
        const ey = yToPx(endP.sr);
        const anchor = endP.iter > xMax * 0.72 ? "end" : "start";
        const dx = anchor === "end" ? -10 : 10;
        const dy = ey < padT + 26 ? 16 : -10;
        svgHtml += `<text x="${ex + dx}" y="${ey + dy}" text-anchor="${anchor}" fill="var(--text)" font-size="12" font-weight="650" class="mono">${esc(primaryLabel)} R${endP.round} (I${endP.iter}): ${endP.sr.toFixed(3)}</text>`;
      }

      const evalTarget = seriesMode === "mean"
        ? (() => {
            const evs = Object.keys(bySeed).map((sk) => bySeed[sk].evalSr).filter((v) => v != null);
            return evs.length ? { sr: evs.reduce((a, b) => a + b, 0) / evs.length, label: `官方 Eval 均值 (${evs.length} seeds)` } : null;
          })()
        : ((bySeed[activeSeedKey] || {}).evalSr != null ? { sr: bySeed[activeSeedKey].evalSr, label: `官方 ${selMeta ? (selMeta.eval_metric_name || "SR") : "SR"}` } : null);
      if (evalTarget) {
        const evX = xToPx(Math.min(xMax, fgpoBudgetMax));
        const evY = yToPx(evalTarget.sr);
        svgHtml += `<polygon points="${evX},${evY - 7} ${evX + 7},${evY} ${evX},${evY + 7} ${evX - 7},${evY}" fill="var(--viz-eval)" stroke="var(--card)" stroke-width="2" />`;
        svgHtml += `<text x="${evX - 10}" y="${evY + 15}" text-anchor="end" fill="var(--text)" font-size="11.5" font-weight="650" class="mono">${esc(evalTarget.label)}: ${evalTarget.sr.toFixed(3)}</text>`;
      }

      svgHtml += `<line id="gigpo-crosshair" x1="0" y1="${padT}" x2="0" y2="${padT + plotH}" stroke="var(--text)" stroke-width="1" opacity="0.45" style="display:none" />`;
      svgHtml += `<circle id="gigpo-hi-gigpo" cx="0" cy="0" r="6" fill="none" stroke="var(--viz-gigpo)" stroke-width="2" style="display:none" />`;
      svgHtml += `<circle id="gigpo-hi-fgpo" cx="0" cy="0" r="6" fill="none" stroke="var(--viz-fgpo)" stroke-width="2" style="display:none" />`;
      svgHtml += `<rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" fill="transparent" style="cursor:crosshair" />`;

      const svgEl = document.getElementById("gigpo-compare-chart");
      if (svgEl) svgEl.innerHTML = svgHtml;

      const tblWrap = document.getElementById("gigpo-milestone-table-wrap");
      if (tblWrap) {
        const keyIters = new Set([1, 10, 25, 50, 60, 75, 100, 140, 150]);
        primaryPoints.forEach((p) => keyIters.add(p.iter));
        const sortedIters = Array.from(keyIters).sort((a, b) => a - b);
        let tHtml = `<table><thead><tr><th>对齐 Iter (Round+1)</th><th>累计轨迹</th><th>${esc(primaryLabel)} SR</th><th>GiGPO 1.5B 论文 SR</th><th>Host A 7B GiGPO (归档)</th><th>差值 (实测 − 论文)</th></tr></thead><tbody>`;
        sortedIters.forEach((it) => {
          const gp = GIGPO_BY_ITER[it];
          const fg = primaryByIter[it];
          const ha = hostAGigpoByIter[it];
          tHtml += `<tr class="mono"><td>Iter ${it}</td><td>${fmtInt(it * 128)}</td><td>${fg ? `${fg.sr.toFixed(3)} (${(fg.sr * 100).toFixed(1)}%)` : "-"}</td><td>${gp ? `${gp.mean.toFixed(3)} [${gp.low.toFixed(3)}–${gp.high.toFixed(3)}]` : "-"}</td><td>${ha ? `${ha.sr.toFixed(3)} (${ha.n} seeds)` : "-"}</td><td>${fg && gp ? fmtSignedDelta(fg.sr - gp.mean) : "-"}</td></tr>`;
        });
        tHtml += `</tbody></table>`;
        tblWrap.innerHTML = tHtml;
      }
    }

    function renderSingleAxisSvg(svgEl, values, labels, color, yMaxFixed) {
      if (!svgEl) return;
      const vb = (svgEl.getAttribute("viewBox") || "0 0 460 200").split(" ").map(Number);
      const W = vb[2] || 460, H = vb[3] || 200;
      const padL = 44, padR = 16, padT = 14, padB = 26;
      const plotW = W - padL - padR, plotH = H - padT - padB;
      const valid = values.filter((v) => v != null);
      if (!valid.length) {
        svgEl.innerHTML = `<text x="${Math.round(W / 2)}" y="${Math.round(H / 2)}" text-anchor="middle" fill="var(--muted)" font-size="12">暂无已完成采样点，等待数据写入...</text>`;
        return;
      }
      const yMax = yMaxFixed != null ? yMaxFixed : Math.max(1, ...valid) * 1.12;
      let html = "";
      for (let i = 0; i <= 4; i++) {
        const y = Math.round(padT + (plotH * i) / 4);
        const val = yMax - (yMax * i) / 4;
        html += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--viz-grid)" stroke-width="1" />`;
        html += `<text x="${padL - 6}" y="${y + 4}" text-anchor="end" fill="var(--muted)" font-size="11" class="mono">${val >= 100 ? Math.round(val) : val.toFixed(1)}</text>`;
      }
      const pts = [];
      values.forEach((v, idx) => {
        if (v == null) return;
        const x = values.length === 1 ? Math.round(padL + plotW / 2) : Math.round(padL + (plotW * idx) / (values.length - 1));
        const y = Math.round(padT + plotH - (Math.max(0, Math.min(yMax, v)) / yMax) * plotH);
        pts.push([x, y, v, labels[idx] || idx]);
      });
      if (pts.length >= 2) {
        const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ");
        html += `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" />`;
      }
      const endP = pts[pts.length - 1];
      if (endP) {
        html += `<circle cx="${endP[0]}" cy="${endP[1]}" r="4" fill="${color}" stroke="var(--card)" stroke-width="2" />`;
      }
      if (labels.length >= 1) {
        html += `<text x="${padL}" y="${H - 6}" fill="var(--muted)" font-size="11" class="mono">${esc(labels[0])}</text>`;
        if (labels.length > 1) {
          html += `<text x="${W - padR}" y="${H - 6}" text-anchor="end" fill="var(--muted)" font-size="11" class="mono">${esc(labels[labels.length - 1])}</text>`;
        }
      }
      svgEl.innerHTML = html;
    }

    function renderOverviewTab(data) {
      const cs = data.cluster_summary || {};
      const vramPct = cs.active_vram_total_mib ? ((cs.active_vram_used_mib / cs.active_vram_total_mib) * 100).toFixed(1) : "0.0";
      const pausedTxt = cs.cells_paused ? ` · ${cs.cells_paused} 暂存` : "";
      document.getElementById("cluster-kpi-row").innerHTML = `
        <div class="card">
          <div class="card-t">在线集群与通道</div>
          <div class="kpi">${cs.servers_online ?? 0}/${cs.servers_enabled ?? 3} 机 · ${cs.lanes_running ?? 0} 通道</div>
          <div class="kpi-sub mono">${cs.active_experiment_gpus ?? 0}/${cs.usable_pool_gpus ?? 9} 可用池 GPU (均值 ${cs.mean_active_gpu_util_pct ?? "-"}%) · ${cs.active_pids_count ?? 0} 进程</div>
        </div>
        <div class="card">
          <div class="card-t">已登记实验单元</div>
          <div class="kpi">${cs.cells_completed ?? 0} 完结 · ${cs.cells_running ?? 0} 运行</div>
          <div class="kpi-sub mono">${cs.cells_queued ?? 0} 已排队 · ${cs.cells_planned ?? 0} 计划${pausedTxt} / 共 ${cs.cells_total ?? 0} 单元</div>
        </div>
        <div class="card">
          <div class="card-t">活跃实验卡显存 (${cs.active_experiment_gpus ?? 0}卡)</div>
          <div class="kpi">${fmtNum(cs.active_vram_used_gib, 1)} / ${fmtNum(cs.active_vram_total_gib, 1)} GiB</div>
          <div class="kpi-sub mono">FGPO ${fmtInt(cs.fgpo_vram_used_mib)} MiB${cs.external_vram_used_mib ? ` + 外部共享 ${fmtInt(cs.external_vram_used_mib)} MiB` : ""} (${vramPct}%)</div>
        </div>
        <div class="card">
          <div class="card-t">当前单元累计环境交互步</div>
          <div class="kpi">${fmtInt(cs.env_steps_cum)}</div>
          <div class="kpi-sub mono">实时汇总已完成 + 运行中单元</div>
        </div>
        <div class="card">
          <div class="card-t">Token 吞吐与 Colab 配额</div>
          <div class="kpi">${fmtNum((cs.input_tokens_cum || 0) / 1e6, 2)}M / ${fmtNum((cs.output_tokens_cum || 0) / 1e6, 2)}M</div>
          <div class="kpi-sub mono">Colab: ${esc(cs.colab_balance || "-")} (${esc(cs.colab_rate || "-")})</div>
        </div>
      `;

      const lanes = (data.lanes || []).filter((ln) => scopeServer === "all" || ln.server_id === scopeServer);
      document.getElementById("lanes-grid").innerHTML = lanes.map((ln) => {
        if (!ln.active_cell) {
          const cards = (ln.gpus || []).map((g) => `GPU ${g.index} · ${fmtInt(g.mem_used_mib)}/${fmtInt(g.mem_total_mib)} MiB`).join(" · ");
          return `<div class="card-sub"><div class="card-h"><b>${esc(ln.lane_title)}</b></div>
            <p>${ln.server_id === "host_a" ? "用于 TextCraft，等正式训练与终评接线及卡位分配。" : "暂无已登记单元。"}</p>
            <div class="kpi-sub mono">${esc(cards || "等待硬件读数")}</div></div>`;
        }
        const ac = ln.active_cell || {};
        const color = LANE_COLORS[ln.lane_id] || "var(--c1)";
        const totalGrp = (ac.total_rounds || 60) * 16;
        const doneGrp = (ac.rounds_done || 0) * 16 + (ac.current_round_groups_done || 0);
        const pct = Math.min(100, (doneGrp / Math.max(1, totalGrp)) * 100);
        let segs = "";
        for (let g = 0; g < 16; g++) {
          const gd = ac.current_round_groups_done || 0;
          const bg = g < gd ? color : (g === gd && ac.live_status === "running" ? "var(--warn)" : "var(--card-2)");
          segs += `<div class="grp-seg" style="background:${bg}"></div>`;
        }
        const gpusTxt = (ln.gpus || []).map((g) => {
          const extPart = g.external_mem_used_mib ? ` [FGPO ${fmtInt(g.fgpo_mem_used_mib)} + 外部 ${fmtInt(g.external_mem_used_mib)} MiB]` : "";
          return `GPU${g.index} ${g.util_gpu_pct ?? "-"}% (${fmtInt(g.mem_used_mib)}/${fmtInt(g.mem_total_mib)} MiB${extPart})`;
        }).join(" · ") || "等待采样";

        const compList = ln.completed_cells || [];
        const compHtml = compList.length
          ? `<div class="kpi-sub mono" style="margin-bottom:4px;color:var(--good);">已完成 (${compList.length}): ${compList.map((cc) => {
              let evTxt = cc.eval_sr_pct != null ? ` ${esc(cc.eval_metric_name || "SR")} ${cc.eval_sr_pct}%` : "";
              if (cc.eval_valid_seen_sr_pct != null) evTxt += ` · seen ${cc.eval_valid_seen_sr_pct}%`;
              return `${esc(cc.method)} s${cc.seed}${evTxt ? ` (${evTxt.trim()})` : ""}`;
            }).join(" · ")}</div>`
          : "";

        const pausedList = ln.paused_cells || [];
        const pausedHtml = pausedList.length
          ? `<div class="kpi-sub mono" style="margin-bottom:4px;color:var(--warn);">已暂存待恢复 (${pausedList.length}): ${pausedList.map((pc) => `${esc(pc.method)} s${pc.seed} (R${pc.rounds_done}/${pc.total_rounds} G${pc.current_round_groups_done}/16)`).join(" · ")}</div>`
          : "";

        const qNames = (ln.queued_cells || []).map((q) => `${q.benchmark === ac.benchmark ? "" : q.benchmark + ":"}${q.method} s${q.seed}${q.staged_handoff ? "(待恢复)" : ""}`).join(" → ") || "无后续排队";
        const planned = (ln.planned_cells || []).map((c) => `${c.method} s${c.seed}`).join(" → ");
        const handoff = ln.gpu2_handoff || {};
        const firstUpdate = handoff.first_successful_update;
        let handoffLabel = "";
        if (ln.handoff_pending) handoffLabel = "等待当前组保存 · F-GPO 优先续跑";
        else if (firstUpdate) handoffLabel = `G2PO 已完成首次更新 · 事件 ${firstUpdate.event_index ?? "-"}`;
        else if (handoff.g2po_runner_pid) handoffLabel = "F-GPO 终评完成 · G2PO 正在运行";
        else if (handoff.fgpo_resume_pid) handoffLabel = "F-GPO 已从检查点续跑";
        else if (handoff.fgpo_saved_snapshot) handoffLabel = "F-GPO 已保存 · 等待续跑";
        const handoffBadge = handoffLabel ? `<span class="badge">${esc(handoffLabel)}</span>` : "";

        return `
          <div class="card-sub" style="display:flex;flex-direction:column;justify-content:space-between;gap:8px;">
            <div>
              <div class="card-h" style="margin-bottom:4px;">
                <span style="font-weight:650;display:inline-flex;align-items:center;gap:6px;">
                  <span class="swatch" style="background:${color}"></span>${esc(ln.lane_title)}
                </span>
                <span style="display:inline-flex;gap:4px;align-items:center;">
                  ${handoffBadge}
                  <span class="badge"><span class="dot ${ac.live_status === "running" || ac.live_status === "completed" ? "good" : "warn"}"></span><span>${esc(ac.live_status || "-")}</span></span>
                </span>
              </div>
              <div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;">
                <span class="mono" style="font-size:12px;color:var(--accent);font-weight:600;">
                  ${esc(ac.suite || "-")} · ${esc(ac.benchmark || "-")} · ${esc(ac.method || "-")} · seed ${ac.seed ?? "-"}
                </span>
                ${ac.sampling_layout ? `<span class="pill-tag mono">${esc(ac.sampling_layout)} · H=${ac.horizon ?? "-"}</span>` : ""}
              </div>
              <div class="card-h mono" style="margin:6px 0 2px;font-size:12px;">
                <span>Rounds: ${ac.rounds_done ?? 0} / ${ac.total_rounds ?? "-"} (${pct.toFixed(1)}%)</span>
                <span>Group: ${ac.current_round_groups_done ?? 0}/16</span>
              </div>
              <div class="bar-bg" style="margin-top:2px;"><div class="bar-fill" style="width:${Math.max(2, Math.round(pct))}%;background:${color}"></div></div>
              <div class="grp-seg-row">${segs}</div>
              <div class="kpi-sub mono" style="margin-top:8px;line-height:1.55;">
                阶段: <b>${esc(ac.last_phase_role || "初始化/采集中")}</b> (${esc(ac.last_group_id || "-")})<br/>
                最新训练 SR: <b>${ac.latest_sr_pct != null ? `${ac.latest_sr_pct}% (${ac.latest_successes}/${ac.latest_trajectories})` : "暂无完整轮"}</b> · Opt ${ac.optimizer_steps ?? 0} · Loss/KL: ${ac.latest_loss != null ? `${fmtNum(ac.latest_loss, 1)} / ${fmtNum(ac.latest_kl, 1)}` : "-"}<br/>
                Env Steps: <b>${fmtInt(ac.env_steps)}</b> · Tok: ${fmtNum((ac.input_tokens || 0) / 1e6, 2)}M in / ${fmtNum((ac.output_tokens || 0) / 1e6, 2)}M out<br/>
                耗时: ${fmtDur(ac.session_elapsed_seconds)} · Runner PID: ${esc(ln.runner_pid || "-")} (${ln.env_worker_count || 0} envs)<br/>
                GPU: ${esc(gpusTxt)}<br/>
                亲和性: ${esc(ln.cpu_affinity)}
              </div>
            </div>
            <div style="border-top:1px solid var(--border);padding-top:6px;">
              ${compHtml}
              ${pausedHtml}
              <div class="kpi-sub mono" style="margin-bottom:6px;">后续排队 (${(ln.queued_cells || []).length}): ${esc(qNames)}</div>
              ${planned ? `<div class="kpi-sub">计划接续 ${esc(planned)}</div>` : ""}
              <div class="controls">
                <button class="btn-link" onclick="window._inspectCell('${esc(ac.global_cell_id || "")}', 'experiment')">查看曲线对比 →</button>
                <button class="btn-link" onclick="window._inspectCell('${esc(ac.global_cell_id || "")}', 'costs')">成本与事件 →</button>
              </div>
            </div>
          </div>
        `;
      }).join("");

      const srvOrder = ["colab", "host_a", "host_b"].filter((k) => scopeServer === "all" || scopeServer === k);
      const hwStack = document.getElementById("servers-hw-stack");
      hwStack.innerHTML = srvOrder.map((sid) => {
        const srv = (data.servers || {})[sid] || {};
        const tel = srv.telemetry || {};
        const mem = tel.mem || {};
        const ramPct = mem.total_mib ? Math.round((mem.used_mib / mem.total_mib) * 100) : 0;
        const gpus = tel.gpus || [];
        const excl = tel.excluded_gpus || [];
        const disks = tel.disks || [];
        const dbk = tel.disk_breakdown || {};

        const gpuHtml = gpus.map((g) => {
          const vp = g.mem_total_mib ? Math.round((g.mem_used_mib / g.mem_total_mib) * 100) : 0;
          const capMark = g.mem_cap_gib && g.mem_total_mib ? `<div style="position:absolute;top:0;bottom:0;left:${Math.round((g.mem_cap_gib * 1024 / g.mem_total_mib) * 100)}%;width:2px;background:var(--warn);"></div>` : "";
          const splitVram = g.external_mem_used_mib > 0
            ? ` · FGPO ${fmtInt(g.fgpo_mem_used_mib)} + 外部共享 ${fmtInt(g.external_mem_used_mib)} MiB`
            : (g.fgpo_mem_used_mib > 0 && g.fgpo_mem_used_mib !== g.mem_used_mib ? ` · FGPO ${fmtInt(g.fgpo_mem_used_mib)} MiB` : "");
          return `
            <div class="card-sub mono" style="font-size:12px;">
              <div class="card-h" style="margin-bottom:3px;">
                <span><b>GPU ${g.index}</b> · ${esc(g.name)}</span>
                <span class="badge"><span class="dot ${g.active_in_experiment ? "good" : "warn"}"></span><span>${esc(g.role_slot || "")}</span></span>
              </div>
              <div>算力: <b>${g.util_gpu_pct ?? "-"}%</b> (Mem ${g.util_mem_pct ?? "-"}%) · ${g.temp_c ?? "-"}°C · ${fmtNum(g.power_draw_w, 0)}/${fmtNum(g.power_limit_w, 0)}W · PIDs: ${esc((g.pids || []).join(",") || "none")}</div>
              <div style="margin-top:2px;">VRAM: ${fmtInt(g.mem_used_mib)} / ${fmtInt(g.mem_total_mib)} MiB (${vp}%)${g.mem_cap_gib ? ` · Cap ${g.mem_cap_gib} GiB` : ""}${splitVram}</div>
              <div class="bar-bg">${capMark}<div class="bar-fill" style="width:${vp}%;background:${g.active_in_experiment ? "var(--c1)" : "var(--muted)"}"></div></div>
            </div>
          `;
        }).join("");

        const diskHtml = disks.map((d) => {
          const dp = Math.round(d.used_pct_num || 0);
          const barCol = dp >= 95 ? "var(--bad)" : (dp >= 88 ? "var(--warn)" : "var(--c2)");
          return `
            <div class="card-sub mono" style="font-size:12px;">
              <div class="card-h" style="margin-bottom:3px;">
                <span><b>${esc(d.mount)}</b> (${esc(d.filesystem)})</span>
                <span>${d.used_gib} / ${d.total_gib} GiB (剩 <b>${d.avail_gib} GiB</b> · ${esc(d.use_pct)})</span>
              </div>
              <div class="bar-bg"><div class="bar-fill" style="width:${dp}%;background:${barCol}"></div></div>
              ${sid === "colab" && dbk.content_visible_gib != null ? `<div class="kpi-sub" style="margin-top:5px;">可见文件 ${dbk.content_visible_gib} GiB + 已删除未释放 mmap <b>${dbk.pinned_deleted_mmap_gib} GiB</b> (${fmtInt(dbk.pinned_deleted_count)} inodes) · libdownward ${dbk.libdownward_mib ?? "-"} MiB · 归档包 ${dbk.history_bundles_count ?? 0} 个</div>` : ""}
            </div>
          `;
        }).join("");

        const exclNote = excl.length ? `<div class="kpi-sub mono" style="margin-top:8px;">排除非本实验物理卡: ${excl.map((e) => `GPU ${e.index} [${e.util_gpu_pct}% util, ${fmtInt(e.mem_used_mib)}/${fmtInt(e.mem_total_mib)} MiB]`).join(" · ")}</div>` : "";

        return `
          <div class="card">
            <div class="card-h">
              <div>
                <span style="font-size:15px;font-weight:650;">${esc(srv.display_name || sid)}</span>
                <span class="kpi-sub mono" style="margin-left:8px;">Host: ${esc(tel.hostname || "-")} · ${tel.cpu_count ?? "-"} CPUs · Load ${(tel.loadavg || []).map((x) => Number(x).toFixed(2)).join("/")} · RAM ${fmtNum((mem.used_mib || 0) / 1024, 1)}/${fmtNum((mem.total_mib || 0) / 1024, 1)} GiB (${ramPct}%) · UTC ${esc(tel.timestamp_utc || "-")}</span>
              </div>
              <button class="btn btn-sm" onclick="window._refreshServer('${sid}')">单机刷新 ↻</button>
            </div>
            <div class="grid-3" style="margin-bottom:10px;">
              <div style="grid-column: span 2; display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:10px;">${gpuHtml || `<div class="kpi-sub">等待 GPU 数据...</div>`}</div>
              <div class="card-sub">
                <div class="card-h" style="margin-bottom:4px;"><span class="card-t">实时采样历史 (GPU% / VRAM%)</span><span class="kpi-sub mono">${(srv.history || []).length} pts</span></div>
                <svg id="hist-svg-${sid}" class="chart" viewBox="0 0 380 125"></svg>
              </div>
            </div>
            <div class="grid-2" style="margin-bottom:0;">${diskHtml}</div>
            ${exclNote}
          </div>
        `;
      }).join("");

      srvOrder.forEach((sid) => {
        const srv = (data.servers || {})[sid] || {};
        const hist = srv.history || [];
        const svg = document.getElementById("hist-svg-" + sid);
        if (!svg) return;
        if (!hist.length) {
          svg.innerHTML = `<text x="190" y="65" text-anchor="middle" fill="var(--muted)" font-size="11">等待采样点...</text>`;
          return;
        }
        const W = 380, H = 125, padL = 30, padR = 10, padT = 10, padB = 18;
        const plotW = W - padL - padR, plotH = H - padT - padB;
        let h = "";
        [0, 50, 100].forEach((yv) => {
          const y = Math.round(padT + plotH - (yv / 100) * plotH);
          h += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--viz-grid)" stroke-width="1" />`;
          h += `<text x="${padL - 4}" y="${y + 4}" text-anchor="end" fill="var(--muted)" font-size="10" class="mono">${yv}%</text>`;
        });
        const drawLine = (arr, col) => {
          const pts = [];
          arr.forEach((v, i) => {
            if (v == null) return;
            const x = arr.length === 1 ? Math.round(padL + plotW / 2) : Math.round(padL + (i / (arr.length - 1)) * plotW);
            const y = Math.round(padT + plotH - (Math.max(0, Math.min(100, v)) / 100) * plotH);
            pts.push(`${x},${y}`);
          });
          if (pts.length >= 2) h += `<polyline points="${pts.join(" ")}" fill="none" stroke="${col}" stroke-width="2" />`;
          else if (pts.length === 1) {
            const [cx, cy] = pts[0].split(",");
            h += `<circle cx="${cx}" cy="${cy}" r="3.5" fill="${col}" />`;
          }
        };
        drawLine(hist.map((x) => x.gpu_util), "var(--c1)");
        drawLine(hist.map((x) => x.vram_pct), "var(--c2)");
        svg.innerHTML = h;
      });
    }

    function renderExperimentTab(data) {
      const cells = data.schedule_18 || data.schedule_21 || [];
      const selEl = document.getElementById("exp-cell-select");
      if (selEl) {
        selEl.innerHTML = cells.map((c) =>
          `<option value="${esc(c.global_cell_id)}" ${c.global_cell_id === selectedGlobalCell ? "selected" : ""}>[${esc(c.live_status.toUpperCase())}] ${esc(c.lane_title)} · ${esc(c.suite)}/${esc(c.benchmark)}/${esc(c.method)}/s${c.seed} (R${c.rounds_done}/${c.total_rounds})</option>`
        ).join("");
      }
      const activeOrExisting = cells.filter((c) => c.exists || c.live_status !== "queued");
      document.getElementById("exp-quick-pills").innerHTML = activeOrExisting.map((c) =>
        `<button class="btn btn-sm ${c.global_cell_id === selectedGlobalCell ? "btn-primary" : ""}" onclick="window._inspectCell('${esc(c.global_cell_id)}', 'experiment')">${esc(c.server_id.toUpperCase())} · ${esc(c.benchmark)} · ${esc(c.method)} s${c.seed} (R${c.rounds_done}/${c.total_rounds})</button>`
      ).join("");

      const selMeta = cells.find((c) => c.global_cell_id === selectedGlobalCell) || cells[0];
      const selRaw = selMeta ? (getRawCellFromData(data, selMeta.server_id, selMeta.cell_key) || {}) : {};
      const rounds = selRaw.training_rounds || [];
      const csum = selRaw.costs_summary || {};

      document.getElementById("seed-detail-summary").innerHTML = selMeta ? `
        <div class="mono" style="font-size:12px;">
          通道: <b>${esc(selMeta.lane_title)}</b> (${esc(selMeta.sampling_layout || "-")}) | 状态: <b>${esc(selMeta.live_status)}</b> (R${selMeta.rounds_done}/${selMeta.total_rounds} · G${selMeta.current_round_groups_done}/16) |
          阶段: ${esc(csum.last_phase_role || "-")} | Last Group: ${esc(csum.last_group_id || "-")} |
          Env Steps: <b>${fmtInt(csum.env_steps)}</b> | 耗时: ${fmtDur(csum.session_elapsed_seconds)}
        </div>
      ` : "";

      const isAlfworld = selMeta?.benchmark === "alfworld";
      document.getElementById("alfworld-gigpo-comparison").classList.toggle("hidden", !isAlfworld);
      if (isAlfworld) renderGigpoComparison(data, selMeta, selRaw);

      const rLabels = rounds.map((r) => `R${r.round}`);
      document.getElementById("sr-meta").textContent = `${rounds.length} / ${selMeta ? selMeta.total_rounds : 60} rounds`;
      renderSingleAxisSvg(
        document.getElementById("sr-chart"),
        rounds.map((r) => r.trajectories ? Number(((100 * r.successes) / r.trajectories).toFixed(2)) : 0),
        rLabels, "var(--c2)", 100
      );
      document.getElementById("loss-meta").textContent = `${rounds.length} rounds`;
      renderSingleAxisSvg(
        document.getElementById("loss-chart"),
        rounds.map((r) => (r.metrics || {}).loss),
        rLabels, "var(--c3)", null
      );
      document.getElementById("kl-meta").textContent = `${rounds.length} rounds`;
      renderSingleAxisSvg(
        document.getElementById("kl-chart"),
        rounds.map((r) => (r.metrics || {}).reference_kl),
        rLabels, "var(--c4)", null
      );

      let tbl = "";
      const evals = selMeta ? (selMeta.evaluations || selRaw.evaluations || {}) : {};
      const evBadges = [];
      if (selRaw.cell_summary && selRaw.cell_summary.evaluation) {
        Object.entries(selRaw.cell_summary.evaluation).forEach(([sp, ev]) => {
          const scoreVal = ev.Score != null ? ev.Score : ev.score;
          const metricTxt = ev.SR != null
            ? `SR=${fmtNum(ev.SR, 2)}% (${ev.success_count ?? "-"} / ${ev.evaluated_count ?? "-"})${scoreVal != null ? ` · Score=${fmtNum(scoreVal, 2)}` : ""}`
            : `TGC=${fmtNum(ev.TGC, 2)}%${ev.SGC != null ? ` · SGC=${fmtNum(ev.SGC, 2)}%` : ""}${ev.evaluated_count != null ? ` (${ev.evaluated_count} tasks)` : ""}`;
          evBadges.push(`<span class="badge"><span class="dot good"></span><span class="mono">官方 ${esc(sp)}: ${metricTxt}</span></span>`);
        });
      }
      if (evals.valid_seen && evals.valid_seen.summary) {
        const vs = evals.valid_seen.summary;
        evBadges.push(`<span class="badge"><span class="dot good"></span><span class="mono">官方 valid_seen 回放: SR=${fmtNum(vs.SR, 2)}% (${vs.success_count ?? "-"}/${vs.evaluated_count ?? "-"}, ${vs.unique_task_count ?? "-"} unique tasks)</span></span>`);
      }
      Object.entries(evals).forEach(([sp, ev]) => {
        if (sp === "valid_seen") return;
        if (!(selRaw.cell_summary && selRaw.cell_summary.evaluation && selRaw.cell_summary.evaluation[sp]) && ev.completed_count != null) {
          evBadges.push(`<span class="badge"><span class="dot warn"></span><span class="mono">评估进行中 ${esc(sp)}: ${ev.completed_count} tasks (succ ${ev.success_count ?? 0}${ev.tgc_pct != null ? `, TGC ${ev.tgc_pct}%` : ""}${ev.score_pct != null ? `, Score ${ev.score_pct}%` : ""})</span></span>`);
        }
      });
      if (evBadges.length) tbl += `<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;">${evBadges.join("")}</div>`;

      if (rounds.length) {
        tbl += `<table><thead><tr><th>Round</th><th>Iter</th><th>Success/Traj</th><th>SR</th>${isAlfworld ? "<th>GiGPO 1.5B</th><th>Δ vs GiGPO</th>" : ""}<th>Opt Steps</th><th>Loss</th><th>Ref KL</th><th>Entropy</th><th>Grad Norm</th><th>Zero/Mixed Grps</th></tr></thead><tbody>`;
        rounds.slice().reverse().forEach((r) => {
          const m = r.metrics || {};
          const dg = r.diagnostics || {};
          const srFrac = r.trajectories ? r.successes / r.trajectories : 0;
          const it = Number(r.round) + 1;
          const gp = isAlfworld ? GIGPO_BY_ITER[it] : null;
          const deltaTxt = gp ? `${srFrac >= gp.mean ? "+" : ""}${(srFrac - gp.mean).toFixed(3)}` : "-";
          const grpDiag = dg.all_zero_groups != null ? `zero:${dg.all_zero_groups}` : (dg.mixed_groups != null ? `mixed:${dg.mixed_groups}` : "-");
          const baselineCols = isAlfworld ? `<td>${gp ? gp.mean.toFixed(3) : "-"}</td><td>${deltaTxt}</td>` : "";
          tbl += `<tr class="mono"><td>${r.round}</td><td>I${it}</td><td>${r.successes}/${r.trajectories}</td><td>${srFrac.toFixed(3)} (${(srFrac * 100).toFixed(1)}%)</td>${baselineCols}<td>${r.optimizer_steps ?? "-"}</td><td>${fmtNum(m.loss, 4)}</td><td>${fmtNum(m.reference_kl, 4)}</td><td>${fmtNum(m.entropy, 4)}</td><td>${fmtNum(m.gradient_norm, 3)}</td><td>${grpDiag}</td></tr>`;
        });
        tbl += `</tbody></table>`;
      } else {
        tbl += `<div class="kpi-sub">当前选中单元暂无完整轮次。</div>`;
      }
      document.getElementById("rounds-table-wrap").innerHTML = tbl;
    }

    function renderScheduleTab(data) {
      const cells = data.schedule_18 || data.schedule_21 || [];
      const methods = [...new Set(cells.map((c) => c.method))];
      document.getElementById("gantt-legend").innerHTML = methods.map((m) =>
        `<span class="legend-item"><span class="swatch" style="background:${METHOD_COLORS[m] || "var(--c1)"}"></span>${esc(m)}</span>`).join("");

      const svg = document.getElementById("gantt-svg");
      const W = 1040, padL = 200, padR = 32, padT = 24, rowH = 36, padB = 30;
      const laneOrder = (data.lanes || []).map((lane) => ({
        id: lane.lane_id,
        label: lane.server_id === "colab" ? "Colab" : `${lane.server_id === "host_a" ? "Host A" : "Host B"} ${lane.short_lane}`,
        total: cells.filter((c) => c.lane_id === lane.lane_id).reduce((sum, c) => sum + c.total_rounds, 0)
      }));
      const maxRoundsAxis = Math.max(60, Math.ceil(Math.max(0, ...laneOrder.map((l) => l.total)) / 60) * 60);
      svg.setAttribute("viewBox", `0 0 ${W} ${padT + laneOrder.length * rowH + padB}`);
      const plotW = W - padL - padR;
      const xOfR = (r) => Math.round(padL + (Math.max(0, Math.min(maxRoundsAxis, r)) / maxRoundsAxis) * plotW);

      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const ns = "http://www.w3.org/2000/svg";
      const mk = (tag, attrs) => {
        const n = document.createElementNS(ns, tag);
        for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, String(v));
        return n;
      };

      Array.from({length: maxRoundsAxis / 60 + 1}, (_, i) => i * 60).forEach((rt) => {
        const x = xOfR(rt);
        svg.appendChild(mk("line", { x1: x, y1: padT - 4, x2: x, y2: padT + laneOrder.length * rowH, stroke: "var(--viz-grid)", "stroke-width": 1 }));
        const tx = mk("text", { x, y: padT + laneOrder.length * rowH + 18, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11, class: "mono" });
        tx.textContent = rt + "r";
        svg.appendChild(tx);
      });

      laneOrder.forEach((ln, idx) => {
        const yRow = padT + idx * rowH;
        const barY = yRow + 6, barH = 24;
        const lbl = mk("text", { x: padL - 10, y: yRow + 22, "text-anchor": "end", fill: "var(--text)", "font-size": 12, "font-weight": "500" });
        lbl.textContent = ln.label;
        svg.appendChild(lbl);
        svg.appendChild(mk("rect", { x: padL, y: barY, width: plotW, height: barH, rx: 4, fill: "var(--card-2)" }));

        const lCells = cells.filter((c) => c.lane_id === ln.id).sort((a, b) => a.lane_order - b.lane_order);
        let curRound = 0;
        lCells.forEach((c) => {
          const x1 = xOfR(curRound);
          const x2 = xOfR(curRound + c.total_rounds);
          curRound += c.total_rounds;
          const w = Math.max(8, x2 - x1 - 3);
          const col = METHOD_COLORS[c.method] || "var(--c1)";
          const isComp = c.live_status === "completed";
          const isRun = c.live_status === "running" || c.live_status === "evaluating";
          const isPaused = c.live_status === "paused";
          const frac = isComp ? 1.0 : Math.min(1.0, ((c.rounds_done || 0) + (c.current_round_groups_done || 0) / 16.0) / Math.max(1, c.total_rounds));

          const g = mk("g", { style: "cursor:pointer;" });
          // Base cell container (high-contrast light & dark)
          const baseRect = mk("rect", {
            x: x1, y: barY, width: w, height: barH, rx: 4,
            fill: col,
            "fill-opacity": isComp ? "0.90" : "0.20",
            stroke: c.handoff_pending ? "var(--warn)" : col,
            "stroke-width": isRun ? 2 : 1,
            "stroke-dasharray": isPaused ? "4,2" : null
          });
          g.appendChild(baseRect);

          // Live progress fill inside running/paused cell
          if (!isComp && frac > 0) {
            const fillW = Math.max(4, Math.round(w * frac));
            g.appendChild(mk("rect", {
              x: x1, y: barY, width: fillW, height: barH, rx: 3,
              fill: col, "fill-opacity": "0.85"
            }));
          }

          if ((isComp || isRun) && w >= 20) {
            g.appendChild(mk("circle", { cx: x1 + 8, cy: barY + 12, r: 3, fill: isComp ? "#ffffff" : "var(--warn)" }));
          }

          const shortBm = ({alfworld: "ALF", webshop: "WS", textcraft: "TC", appworld: "AW"})[c.benchmark] || c.benchmark;
          const shortM = c.method.replace("fgpo_no_transfer", "no_tr").replace("fgpo_obs_only", "obs");
          if (w >= 72) {
            const stxt = mk("text", {
              x: x1 + ((isComp || isRun) ? 15 : 6), y: barY + 16,
              fill: isComp ? "#ffffff" : "var(--text)",
              "paint-order": isComp ? "stroke" : null,
              stroke: isComp ? "rgba(0,0,0,0.45)" : null,
              "stroke-width": isComp ? 2 : null,
              "font-size": 11, "font-weight": "600"
            });
            stxt.textContent = `${shortBm}:${shortM} s${c.seed}`;
            g.appendChild(stxt);
          }

          g.addEventListener("pointermove", (ev) => {
            showFloatTip(ev, `${c.lane_title} · Order #${c.lane_order} [${c.live_status.toUpperCase()}]`, [
              { label: "Cell", value: `${c.suite}/${c.benchmark}/${c.method}/s${c.seed}`, color: col },
              { label: "Progress", value: `R${c.rounds_done}/${c.total_rounds} · G${c.current_round_groups_done}/16 (${fmtInt(c.total_training_trajectories)} traj)` },
              { label: "Layout / Horizon", value: `${c.sampling_layout || "-"} · H=${c.horizon}` },
              { label: "Official Eval", value: `${c.official_eval || "-"} (${c.official_endpoint_tasks} tasks)` },
              { label: "Dependency", value: c.dependency || "-" }
            ]);
          });
          g.addEventListener("pointerleave", hideFloatTip);
          g.addEventListener("click", () => window._inspectCell(c.global_cell_id, "experiment"));
          svg.appendChild(g);
        });
      });

      const fSrv = document.getElementById("sched-srv-filter").value;
      const fBen = document.getElementById("sched-bench-filter").value;
      const filtered = cells.filter((c) => (fSrv === "all" || c.server_id === fSrv) && (fBen === "all" || c.benchmark === fBen));
      document.getElementById("sched-table-wrap").innerHTML = `
        <table><thead><tr><th>#</th><th>通道 (Order)</th><th>Suite / Benchmark</th><th>Method · Seed</th><th>Sampling Layout · H</th><th>实时状态与进度</th><th>最新 / 官方指标</th><th>Env Steps</th><th>前置依赖 (Dependency)</th><th>操作</th></tr></thead><tbody>
        ${filtered.map((c) => {
          let metricCell = "-";
          if (c.eval_sr_pct != null) {
            metricCell = `<b>${esc(c.eval_metric_name || "SR")} ${c.eval_sr_pct}${c.eval_metric_name === "Score" ? "/100" : "%"}</b>${c.eval_valid_seen_sr_pct != null ? ` · seen ${c.eval_valid_seen_sr_pct}%` : ""}${c.eval_score_pct != null && c.eval_metric_name !== "Score" ? ` · Score ${c.eval_score_pct}/100` : ""}`;
          } else if (c.latest_sr_pct != null) {
            metricCell = `Train ${c.latest_sr_pct}%`;
          }
          const stLabel = c.handoff_pending ? "running (handoff)" : (c.staged_handoff ? "queued (staged)" : c.live_status);
          return `<tr class="mono">
            <td>${c.index}</td>
            <td style="font-family:inherit;font-weight:500;">${esc(c.lane_title)} <span class="pill-tag">#${c.lane_order}</span></td>
            <td>${esc(c.suite)} / <b>${esc(c.benchmark)}</b></td>
            <td><span class="swatch" style="background:${METHOD_COLORS[c.method]}"></span> ${esc(c.method)} · s${c.seed}</td>
            <td>${esc(c.sampling_layout || "-")} · H=${c.horizon}</td>
            <td><span class="badge"><span class="dot ${c.live_status === "running" || c.live_status === "completed" ? "good" : "warn"}"></span><span>${esc(stLabel)} R${c.rounds_done}/${c.total_rounds} G${c.current_round_groups_done}/16</span></span></td>
            <td>${metricCell}</td>
            <td>${fmtInt(c.env_steps)}</td>
            <td style="font-family:inherit;color:var(--muted);max-width:260px;">${esc(c.dependency || "-")}</td>
            <td><button class="btn-link" onclick="window._inspectCell('${esc(c.global_cell_id)}', 'experiment')">曲线</button></td>
          </tr>`;
        }).join("")}
        </tbody></table>
      `;

      // Archived Host A GiGPO table
      const archMap = (((data.servers || {}).host_a || {}).telemetry || {}).archived_cells || {};
      const archEntries = Object.entries(archMap);
      document.getElementById("archived-table-wrap").innerHTML = archEntries.length ? `
        <table><thead><tr><th>Archived Cell</th><th>Completed Rounds</th><th>Latest R0 SR</th><th>Cum Opt Steps</th><th>Env Steps</th><th>Elapsed</th><th>Saved Checkpoint Snapshot</th></tr></thead><tbody>
        ${archEntries.map(([ak, av]) => {
          const rds = av.training_rounds || [];
          const lr = rds.length ? rds[rds.length - 1] : null;
          const srTxt = lr && lr.trajectories ? `${((100 * lr.successes) / lr.trajectories).toFixed(2)}% (${lr.successes}/${lr.trajectories})` : "-";
          const cs = av.costs_summary || {};
          const ckSnap = (av.checkpoint_current || {}).snapshot || "-";
          return `<tr class="mono">
            <td>Host A · ${esc(ak)}</td>
            <td>${rds.length} / 150 (stopped_archived)</td>
            <td>${srTxt}</td>
            <td>${cs.successful_adamw_count ?? (lr ? lr.optimizer_steps : "-")}</td>
            <td>${fmtInt(cs.env_steps)}</td>
            <td>${fmtDur(cs.session_elapsed_seconds)}</td>
            <td>${esc(ckSnap)}</td>
          </tr>`;
        }).join("")}
        </tbody></table>
      ` : `<div class="kpi-sub">等待 Host A 远端探测返回归档单元信息...</div>`;
    }

    function renderCostsTab(data) {
      const cells = data.schedule_18 || data.schedule_21 || [];
      const selEl = document.getElementById("cost-cell-select");
      if (selEl) {
        selEl.innerHTML = cells.map((c) =>
          `<option value="${esc(c.global_cell_id)}" ${c.global_cell_id === selectedGlobalCell ? "selected" : ""}>[${esc(c.live_status.toUpperCase())}] ${esc(c.lane_title)} · ${esc(c.benchmark)}/${esc(c.method)} s${c.seed}</option>`
        ).join("");
      }

      document.getElementById("stage-legend").innerHTML = STAGE_META.map((m) =>
        `<span class="legend-item"><span class="swatch" style="background:${m.color}"></span>${esc(m.label)}</span>`
      ).join("");

      const lanes = data.lanes || [];
      const laneStats = lanes.map((ln) => {
        const ac = ln.active_cell || {};
        const rc = ac.cell_key ? (getRawCellFromData(data, ln.server_id, ac.cell_key) || {}) : {};
        const rs = (rc.costs_summary || {}).role_seconds || {};
        const cats = categorizeRoleSeconds(rs);
        const sumSec = Object.values(cats).reduce((a, b) => a + b, 0);
        return { ln, ac, cats, sumSec };
      });
      const maxSec = Math.max(100, ...laneStats.map((x) => x.sumSec));
      const xDomain = stageNormMode === "pct" ? 100 : Math.ceil(maxSec / 5000) * 5000;

      const svg = document.getElementById("stage-svg");
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const ns = "http://www.w3.org/2000/svg";
      const mk = (tag, attrs) => {
        const n = document.createElementNS(ns, tag);
        for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, String(v));
        return n;
      };
      const W = 920, padL = 210, padR = 70, padT = 18, rowH = 34;
      const plotW = W - padL - padR;

      const ticks = stageNormMode === "pct" ? [0, 25, 50, 75, 100] : [0, Math.round(xDomain * 0.25), Math.round(xDomain * 0.5), Math.round(xDomain * 0.75), xDomain];
      ticks.forEach((t) => {
        if (t > xDomain) return;
        const x = Math.round(padL + (t / xDomain) * plotW);
        svg.appendChild(mk("line", { x1: x, y1: padT - 4, x2: x, y2: padT + laneStats.length * rowH, stroke: "var(--viz-grid)", "stroke-width": 1 }));
        const tx = mk("text", { x, y: padT + laneStats.length * rowH + 16, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11, class: "mono" });
        tx.textContent = stageNormMode === "pct" ? t + "%" : (t >= 1000 ? (t / 1000).toFixed(1) + "ks" : t + "s");
        svg.appendChild(tx);
      });

      laneStats.forEach((st, idx) => {
        const yRow = padT + idx * rowH;
        const lbl = mk("text", { x: padL - 10, y: yRow + 20, "text-anchor": "end", fill: "var(--text)", "font-size": 12 });
        lbl.textContent = st.ln.lane_title;
        svg.appendChild(lbl);

        let curX = padL;
        STAGE_META.forEach((m) => {
          const val = st.cats[m.key] || 0;
          if (val <= 0 || st.sumSec <= 0) return;
          const ratio = stageNormMode === "pct" ? (val / st.sumSec) : (val / xDomain);
          const segW = Math.max(2, Math.round(ratio * plotW) - 2);
          const rect = mk("rect", { x: curX, y: yRow + 6, width: segW, height: 20, rx: 2, fill: m.color, style: "cursor:pointer;" });
          rect.addEventListener("pointermove", (ev) => {
            showFloatTip(ev, `${st.ln.lane_title} · 非重叠阶段分解`, STAGE_META.map((sm) => ({
              label: sm.label,
              value: `${fmtNum(st.cats[sm.key], 1)}s (${st.sumSec > 0 ? fmtNum((st.cats[sm.key] / st.sumSec) * 100, 1) : "0.0"}%)`,
              color: sm.color
            })));
          });
          rect.addEventListener("pointerleave", hideFloatTip);
          rect.addEventListener("click", () => {
            if (st.ac && st.ac.global_cell_id) {
              selectedGlobalCell = st.ac.global_cell_id;
              render(lastData);
            }
          });
          svg.appendChild(rect);
          curX += segW + 2;
        });

        const rTxt = mk("text", { x: curX + 4, y: yRow + 20, fill: "var(--muted)", "font-size": 11, class: "mono" });
        rTxt.textContent = fmtNum(st.sumSec / 3600, 2) + "h";
        svg.appendChild(rTxt);
      });

      const selMeta = cells.find((c) => c.global_cell_id === selectedGlobalCell) || cells[0];
      const selRaw = selMeta ? (getRawCellFromData(data, selMeta.server_id, selMeta.cell_key) || {}) : {};
      const sdCsum = selRaw.costs_summary || {};
      const roleCounts = sdCsum.role_counts || {};
      const roleSecs = sdCsum.role_seconds || {};
      const rolesSorted = Object.entries(roleSecs).sort((a, b) => b[1] - a[1]);
      const maxSecRole = Math.max(1, ...rolesSorted.map((x) => x[1]));

      document.getElementById("cost-total-meta").textContent = `${fmtInt(sdCsum.total_events)} events · elapsed ${fmtNum(sdCsum.session_elapsed_seconds, 1)}s`;
      document.getElementById("cost-bars").innerHTML = rolesSorted.length ? rolesSorted.map(([role, sec]) => {
        const w = Math.max(1, Math.round((sec / maxSecRole) * 100));
        const cnt = roleCounts[role] || 0;
        return `
          <div style="margin-bottom:8px;">
            <div class="card-h" style="margin-bottom:2px;"><span class="mono">${esc(role)}${role === "rollout.group" ? " (enclosing)" : ""}</span><span class="mono">${fmtNum(sec, 2)}s · ${fmtInt(cnt)} 次</span></div>
            <div class="bar-bg" style="height:6px;margin-top:0;"><div class="bar-fill" style="width:${w}%;background:var(--c1)"></div></div>
          </div>
        `;
      }).join("") : `<div class="kpi-sub">该单元暂无 costs.jsonl 事件</div>`;

      const recEvents = selRaw.recent_cost_events || [];
      document.getElementById("recent-events-wrap").innerHTML = recEvents.length ? `
        <table><thead><tr><th>#</th><th>Role</th><th>Group</th><th>Tok (in/out)</th><th>Steps</th><th>Sec</th><th>Outcome</th></tr></thead><tbody>
        ${recEvents.slice().reverse().map((e) => `<tr class="mono"><td>${e.event_index ?? "-"}</td><td>${esc(e.role)}</td><td>${esc(e.group_id || "-")}</td><td>${e.input_tokens || 0}/${e.output_tokens || 0}</td><td>${e.env_steps || 0}</td><td>${fmtNum(e.seconds, 3)}s</td><td>${esc(e.outcome || "-")}</td></tr>`).join("")}
        </tbody></table>` : `<div class="kpi-sub">暂无最近事件</div>`;
    }

    function renderLogsTab(data) {
      const q = (document.getElementById("proc-search").value || "").trim().toLowerCase();
      const fSrv = document.getElementById("proc-srv-filter").value;
      const fRole = document.getElementById("proc-role-filter").value;

      const allProcs = [];
      ["colab", "host_a", "host_b"].forEach((sid) => {
        const tel = ((data.servers || {})[sid] || {}).telemetry || {};
        (tel.proc_list || []).forEach((p) => {
          allProcs.push(Object.assign({ server_id: sid }, p));
        });
      });

      const filtered = allProcs.filter((p) => {
        if (fSrv !== "all" && p.server_id !== fSrv) return false;
        const rl = (p.role || "").toLowerCase();
        if (fRole === "supervisor" && !rl.includes("supervisor") && !rl.includes("queue") && !rl.includes("handoff")) return false;
        if (fRole === "runner" && !rl.includes("runner")) return false;
        if (fRole === "envs" && !rl.includes("envs")) return false;
        if (q) {
          const hay = `${p.pid} ${p.ppid} ${p.server_id} ${p.lane} ${p.role} ${p.affinity} ${p.cmd}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });

      document.getElementById("procs-wrap").innerHTML = filtered.length ? `
        <table><thead><tr><th>PID</th><th>PPID</th><th>Server / Lane</th><th>Role</th><th>CPU%</th><th>MEM% (RSS)</th><th>NUMA Affinity</th><th>Elapsed</th><th>Command (Sanitized)</th></tr></thead><tbody>
        ${filtered.map((p) => `<tr class="mono">
          <td><b>${esc(p.pid)}</b></td>
          <td>${esc(p.ppid)}</td>
          <td>${esc(p.server_id.toUpperCase())} · ${esc(p.lane || "-")}</td>
          <td>${esc(p.role || "-")}</td>
          <td>${esc(p.cpu_pct)}%</td>
          <td>${esc(p.mem_pct)}%${p.rss_mib != null ? ` (${fmtInt(Math.round(p.rss_mib))} MiB)` : ""}</td>
          <td>${esc(p.affinity || "-")}</td>
          <td>${esc(p.elapsed)}</td>
          <td style="word-break:break-all;min-width:280px;">${esc(p.cmd)}</td>
        </tr>`).join("")}
        </tbody></table>
      ` : `<div class="kpi-sub">无匹配活跃进程</div>`;

      const logMap = {};
      ["colab", "host_a", "host_b"].forEach((sid) => {
        const tel = ((data.servers || {})[sid] || {}).telemetry || {};
        Object.entries(tel.log_tails || {}).forEach(([k, v]) => {
          logMap[k] = v;
        });
      });
      const keys = Object.keys(logMap);
      if (keys.length && (!selectedLogKey || !logMap[selectedLogKey])) {
        selectedLogKey = keys[0];
      }
      document.getElementById("log-key-pills").innerHTML = keys.map((k) =>
        `<button class="btn btn-sm ${k === selectedLogKey ? "btn-primary" : ""}" onclick="window._selLog('${esc(k)}')">${esc(k)}</button>`
      ).join("");
      window._selLog = (k) => { selectedLogKey = k; if (lastData) renderLogsTab(lastData); };

      const curLog = logMap[selectedLogKey] || { path: "-", text: "(等待远端日志拉取...)" };
      document.getElementById("log-path-label").textContent = curLog.path || "-";
      document.getElementById("log-tail").textContent = curLog.text || "(日志为空)";
    }

    function render(data) {
      lastData = data;
      const cs = data.cluster_summary || {};
      const badge = document.getElementById("cluster-badge");
      if (cs.servers_online === cs.servers_enabled && cs.servers_online > 0) {
        badge.innerHTML = `<span class="dot good"></span><span>${cs.servers_online}/${cs.servers_enabled} 机在线并发监控中</span>`;
      } else if (cs.servers_online > 0) {
        badge.innerHTML = `<span class="dot warn"></span><span>部分在线 (${cs.servers_online}/${cs.servers_enabled} 机)</span>`;
      } else {
        badge.innerHTML = `<span class="dot warn"></span><span>探测连接中...</span>`;
      }

      const pNote = cs.cells_paused ? ` / ${cs.cells_paused} 暂存` : "";
      document.getElementById("subtitle").textContent =
        `本地快照 ${data.generated_at || "-"} · ${cs.cells_completed ?? 0} 完成 / ${cs.cells_running ?? 0} 运行${pNote} / ${cs.cells_queued ?? 0} 已排队 / ${cs.cells_planned ?? 0} 计划`;

      const srvPills = document.getElementById("server-status-pills");
      srvPills.innerHTML = ["colab", "host_a", "host_b"].map((sid) => {
        const s = (data.servers || {})[sid] || {};
        if (!s.enabled) return "";
        const dotCls = s.status === "online" ? "good" : (s.status === "error" ? "bad" : "warn");
        return `<span class="srv-pill mono">
          <span class="dot ${dotCls}"></span>
          <b>${esc(sid.toUpperCase())}</b>
          <span style="color:var(--muted)">${esc(s.status)}${s.polling ? " (polling)" : ""} · ${s.last_ok_at ? s.last_ok_at.slice(11) : "-"} · ${s.latency_ms != null ? s.latency_ms + "ms" : "-"}</span>
          <button title="单独刷新此服务器" onclick="window._refreshServer('${sid}')">↻</button>
        </span>`;
      }).join("");

      const errs = [];
      ["colab", "host_a", "host_b"].forEach((sid) => {
        const s = (data.servers || {})[sid] || {};
        if (s.enabled && s.last_error) {
          errs.push(`[${sid.toUpperCase()}] ${s.last_error}`);
        }
      });
      const errBox = document.getElementById("err-box");
      if (errs.length) {
        errBox.classList.remove("hidden");
        errBox.textContent = "部分服务器最近探测告警（已保留上次成功快照，互不阻塞）: " + errs.join("  ||  ");
      } else {
        errBox.classList.add("hidden");
      }

      if (currentTab === "overview") renderOverviewTab(data);
      else if (currentTab === "experiment") renderExperimentTab(data);
      else if (currentTab === "schedule") renderScheduleTab(data);
      else if (currentTab === "costs") renderCostsTab(data);
      else if (currentTab === "logs") renderLogsTab(data);
    }

    document.getElementById("refresh-btn").addEventListener("click", async () => {
      const btn = document.getElementById("refresh-btn");
      btn.disabled = true;
      btn.textContent = "三机并发探测中...";
      try {
        const target = scopeServer === "all" ? "all" : scopeServer;
        const res = await fetch("/api/refresh?server=" + encodeURIComponent(target), { method: "POST" });
        const data = await res.json();
        render(data);
      } finally {
        btn.disabled = false;
        btn.textContent = "立即探测三机";
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

    fetchState();
    resetTimer();
  </script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# HTTP Server & CLI Entrypoint
# ---------------------------------------------------------------------------

class MonitorHandler(BaseHTTPRequestHandler):
    poller: ClusterPoller

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
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/state":
            self._send_json(self.poller.get_snapshot())
        else:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/refresh":
            qs = urllib.parse.parse_qs(parsed.query)
            srv = (qs.get("server") or ["all"])[0]
            if srv not in ("all", "colab", "host_a", "host_b"):
                self._send_json({"error": "invalid_server"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.poller.poll_once(target_server=srv, force_meta=(srv in ("all", "colab"))))
        else:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)


def _background_loop(poller: ClusterPoller, interval: int, stop_event: threading.Event) -> None:
    poller.poll_once(target_server="all", force_meta=True)
    if interval <= 0:
        return
    while not stop_event.wait(interval):
        poller.poll_once(target_server="all")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_ssh_doc = str(repo_root / "SSH_Document.md")
    default_sched_csv = str(repo_root / "outputs" / "exp-plan-20261118" / "compute_schedule_current.csv")

    parser = argparse.ArgumentParser(description="Unified live monitor for F-GPO experiments across Colab, Host A, and Host B.")
    # Preserved backward-compatible CLI flags
    parser.add_argument("--session", default="fgpo-r3-956ceab-alf")
    parser.add_argument("--output-dir", default="/content/outputs/fgpo-r3-published-v4-alf")
    parser.add_argument("--log-path", default="auto")
    parser.add_argument("--config-path", default="/content/fgpo-r3/configs/fgpo-r3-colab-a100.json")
    parser.add_argument("--suite", default="comparison")
    parser.add_argument("--benchmark", default="alfworld")
    parser.add_argument("--method", default="fgpo")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=int, default=60, help="Auto-poll interval in seconds (0 = manual only)")
    parser.add_argument("--transport", choices=("exec", "ssh"), default="exec")
    parser.add_argument("--timeout", type=int, default=90, help="Remote probe timeout in seconds")
    parser.add_argument("--colab-bin", default="~/.local/bin/colab")
    # Multi-server V4 flags
    parser.add_argument("--ssh-doc", default=default_ssh_doc, help="Path to SSH_Document.md for Host A & Host B")
    parser.add_argument("--schedule-csv", default=default_sched_csv, help="Path to the current schedule CSV")
    parser.add_argument("--host-a-dir", default="auto", help="Deployment directory on Host A (or 'auto')")
    parser.add_argument("--host-b-dir", default="auto", help="Deployment directory on Host B (or 'auto')")
    parser.add_argument("--disable-colab", action="store_true", help="Disable Colab polling")
    parser.add_argument("--disable-host-a", action="store_true", help="Disable Host A polling")
    parser.add_argument("--disable-host-b", action="store_true", help="Disable Host B polling")
    args = parser.parse_args()

    poller = ClusterPoller(args)
    stop_event = threading.Event()
    bg = threading.Thread(target=_background_loop, args=(poller, args.interval, stop_event), daemon=True)
    bg.start()

    class _Handler(MonitorHandler):
        pass

    _Handler.poller = poller

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    enabled_names = [k for k, v in poller.enabled.items() if v]
    print(
        f"F-GPO Unified 3-Server Monitor UI running at http://{args.host}:{args.port} "
        f"(servers={','.join(enabled_names)}, colab_session={args.session}, interval={args.interval}s)"
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
