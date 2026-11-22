#!/usr/bin/env python3
"""Unified live local web dashboard backend for monitoring remote F-GPO experiments
across all 3 servers in parallel (`Colab`, `Host A`, `Host B`).

- Dynamic hot-reloading UI (`scripts/fgpo_monitor_dashboard.html`) with zero restart.
- Independent async per-server probe loops with SSE push (`/api/events`),
  delta-checked gzipped `/api/state?since_seq=...`, non-blocking `/api/refresh?async=1`,
  and graceful port handoff (`/api/health`, `/api/shutdown`).
- Remote auto-discovery of unscheduled study cells, per-cell velocity/ETA & idle
  heartbeat, and local schedule CSV / archive hot-reload.
- Strictly read-only remote telemetry collection using Python 3 stdlib only,
  with zero hardcoded credentials, IPs, ports, usernames, or tokens.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from datetime import datetime
import errno
import fcntl
import gzip
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
import sys
import termios
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from fgpo_monitor_rounds import project_training_round


DASHBOARD_HTML_PATH = _SCRIPT_DIR / "fgpo_monitor_dashboard.html"

_FALLBACK_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8" /><title>F-GPO 实验监控与排程</title></head>
<body style="background:#0b0f17;color:#e6edf7;font-family:sans-serif;padding:24px">
  <h2>F-GPO 实验监控后端已就绪</h2>
  <p>未找到前端视图文件 <code>fgpo_monitor_dashboard.html</code>，请确认文件位于同级目录后刷新页面。</p>
</body>
</html>
"""


def _get_ui_mtime() -> int:
    try:
        if DASHBOARD_HTML_PATH.is_file():
            return int(DASHBOARD_HTML_PATH.stat().st_mtime)
        return int(Path(__file__).resolve().stat().st_mtime)
    except OSError:
        return 0


def _load_dashboard_html() -> str:
    try:
        if DASHBOARD_HTML_PATH.is_file():
            return DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
    except OSError:
        pass
    return _FALLBACK_DASHBOARD_HTML


DASHBOARD_HTML = _load_dashboard_html()


def _parse_gpu_indices(physical_gpu_ids: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"[+,/\s]+", str(physical_gpu_ids or "").strip()):
        if part.isdigit():
            out.append(int(part))
    return out


LANE_DEFINITIONS: list[dict[str, Any]] = [
    {"lane_id": "colab_gpu_0", "server_id": "colab", "lane_title": "Colab · GPU 0", "short_lane": "Colab GPU 0", "role_slot": "Colab GPU 0", "gpu_indices": [0]},
    {"lane_id": "host_a_l40_3_4", "server_id": "host_a", "lane_title": "Host A · L40 3+4 (Queue A1)", "short_lane": "L40 3+4", "role_slot": "Queue A1 (3+4)", "gpu_indices": [3, 4]},
    {"lane_id": "host_a_l40_1_7", "server_id": "host_a", "lane_title": "Host A · L40 1+7 (Queue A2)", "short_lane": "L40 1+7", "role_slot": "Queue A2 (1+7)", "gpu_indices": [1, 7]},
    {"lane_id": "host_a_gpu_7", "server_id": "host_a", "lane_title": "Host A · GPU 7 (单卡)", "short_lane": "GPU 7", "role_slot": "Host A GPU 7 (单卡)", "gpu_indices": [7]},
    {"lane_id": "host_a_gpu_3", "server_id": "host_a", "lane_title": "Host A · GPU 3 (单卡)", "short_lane": "GPU 3", "role_slot": "Host A GPU 3 (单卡)", "gpu_indices": [3]},
    {"lane_id": "host_b_gpu_0", "server_id": "host_b", "lane_title": "Host B · GPU 0 (Queue B0)", "short_lane": "GPU 0", "role_slot": "Queue B0 (GPU 0 · 80 GiB cap)", "gpu_indices": [0]},
    {"lane_id": "host_b_gpu_2", "server_id": "host_b", "lane_title": "Host B · GPU 2 (Queue B2)", "short_lane": "GPU 2", "role_slot": "Queue B2 (GPU 2 · 80 GiB cap)", "gpu_indices": [2]},
]


LITERATURE_EXTRA_BASELINES: list[dict[str, Any]] = [
    # WebShop 7B references documented in BiPACE Table 2, notes_core.md & fgpo_webshop_host_b_schedule.md
    {"benchmark": "webshop", "backbone": "7B", "method": "BiPACE-Q", "method_type": "RL Training", "sr": 79.7, "sd": 3.3, "score": 89.6, "score_sd": 1.3, "source": "BiPACE 表 2", "protocol_note": "150 轮, G=12, H=15 · val@max 峰值（环境版本与评测抽样协议不同，仅作文献坐标）"},
    {"benchmark": "webshop", "backbone": "7B", "method": "G2PO", "method_type": "RL Training", "sr": 78.3, "sd": 0.6, "score": 89.8, "score_sd": 1.7, "source": "G2PO 表 1", "protocol_note": "100 轮 · 3 种子均值 ± SD"},
    {"benchmark": "webshop", "backbone": "7B", "method": "ProxMO", "method_type": "RL Training", "sr": 76.5, "sd": None, "score": 87.2, "score_sd": None, "source": "ProxMO 表 1", "protocol_note": "150 轮 · 文献报告均值"},
    {"benchmark": "webshop", "backbone": "7B", "method": "HGPO", "method_type": "RL Training", "sr": 76.43, "sd": 1.47, "score": 88.93, "score_sd": 0.84, "source": "HGPO 附录表 7", "protocol_note": "160 轮, 15 步, K=2 · 均值 ± SD（表 1 30 步 / BiPACE 表 2 引用为 78.5±1.4%, Score 89.0±1.0）"},
    {"benchmark": "webshop", "backbone": "7B", "method": "GiGPO (原环境)", "method_type": "RL Training", "sr": 75.2, "sd": 3.8, "score": 86.2, "score_sd": 2.6, "source": "BiPACE 表 2 (Feng et al. 2025)", "protocol_note": "150 轮, 15 步 · GiGPO 原环境报告值（对照 GraphGPO text-rich 5120 复现的 78.4%）"},
    {"benchmark": "webshop", "backbone": "7B", "method": "ECPO", "method_type": "RL Training", "sr": 74.7, "sd": 3.0, "score": 85.5, "score_sd": 0.5, "source": "ECPO 表 1", "protocol_note": "150 轮 · 3 种子均值 ± SD"},
    {"benchmark": "webshop", "backbone": "7B", "method": "RewardFlow", "method_type": "RL Training", "sr": 74.7, "sd": 3.0, "score": 85.3, "score_sd": 0.9, "source": "RewardFlow 表 1", "protocol_note": "100 轮（波动来自固定检查点的三个推理种子）"},
    {"benchmark": "webshop", "backbone": "7B", "method": "GRPO (原环境)", "method_type": "RL Training", "sr": 66.1, "sd": 3.7, "score": 79.3, "score_sd": 2.8, "source": "BiPACE 表 2 (Feng et al. 2025)", "protocol_note": "150 轮, 15 步 · GRPO 原环境报告值（对照 GraphGPO text-rich 5120 复现的 75.0%）"},
    # WebShop 1.5B references documented in GraphGPO Table 3, BiPACE Table 2 & notes_core.md
    {"benchmark": "webshop", "backbone": "1.5B", "method": "GraphGPO + DS", "method_type": "RL Training", "sr": 85.68, "sd": 3.52, "score": None, "score_sd": None, "source": "GraphGPO 表 3", "protocol_note": "150 轮 · text-rich + 动态重采样 DS（全成/全败组最多重采 10 次，仅报 SR）"},
    {"benchmark": "webshop", "backbone": "1.5B", "method": "GiGPO + DS", "method_type": "RL Training", "sr": 81.25, "sd": 2.78, "score": None, "score_sd": None, "source": "GraphGPO 表 3", "protocol_note": "150 轮 · text-rich + 动态重采样 DS（最多重采 10 次，仅报 SR）"},
    {"benchmark": "webshop", "backbone": "1.5B", "method": "BiPACE-Q", "method_type": "RL Training", "sr": 71.9, "sd": 2.1, "score": 85.8, "score_sd": 1.1, "source": "BiPACE 表 2", "protocol_note": "200 轮, G=12, H=15 · val@max 峰值"},
    {"benchmark": "webshop", "backbone": "1.5B", "method": "ECPO", "method_type": "RL Training", "sr": 71.9, "sd": 1.7, "score": 84.4, "score_sd": None, "source": "ECPO 表 1", "protocol_note": "150 轮 · 3 种子均值 ± SD"},
    {"benchmark": "webshop", "backbone": "1.5B", "method": "HGPO", "method_type": "RL Training", "sr": 71.54, "sd": 4.00, "score": 85.6, "score_sd": 2.9, "source": "HGPO 表 1 / BiPACE 表 2", "protocol_note": "160 轮, K=2, 30 步 · 均值 ± SD（K=4 为 78.12±2.06%）"},
    {"benchmark": "webshop", "backbone": "1.5B", "method": "G2PO", "method_type": "RL Training", "sr": 71.2, "sd": 2.6, "score": 85.1, "score_sd": None, "source": "G2PO 表 1", "protocol_note": "100 轮 · 3 种子均值 ± SD"},
    {"benchmark": "webshop", "backbone": "1.5B", "method": "GiGPO (原环境)", "method_type": "RL Training", "sr": 67.4, "sd": 4.5, "score": 83.5, "score_sd": 1.8, "source": "BiPACE 表 2 (Feng et al. 2025)", "protocol_note": "150 轮, 15 步 · GiGPO 原环境报告值（BiPACE 标注部分完成种子，对照 text-rich 73.8%）"},
    {"benchmark": "webshop", "backbone": "1.5B", "method": "ProxMO", "method_type": "RL Training", "sr": 67.1, "sd": None, "score": 85.3, "score_sd": None, "source": "ProxMO 表 1", "protocol_note": "150 轮 · 文献报告均值"},
    {"benchmark": "webshop", "backbone": "1.5B", "method": "RewardFlow", "method_type": "RL Training", "sr": 66.4, "sd": None, "score": None, "score_sd": None, "source": "RewardFlow 表 1", "protocol_note": "100 轮 · Table 1 报告值"},
    {"benchmark": "webshop", "backbone": "1.5B", "method": "GRPO (原环境)", "method_type": "RL Training", "sr": 56.8, "sd": 3.8, "score": 75.8, "score_sd": 3.5, "source": "BiPACE 表 2 (Feng et al. 2025)", "protocol_note": "150 轮, 15 步 · GRPO 原环境报告值（对照 text-rich 71.4%）"},
    # ALFWorld 7B references documented in BiPACE Table 2 & notes_core.md
    {"benchmark": "alfworld", "backbone": "7B", "method": "BiPACE-Q", "method_type": "RL Training", "sr": 97.1, "sd": 0.9, "score": None, "score_sd": None, "task_types": {"Pick": {"sr": 100.0}, "Look": {"sr": 100.0}, "Clean": {"sr": 100.0}, "Heat": {"sr": 100.0}, "Cool": {"sr": 95.3}, "Pick2": {"sr": 100.0}, "All": {"sr": 97.1}}, "source": "BiPACE 表 2", "protocol_note": "150 轮, G=8, H=50 · valid_seen (|V|=128) val@max 峰值（与 valid_unseen 终态不同口径）"},
    {"benchmark": "alfworld", "backbone": "7B", "method": "G2PO", "method_type": "RL Training", "sr": 96.9, "sd": 1.3, "score": None, "score_sd": None, "source": "G2PO 表 1", "protocol_note": "100 轮 · 3 种子均值 ± SD"},
    {"benchmark": "alfworld", "backbone": "7B", "method": "HGPO", "method_type": "RL Training", "sr": 95.44, "sd": 0.62, "score": None, "score_sd": None, "source": "HGPO 表 1", "protocol_note": "160 轮, K=2 · ID 均值 ± SD（K=4 ID 95.96±0.49%，OOD K=2 92.05% / K=4 94.87%）"},
    {"benchmark": "alfworld", "backbone": "7B", "method": "ProxMO", "method_type": "RL Training", "sr": 94.5, "sd": None, "score": None, "score_sd": None, "source": "ProxMO 表 1", "protocol_note": "150 轮 · 文献报告均值"},
    {"benchmark": "alfworld", "backbone": "7B", "method": "ECPO", "method_type": "RL Training", "sr": 91.9, "sd": 0.7, "score": None, "score_sd": None, "source": "ECPO 表 1", "protocol_note": "150 轮 · 3 种子均值 ± SD"},
    {"benchmark": "alfworld", "backbone": "7B", "method": "GiGPO (原环境)", "method_type": "RL Training", "sr": 90.8, "sd": 1.3, "score": None, "score_sd": None, "task_types": {"Pick": {"sr": 97.7}, "Look": {"sr": 82.7}, "Clean": {"sr": 98.8}, "Heat": {"sr": 83.7}, "Cool": {"sr": 89.3}, "Pick2": {"sr": 79.2}, "All": {"sr": 90.8}}, "source": "BiPACE 表 2 (Feng et al. 2025)", "protocol_note": "150 轮, H=50 · GiGPO 原观测环境报告值（对照 GraphGPO 增强观测复现的 94.3%）"},
    {"benchmark": "alfworld", "backbone": "7B", "method": "RewardFlow", "method_type": "RL Training", "sr": 84.6, "sd": 4.5, "score": None, "score_sd": None, "source": "RewardFlow 表 1", "protocol_note": "100 轮, H=25, temp=0.4 · ID 均值 ± SD（OOD 80.5%，与 50 步协议不同）"},
    {"benchmark": "alfworld", "backbone": "7B", "method": "GRPO (原环境)", "method_type": "RL Training", "sr": 77.6, "sd": 5.2, "score": None, "score_sd": None, "task_types": {"Pick": {"sr": 90.8}, "Look": {"sr": 66.1}, "Clean": {"sr": 89.3}, "Heat": {"sr": 74.7}, "Cool": {"sr": 72.5}, "Pick2": {"sr": 64.7}, "All": {"sr": 77.6}}, "source": "BiPACE 表 2 (Feng et al. 2025)", "protocol_note": "150 轮, H=50 · GRPO 原观测环境报告值（对照 GraphGPO 增强观测复现的 83.3%）"},
    # ALFWorld 1.5B references documented in GraphGPO Table 3, BiPACE Table 2 & notes_core.md
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "GraphGPO + DS", "method_type": "RL Training", "sr": 98.43, "sd": 1.28, "score": None, "score_sd": None, "task_types": {"Pick": {"sr": 100.0}, "Clean": {"sr": 98.41}, "Cool": {"sr": 95.15}, "Look": {"sr": 97.62}, "Heat": {"sr": 100.0}, "Pick2": {"sr": 100.0}, "All": {"sr": 98.43}}, "source": "GraphGPO 表 3", "protocol_note": "150 轮 · 增强观测 + 动态重采样 DS（全成/全败组最多重采 10 次）"},
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "GiGPO + DS", "method_type": "RL Training", "sr": 96.35, "sd": 1.95, "score": None, "score_sd": None, "task_types": {"Pick": {"sr": 97.57}, "Clean": {"sr": 100.0}, "Cool": {"sr": 90.21}, "Look": {"sr": 95.24}, "Heat": {"sr": 98.0}, "Pick2": {"sr": 98.41}, "All": {"sr": 96.35}}, "source": "GraphGPO 表 3", "protocol_note": "150 轮 · 增强观测 + 动态重采样 DS（最多重采 10 次）"},
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "G2PO", "method_type": "RL Training", "sr": 95.0, "sd": 0.8, "score": None, "score_sd": None, "source": "G2PO 表 1", "protocol_note": "100 轮 · 3 种子均值 ± SD"},
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "BiPACE-Q", "method_type": "RL Training", "sr": 93.5, "sd": 1.2, "score": None, "score_sd": None, "task_types": {"Pick": {"sr": 100.0}, "Look": {"sr": 97.4}, "Clean": {"sr": 100.0}, "Heat": {"sr": 100.0}, "Cool": {"sr": 96.5}, "Pick2": {"sr": 92.0}, "All": {"sr": 93.5}}, "source": "BiPACE 表 2", "protocol_note": "200 轮, G=8, H=50 · valid_seen (|V|=128) val@max 峰值"},
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "HGPO", "method_type": "RL Training", "sr": 92.77, "sd": 1.08, "score": None, "score_sd": None, "source": "HGPO 表 1", "protocol_note": "160 轮, K=2 · ID 均值 ± SD（K=4 ID 94.85±0.92%，OOD K=2 90.16% / K=4 92.12%）"},
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "ECPO", "method_type": "RL Training", "sr": 92.7, "sd": 0.7, "score": None, "score_sd": None, "source": "ECPO 表 1", "protocol_note": "150 轮 · 3 种子均值 ± SD"},
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "ProxMO", "method_type": "RL Training", "sr": 90.6, "sd": None, "score": None, "score_sd": None, "source": "ProxMO 表 1", "protocol_note": "150 轮 · 文献报告均值"},
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "GiGPO (原环境)", "method_type": "RL Training", "sr": 86.7, "sd": 1.7, "score": None, "score_sd": None, "task_types": {"Pick": {"sr": 94.4}, "Look": {"sr": 67.5}, "Clean": {"sr": 94.8}, "Heat": {"sr": 94.4}, "Cool": {"sr": 79.8}, "Pick2": {"sr": 76.4}, "All": {"sr": 86.7}}, "source": "BiPACE 表 2 (Feng et al. 2025)", "protocol_note": "150 轮, H=50 · GiGPO 原观测环境报告值（对照 GraphGPO 增强观测复现的 90.9%）"},
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "GRPO (原环境)", "method_type": "RL Training", "sr": 75.2, "sd": 3.8, "score": None, "score_sd": None, "task_types": {"Pick": {"sr": 85.3}, "Look": {"sr": 53.7}, "Clean": {"sr": 84.5}, "Heat": {"sr": 78.2}, "Cool": {"sr": 59.7}, "Pick2": {"sr": 53.5}, "All": {"sr": 75.2}}, "source": "BiPACE 表 2 (Feng et al. 2025)", "protocol_note": "150 轮, H=50 · GRPO 原观测环境报告值（对照 GraphGPO 增强观测复现的 77.9%）"},
    {"benchmark": "alfworld", "backbone": "1.5B", "method": "RewardFlow", "method_type": "RL Training", "sr": 62.8, "sd": 1.8, "score": None, "score_sd": None, "source": "RewardFlow 表 1", "protocol_note": "100 轮, H=25, temp=0.4 · ID 均值 ± SD（OOD 66.1%）"},
    # TextCraft 7B & 1.5B references documented in BiPACE Table 3 & Table I.1
    {"benchmark": "textcraft", "backbone": "7B", "method": "BiPACE-Q", "method_type": "RL Training", "sr": 91.1, "sd": 2.4, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 95.7, "sd": 2.1}, "3": {"sr": 87.4, "sd": 13.4}}, "source": "BiPACE 表 3", "protocol_note": "50 轮, G=12, H=70 · 3 种子验证峰值（非本地 30 步/79 题同口径）"},
    {"benchmark": "textcraft", "backbone": "7B", "method": "HGPO", "method_type": "RL Training", "sr": 87.5, "sd": 2.3, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 93.6, "sd": 1.8}, "3": {"sr": 85.0, "sd": 6.5}}, "source": "BiPACE 表 3", "protocol_note": "50 轮, G=12, H=70 · BiPACE 作者复现 3 种子验证峰值"},
    {"benchmark": "textcraft", "backbone": "7B", "method": "GiGPO", "method_type": "RL Training", "sr": 87.0, "sd": 2.4, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 94.0, "sd": 1.6}, "3": {"sr": 75.0, "sd": 3.5}}, "source": "BiPACE 表 3", "protocol_note": "50 轮, G=12, H=70 · BiPACE 作者复现 3 种子验证峰值"},
    {"benchmark": "textcraft", "backbone": "7B", "method": "GRPO", "method_type": "RL Training", "sr": 76.5, "sd": 3.1, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 90.3, "sd": 0.8}, "3": {"sr": 55.5, "sd": 10.0}}, "source": "BiPACE 表 3", "protocol_note": "50 轮, G=12, H=70 · BiPACE 作者复现 3 种子验证峰值"},
    {"benchmark": "textcraft", "backbone": "7B", "method": "Qwen2.5", "method_type": "Prompting", "sr": 6.8, "sd": 2.7, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 7.9, "sd": 2.7}, "3": {"sr": 4.0, "sd": 2.9}}, "source": "BiPACE 表 3", "protocol_note": "零样本基座 Prompting 参考（H=70）"},
    {"benchmark": "textcraft", "backbone": "1.5B", "method": "BiPACE-Q", "method_type": "RL Training", "sr": 65.1, "sd": 7.0, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 84.9, "sd": 3.0}, "3": {"sr": 29.6, "sd": 12.0}}, "source": "BiPACE 表 3", "protocol_note": "100 轮, G=12, H=70 · 3 种子验证峰值（非本地 50 轮/30 步/79 题同口径）"},
    {"benchmark": "textcraft", "backbone": "1.5B", "method": "HGPO", "method_type": "RL Training", "sr": 59.4, "sd": 8.0, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 77.8, "sd": 8.6}, "3": {"sr": 17.6, "sd": 4.6}}, "source": "BiPACE 表 3", "protocol_note": "100 轮, G=12, H=70 · BiPACE 作者复现 3 种子验证峰值"},
    {"benchmark": "textcraft", "backbone": "1.5B", "method": "GiGPO", "method_type": "RL Training", "sr": 58.3, "sd": 7.8, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 76.7, "sd": 5.4}, "3": {"sr": 21.8, "sd": 9.6}}, "source": "BiPACE 表 3", "protocol_note": "100 轮, G=12, H=70 · BiPACE 作者复现 3 种子验证峰值"},
    {"benchmark": "textcraft", "backbone": "1.5B", "method": "GRPO", "method_type": "RL Training", "sr": 43.8, "sd": 2.2, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 59.5, "sd": 1.9}, "3": {"sr": 20.6, "sd": 4.2}}, "source": "BiPACE 表 3", "protocol_note": "100 轮, G=12, H=70 · BiPACE 作者复现 3 种子验证峰值"},
    {"benchmark": "textcraft", "backbone": "1.5B", "method": "Qwen2.5", "method_type": "Prompting", "sr": 4.2, "sd": 1.5, "score": None, "score_sd": None, "recipe_depths": {"2": {"sr": 5.7, "sd": 2.0}, "3": {"sr": 0.0, "sd": 0.0}}, "source": "BiPACE 表 3", "protocol_note": "零样本基座 Prompting 参考（H=70）"},
]


# ---------------------------------------------------------------------------
# Shared Read-Only Remote Probe Helpers (Python 3.8+ universal stdlib)
# ---------------------------------------------------------------------------

ROUND_PROJECTION = (_SCRIPT_DIR / "fgpo_monitor_rounds.py").read_text(encoding="utf-8")
FROZEN_AUTHOR_JOBS = {
    "7b": ("output-author-eval-542-20261121", {"webshop": 384, "textcraft-fgpo": 79, "textcraft-graphgpo": 79}),
    "1p5b": ("output-author-eval-1p5b-414-20261121",
             {"webshop-original": 128, "webshop-adapted": 128, "textcraft-base": 79, "textcraft-trained": 79}),
}
FROZEN_AUTHOR_SNAPSHOTS = {"webshop": "c4862b17332242cabea67fa93c9d85f7"}
FROZEN_AUTHOR_INTERFACE = "native_training_v1"
REMOTE_COMMON_HELPERS = ROUND_PROJECTION + (
    f"\nFROZEN_AUTHOR_JOBS = {FROZEN_AUTHOR_JOBS!r}\n"
    f"FROZEN_AUTHOR_SNAPSHOTS = {FROZEN_AUTHOR_SNAPSHOTS!r}\n"
    f"FROZEN_AUTHOR_INTERFACE = {FROZEN_AUTHOR_INTERFACE!r}\n"
) + r'''
import glob, hashlib, json, os, pathlib, subprocess, time

MY_PID = os.getpid()
PROBE_EPOCH = time.time()

def read_frozen_author_evaluation(deploy_dir, cohort="7b"):
    if deploy_dir is None:
        return {}
    directory, jobs = FROZEN_AUTHOR_JOBS[cohort]
    root = deploy_dir / directory / "native-training"
    panels = (0, 1, 2) if cohort == "7b" else (1,)
    results = {}
    for name in jobs:
        webshop = name.startswith("webshop")
        expected_snapshot = FROZEN_AUTHOR_SNAPSHOTS.get(name)
        try:
            state = json.loads((root / (name + ".state.json")).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = None
        if not isinstance(state, dict) or state.get("interface_protocol") != FROZEN_AUTHOR_INTERFACE:
            state = None
        state_snapshot = (state or {}).get("model_snapshot")
        if state_snapshot is not None and (
                not isinstance(state_snapshot, str) or not state_snapshot
                or (expected_snapshot is not None and state_snapshot != expected_snapshot)):
            state = state_snapshot = None
        actor = expected_snapshot or state_snapshot
        by_actor = {}
        try:
            with (root / (name + ".jsonl")).open("rb") as journal:
                for line in journal:
                    if not line.endswith(b"\n"):
                        break
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(row, dict) or type(row.get("success")) is not bool:
                        continue
                    snapshot, origin = row.get("model_snapshot"), row.get("origin")
                    if (row.get("interface_protocol") != FROZEN_AUTHOR_INTERFACE
                            or not isinstance(snapshot, str) or not snapshot
                            or (actor is not None and snapshot != actor)
                            or origin not in ("archived_native", "new_native")):
                        continue
                    ordinal = row.get("ordinal")
                    if type(ordinal) is not int or not 0 <= ordinal < (128 if webshop else 79):
                        continue
                    if webshop:
                        panel, score = row.get("panel"), row.get("score")
                        if type(panel) is not int or panel not in panels:
                            continue
                        if type(score) not in (float, int) or not 0 <= score <= 1:
                            continue
                        if origin != ("archived_native" if cohort == "1p5b" or panel == 1 else "new_native"):
                            continue
                        fields = ("panel", "ordinal", "task_id", "worker_seed", "session_index", "sampling_seed",
                                  "success", "score", "steps")
                    else:
                        panel = None
                        first = row.get("first_success_step")
                        if row["success"] and (type(first) is not int or not 1 <= first <= 70):
                            continue
                        if not row["success"] and first is not None:
                            continue
                        fields = ("ordinal", "task_id", "depth", "initial_seed", "sampling_seed", "success",
                                  "first_success_step", "steps")
                    fields += ("model_snapshot", "interface_protocol", "origin")
                    by_actor.setdefault(snapshot, {}).setdefault((panel, ordinal), {key: row.get(key) for key in fields})
        except OSError:
            pass
        if actor is None and len(by_actor) == 1:
            actor = next(iter(by_actor))
        rows = by_actor.get(actor, {})
        if state is not None and state_snapshot is None and state.get("status") in ("running", "evaluating", "preparing"):
            state = None
        if rows or state is not None:
            results[name] = {
                "interface_protocol": FROZEN_AUTHOR_INTERFACE,
                "model_snapshot": actor,
                "rows": list(rows.values()),
                "state": {key: state[key] for key in ("status", "updated_at", "runner_pid") if key in state}
                         if state is not None else {},
            }
    return results

PHASE_ROLES = (
    "rollout.group", "rollout.tree_group", "rollout.step_native", "rollout.decode",
    "rollout.segment_replay", "rollout.merge_cohorts",
    "native.step", "native.reset",
    "gigpo.actor.decode", "gigpo.actor.decode.head", "gigpo.actor.batch_prefill",
    "g2po.actor.decode", "g2po.actor.decode.head", "g2po.actor.batch_prefill",
    "graphgpo.actor.decode", "graphgpo.actor.decode.head", "graphgpo.actor.batch_prefill",
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

def read_proc_log_fds(pid):
    out = []
    for fd in ("1", "2"):
        try:
            tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
            if tgt.endswith(".log") and os.path.isfile(tgt) and tgt not in out:
                out.append(tgt)
        except OSError:
            pass
    return out

def parse_runner_cli(cmd):
    toks = cmd.split()
    info = {"target_resume": "--resume" in toks}
    is_q15 = "run_fgpo_host_1p5b_queue" in cmd
    model_gpus = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if is_q15 and t == "--gpu" and i > 0 and toks[i - 1] in ("alfworld", "textcraft", "webshop"):
            info.setdefault("target_benchmark", toks[i - 1])
        if t == "--config" and i + 1 < len(toks):
            info["target_config"] = toks[i + 1]
            i += 2
            continue
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
        if t in ("--gpu", "--model-gpu") and i + 1 < len(toks):
            gpus = []
            j = i + 1
            if is_q15 and t == "--gpu":
                for part in toks[j].split(","):
                    if part.strip().isdigit():
                        gpus.append(part.strip())
                j += 1
                q_seeds = []
                while j < len(toks) and not toks[j].startswith("--"):
                    if toks[j].strip().isdigit():
                        q_seeds.append(int(toks[j].strip()))
                    j += 1
                if q_seeds and "target_seeds" not in info:
                    info["target_seeds"] = q_seeds
            else:
                while j < len(toks) and not toks[j].startswith("--"):
                    for part in toks[j].split(","):
                        if part.strip().isdigit():
                            gpus.append(part.strip())
                    j += 1
            if t == "--gpu" and gpus:
                info["target_gpus"] = gpus
            elif t == "--model-gpu" and gpus:
                model_gpus = gpus
            i = j
            continue
        i += 1
    if "target_gpus" not in info and model_gpus:
        info["target_gpus"] = model_gpus
    return info

def collect_fgpo_processes(default_lane="", gpus=None):
    raw = sh("ps -eo pid=,ppid=,pcpu=,pmem=,rss=,etime=,args= 2>/dev/null")
    pat_tokens = (
        "fgpo.runner", "fgpo.tree_runner", "fgpo.envs", "capped_study.py", "launch_lane.sh",
        "launch-webshop.sh", "webshop-3-4",
        "run-queue-", "run_colab_seq", "run_fgpo_study", "fgpo_gcs_checkpoint",
        ".fgpo-colab-supervisor.sh", "run_fgpo_host_b_webshop_queue.sh", "launch-host-a-textcraft.sh",
        "run_fgpo_host_b_webshop150_queue.sh", "run_fgpo_host_1p5b_queue.sh",
        "study-textcraft", "launch-host-a", "run_fgpo_host_",
    )
    pid_to_gpus = {}
    if gpus:
        for g in gpus:
            idx = g.get("index")
            if not isinstance(idx, int):
                continue
            for p_str in (g.get("pid_mem_mib") or {}):
                if str(p_str).isdigit():
                    pid_to_gpus.setdefault(int(p_str), set()).add(idx)
            for pid_v in (g.get("pids") or []):
                if str(pid_v).isdigit():
                    pid_to_gpus.setdefault(int(pid_v), set()).add(idx)

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

    ps_by_pid = {rec[0]: rec for rec in all_ps}
    # Capture direct parent shell supervisors/queues of matched runners (never PID 1 or sshd)
    for rec in list(all_ps):
        if rec[0] in matched_pids and rec[1] > 1 and rec[1] in ps_by_pid:
            p_cmd = ps_by_pid[rec[1]][6]
            if ("bash" in p_cmd or "sh " in p_cmd) and "sshd" not in p_cmd and "python3 -" not in p_cmd:
                matched_pids.add(rec[1])
    # Also capture direct child waiters.
    for rec in all_ps:
        if rec[1] in matched_pids and ("tail " in rec[6] and "--pid=" in rec[6]):
            matched_pids.add(rec[0])

    rows = []
    for pid_i, ppid_i, pcpu, pmem, rss_s, etime, cmd in all_ps:
        if pid_i not in matched_pids:
            continue
        rss_mib = round(int(rss_s) / 1024.0, 1) if rss_s.isdigit() else None
        role = "process"
        if "tail " in cmd and "--pid=" in cmd:
            role = "handoff_waiter"
        elif "fgpo.envs" in cmd:
            role = "fgpo.envs"
        elif "capped_study.py" in cmd:
            role = "capped_study / fgpo.runner"
        elif "fgpo.tree_runner" in cmd:
            role = "fgpo.tree_runner"
        elif "fgpo.runner" in cmd:
            role = "fgpo.runner"
        elif any(name in cmd for name in ("run-queue-", "launch-webshop", "webshop-3-4",
                                          "run_fgpo_host_b_webshop_queue.sh", "launch-host-a-textcraft.sh",
                                          "run_fgpo_host_b_webshop150_queue.sh", "run_fgpo_host_1p5b_queue.sh",
                                          "launch-host-a", "queue")):
            role = "queue_controller"
        elif "launch_lane.sh" in cmd or "supervisor" in cmd or "run_colab_seq" in cmd or cmd.startswith(("bash ", "/bin/bash ", "sh ")):
            role = "supervisor"

        item = {
            "pid": str(pid_i),
            "ppid": str(ppid_i),
            "cpu_pct": pcpu,
            "mem_pct": pmem,
            "rss_mib": rss_mib,
            "affinity": read_cpu_affinity(pid_i),
            "elapsed": etime,
            "role": role,
            "lane": default_lane,
            "cmd": cmd,
        }
        if "runner" in role or role in ("queue_controller", "supervisor"):
            item.update(parse_runner_cli(cmd))
        if "runner" in role and item.get("target_config"):
            try:
                cfg_obj = json.loads(pathlib.Path(item["target_config"]).read_text(encoding="utf-8", errors="replace"))
                if isinstance(cfg_obj, dict):
                    if not item.get("target_benchmark") and isinstance(cfg_obj.get("benchmarks"), list) and cfg_obj["benchmarks"]:
                        item["target_benchmark"] = cfg_obj["benchmarks"][0]
                    if not item.get("target_method") and isinstance(cfg_obj.get("methods"), list) and cfg_obj["methods"]:
                        item["target_method"] = cfg_obj["methods"][0]
                    if not item.get("target_seeds") and isinstance(cfg_obj.get("seeds"), list):
                        item["target_seeds"] = [int(s) for s in cfg_obj["seeds"] if str(s).isdigit()]
                    if not item.get("target_suite") and cfg_obj.get("suite"):
                        item["target_suite"] = str(cfg_obj["suite"])
            except Exception:
                pass

        # Priority 1: nvidia-smi PID -> GPU mapping
        gpu_ids = sorted(pid_to_gpus.get(pid_i) or [])
        # Priority 2: parsed --gpu CLI args
        if not gpu_ids and item.get("target_gpus"):
            gpu_ids = sorted({int(x) for x in item["target_gpus"] if str(x).isdigit()})
        # Priority 3: substring fallback
        if not gpu_ids:
            if "3-4" in cmd or "3+4" in cmd or "--gpu 3 4" in cmd or "launch-webshop" in cmd or "study-webshop" in cmd:
                gpu_ids = [3, 4]
            elif "1-7" in cmd or "1+7" in cmd or "--gpu 1 7" in cmd:
                gpu_ids = [1, 7]
            elif "launch_lane.sh 2" in cmd or "gpu-2" in cmd or "webshop_queue.sh 2" in cmd or "webshop150_queue.sh 2" in cmd:
                gpu_ids = [2]
            elif "launch_lane.sh 0" in cmd or "gpu-0" in cmd or "webshop_queue.sh 0" in cmd or "webshop150_queue.sh 0" in cmd:
                gpu_ids = [0]
            elif "gpu-7" in cmd:
                gpu_ids = [7]
            elif "gpu-3" in cmd:
                gpu_ids = [3]
            elif "gpu-1" in cmd:
                gpu_ids = [1]
        item["gpu_ids"] = gpu_ids
        rows.append(item)

    by_pid = {r["pid"]: r for r in rows}
    # Parent -> child first (env workers inherit runner's GPU), then child -> parent if parent had none
    for _ in range(3):
        for r in rows:
            if not r["gpu_ids"] and r["ppid"] in by_pid and by_pid[r["ppid"]]["gpu_ids"]:
                r["gpu_ids"] = list(by_pid[r["ppid"]]["gpu_ids"])
        for r in rows:
            if r["gpu_ids"] and r["ppid"] in by_pid and not by_pid[r["ppid"]]["gpu_ids"]:
                by_pid[r["ppid"]]["gpu_ids"] = list(r["gpu_ids"])

    for r in rows:
        gids = sorted(set(r["gpu_ids"]))
        r["gpu_ids"] = gids
        r["gpu_key"] = "+".join(str(x) for x in gids)
        if default_lane:
            r["lane"] = default_lane
        elif len(gids) >= 2:
            r["lane"] = f"L40 {r['gpu_key']}"
        elif len(gids) == 1:
            r["lane"] = f"GPU {gids[0]}"
        else:
            r["lane"] = ""
    return rows

def matches_scheduled_probe(spec, runner):
    seeds = runner.get("target_seeds") or []
    sub = spec.get("output_subdir") or ""
    return (runner.get("target_suite", "comparison") == spec["suite"]
            and runner.get("target_benchmark") == spec["benchmark"]
            and runner.get("target_method") == spec["method"]
            and spec["seed"] in seeds
            and (not sub or pathlib.Path(runner.get("target_output_dir") or "").name == sub))

def read_costs_incremental(cj, rounds_done):
    key_hash = hashlib.sha1(str(cj).encode("utf-8")).hexdigest()[:16]
    cache_path = pathlib.Path(f"/tmp/fgpo_mon_v5_{key_hash}.json")
    st = cj.stat()
    cache = None
    if cache_path.is_file():
        try:
            c = json.loads(cache_path.read_text())
            if (c.get("schema_version") == 5
                    and c.get("ino") == st.st_ino and c.get("dev") == st.st_dev
                    and "completed_group_events" in c and 0 <= c.get("offset", 0) <= st.st_size):
                cache = c
        except Exception:
            cache = None
    if cache is None:
        cache = {
            "schema_version": 5,
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
            "completed_group_events": {},
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
                    if role == "ledger.restore":
                        # The cost journal keeps work performed after the restored checkpoint.
                        checkpoint_event = ev["event_index"] - det["imported_events"] - 1
                        cache["completed_group_events"] = {
                            group: index for group, index in cache["completed_group_events"].items()
                            if index <= checkpoint_event
                        }
                    if (role == "rollout.group" or (role == "rollout.tree_group" and det.get("status") == "OK")) and isinstance(gid, str):
                        parts = gid.split(":")
                        if len(parts) >= 4:
                            cache["completed_group_events"][gid] = ev["event_index"]
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

    cur_round_groups_done = sum(
        gid.split(":")[2] == str(rounds_done) for gid in cache["completed_group_events"]
    )
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

def compute_round_velocity_sec(wall_times):
    if not wall_times:
        return None
    if len(wall_times) == 1:
        return round(float(wall_times[0]), 2) if wall_times[0] and wall_times[0] > 0 else None
    deltas = []
    tail = wall_times[-7:]
    for i in range(1, len(tail)):
        if tail[i] is not None and tail[i - 1] is not None:
            dt = float(tail[i]) - float(tail[i - 1])
            if dt > 1.0:
                deltas.append(dt)
    if deltas:
        return round(sum(deltas[-5:]) / len(deltas[-5:]), 2)
    return round(float(wall_times[-1]) / len(wall_times), 2) if wall_times[-1] and wall_times[-1] > 0 else None

def _clean_eval_summary(obj):
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if k in ("task_types", "recipe_depths") and isinstance(v, dict):
            sub_out = {}
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    sub_out[sk] = {
                        kk: vv for kk, vv in sv.items()
                        if kk not in ("missing_tasks", "unexpected_tasks")
                    }
                else:
                    sub_out[sk] = sv
            out[k] = sub_out
        elif k in ("missing_tasks", "unexpected_tasks"):
            continue
        else:
            out[k] = v
    return out

def _extract_cell_config(cfg_f, bm_hint=""):
    if not cfg_f.is_file():
        return None
    try:
        cfg_obj = json.loads(cfg_f.read_text())
        if not isinstance(cfg_obj, dict):
            return None
        mp = str(cfg_obj.get("model_path") or "")
        mp_low = mp.lower()
        backbone = "1.5B" if "1.5b" in mp_low else ("7B" if "7b" in mp_low else None)
        bms = cfg_obj.get("benchmarks")
        bm_cfg = bm_hint or (bms[0] if isinstance(bms, list) and bms else "")
        out_tok_map = cfg_obj.get("output_tokens_by_benchmark") or {}
        out_tok = out_tok_map.get(bm_cfg) if isinstance(out_tok_map, dict) else None
        if out_tok is None:
            out_tok = cfg_obj.get("output_tokens")
        hor_map = cfg_obj.get("horizons") or {}
        horizon = hor_map.get(bm_cfg) if isinstance(hor_map, dict) else cfg_obj.get("horizon")
        if horizon is None:
            horizon = cfg_obj.get("horizon")
        return {
            "model_name": pathlib.Path(mp).name if mp else None,
            "backbone": backbone,
            "learning_rate": cfg_obj.get("learning_rate"),
            "credit_radius": cfg_obj.get("credit_radius"),
            "credit_prior_weight": cfg_obj.get("credit_prior_weight"),
            "credit_transfer_max": cfg_obj.get("credit_transfer_max"),
            "entropy_coef": cfg_obj.get("entropy_coef"),
            "kl_coef": cfg_obj.get("kl_coef"),
            "rounds": cfg_obj.get("rounds"),
            "horizon": horizon,
            "output_tokens": out_tok,
            "study_protocol": cfg_obj.get("study_protocol"),
        }
    except Exception:
        return None

def read_cell_dir(d, default_splits, include_du=False):
    if not d.exists():
        return {"exists": False}
    s = {"exists": True, "probe_epoch": time.time()}
    if include_du:
        s["du"] = sh(f"du -sh '{d}'/* 2>/dev/null || true")
    tj = d / "training.jsonl"
    cj = d / "costs.jsonl"
    csum_file = d / "costs_summary.json"
    ts = d / "training_summary.json"
    cs = d / "cell_summary.json"
    rs_file = d / "reporting_summary.json"
    cfg_file = d / "config.json"

    bm_guess = d.parent.parent.name if d.parent and d.parent.parent else ""
    cell_cfg = _extract_cell_config(cfg_file, bm_guess)
    if cell_cfg:
        s["cell_config"] = cell_cfg

    mtimes = []
    for pf in (tj, cj, csum_file, ts, cs, rs_file):
        try:
            if pf.is_file():
                mtimes.append(pf.stat().st_mtime)
        except OSError:
            pass
    for ep in d.glob("evaluation-*.jsonl"):
        try:
            mtimes.append(ep.stat().st_mtime)
        except OSError:
            pass
    s["last_mtime"] = max(mtimes) if mtimes else None

    rounds = []
    wall_times = []
    last_raw_dev_sec = None
    if tj.is_file():
        for ln in tj.read_text(errors="replace").splitlines():
            if ln.strip():
                try:
                    raw_row = json.loads(ln)
                    c_blk = raw_row.get("costs") or {}
                    w_sec = c_blk.get("wall_seconds")
                    if w_sec is not None:
                        wall_times.append(float(w_sec))
                    dev_sec = c_blk.get("allocated_device_seconds")
                    if dev_sec is not None:
                        last_raw_dev_sec = float(dev_sec)
                    rounds.append(project_training_round(raw_row, bm_guess))
                except Exception:
                    pass
    s["training_rounds"] = rounds
    s["recent_sec_per_round"] = compute_round_velocity_sec(wall_times)
    cur_round = len(rounds)
    groups_dir = d / "groups"
    if groups_dir.is_dir():
        s["current_round_group_files"] = len(list(groups_dir.glob(f"round-{cur_round}-group-*.pt")))
    else:
        s["current_round_group_files"] = 0

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
    if ts.is_file():
        try:
            ts_obj = json.loads(ts.read_text())
            s["training_summary"] = {k: v for k, v in ts_obj.items() if k not in ("rounds", "history")} if isinstance(ts_obj, dict) else ts_obj
        except Exception:
            pass
    if cs.is_file():
        try:
            cs_obj = json.loads(cs.read_text())
            if isinstance(cs_obj, dict):
                cleaned_cs = {k: v for k, v in cs_obj.items() if k not in ("rounds", "history")}
                if isinstance(cleaned_cs.get("evaluation"), dict):
                    cleaned_cs["evaluation"] = {
                        ek: _clean_eval_summary(ev) for ek, ev in cleaned_cs["evaluation"].items()
                    }
                s["cell_summary"] = cleaned_cs
            else:
                s["cell_summary"] = cs_obj
        except Exception:
            pass
    if rs_file.is_file():
        try:
            rs_obj = json.loads(rs_file.read_text())
            if isinstance(rs_obj, dict):
                cur_cs = s.get("cell_summary")
                if not isinstance(cur_cs, dict):
                    cleaned_rs = {k: v for k, v in rs_obj.items() if k not in ("rounds", "history")}
                    if isinstance(cleaned_rs.get("evaluation"), dict):
                        cleaned_rs["evaluation"] = {
                            ek: _clean_eval_summary(ev) for ek, ev in cleaned_rs["evaluation"].items()
                        }
                    s["cell_summary"] = cleaned_rs
                else:
                    if isinstance(rs_obj.get("evaluation"), dict):
                        ev_tgt = cur_cs.setdefault("evaluation", {})
                        for ek, ev in rs_obj["evaluation"].items():
                            ce = _clean_eval_summary(ev)
                            if ek not in ev_tgt or not isinstance(ev_tgt[ek], dict):
                                ev_tgt[ek] = ce
                            elif isinstance(ce, dict):
                                for bk in ("task_types", "recipe_depths", "SR", "Score", "score"):
                                    if bk not in ev_tgt[ek] and bk in ce:
                                        ev_tgt[ek][bk] = ce[bk]
                    if isinstance(rs_obj.get("costs"), dict) and "costs" not in cur_cs:
                        cur_cs["costs"] = rs_obj["costs"]
        except Exception:
            pass

    cs_blk = s.get("cell_summary") if isinstance(s.get("cell_summary"), dict) else {}
    alloc_dev_sec = (
        ((cs_blk.get("costs") or {}).get("allocated_device_seconds"))
        or (((cs_blk.get("training") or {}).get("costs") or {}).get("allocated_device_seconds"))
        or last_raw_dev_sec
    )
    if alloc_dev_sec is not None:
        try:
            s["allocated_device_seconds"] = round(float(alloc_dev_sec), 2)
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
                ev_info["summary"] = _clean_eval_summary(json.loads(ev_sum.read_text()))
            except Exception:
                pass
        ev_rec_p = d / split / "evaluation_records.json"
        if bm_guess == "webshop" and ev_rec_p.is_file():
            try:
                if ev_rec_p.stat().st_size < 2 * 1024 * 1024:
                    recs = json.loads(ev_rec_p.read_text())
                    if isinstance(recs, list) and recs:
                        zero_cnt = 0
                        nonzero_cnt = 0
                        purch_cnt = 0
                        has_purch = False
                        for rr in recs:
                            oe = rr.get("official_evaluation") or {}
                            sc = oe.get("score")
                            if sc is None:
                                sc = rr.get("score")
                            if sc is not None:
                                if float(sc) == 0.0:
                                    zero_cnt += 1
                                else:
                                    nonzero_cnt += 1
                            if "purchased" in oe:
                                has_purch = True
                                if oe.get("purchased"):
                                    purch_cnt += 1
                        ev_info["zero_score_count"] = zero_cnt
                        ev_info["nonzero_score_count"] = nonzero_cnt
                        if has_purch:
                            ev_info["purchased_count"] = purch_cnt
            except Exception:
                pass
        if ev_info:
            s["evaluations"][split] = ev_info
    return s

def discover_unscheduled_cells(candidate_globs, procs, scheduled_dirs, archived_dirs, existing_cell_keys, default_gpu_str):
    discovered = []
    discovered_map = {}
    seen_dirs = set(scheduled_dirs) | set(archived_dirs)

    cand_dirs = []
    for p in procs:
        if "runner" not in (p.get("role") or ""):
            continue
        od = p.get("target_output_dir")
        tsuite = p.get("target_suite") or "comparison"
        tbench = p.get("target_benchmark")
        tmeth = p.get("target_method")
        tseeds = p.get("target_seeds") or []
        if od:
            od_p = pathlib.Path(od)
            if tbench and tmeth:
                for sd in tseeds:
                    cand_dirs.append(od_p / tsuite / tbench / tmeth / f"seed-{sd}")
            if od_p.is_dir():
                cand_dirs.extend(sorted(od_p.glob("*/*/*/seed-*")))

    for pat in candidate_globs:
        for hit in sorted(glob.glob(pat)):
            cand_dirs.append(pathlib.Path(hit))

    used_keys = set(existing_cell_keys)
    for cdir in cand_dirs:
        try:
            if not cdir.is_dir():
                continue
            resolved = str(cdir.resolve())
            abspath = os.path.abspath(str(cdir))
            if resolved in seen_dirs or abspath in seen_dirs:
                continue
            seen_dirs.add(resolved)
            seen_dirs.add(abspath)
            if not cdir.name.startswith("seed-"):
                continue
            seed_str = cdir.name[len("seed-"):]
            if not seed_str.isdigit():
                continue
            seed_i = int(seed_str)
            method = cdir.parent.name
            benchmark = cdir.parents[1].name
            suite = cdir.parents[2].name
            study_root = cdir.parents[3]
            if suite not in ("comparison", "ablation") or benchmark not in ("alfworld", "textcraft", "webshop"):
                continue
            if not any((cdir / fn).is_file() for fn in ("training.jsonl", "costs.jsonl", "cell_summary.json", "costs_summary.json")):
                continue

            cfg_obj = {}
            cfg_f = cdir / "config.json"
            if cfg_f.is_file():
                try:
                    cfg_obj = json.loads(cfg_f.read_text())
                except Exception:
                    cfg_obj = {}

            tot_rounds = int(cfg_obj.get("rounds") or (50 if benchmark == "textcraft" else (150 if benchmark == "webshop" else 60)))
            hor_map = cfg_obj.get("horizons")
            horizon = int(
                (hor_map.get(benchmark) if isinstance(hor_map, dict) and hor_map.get(benchmark) is not None else None)
                or cfg_obj.get("horizon")
                or (30 if benchmark == "textcraft" else (50 if benchmark == "alfworld" else 15))
            )
            ev_splits_cfg = cfg_obj.get("evaluation_splits")
            if isinstance(ev_splits_cfg, dict) and ev_splits_cfg.get(benchmark):
                def_splits = [str(x) for x in ev_splits_cfg[benchmark]]
            elif isinstance(ev_splits_cfg, list) and ev_splits_cfg:
                def_splits = [str(x) for x in ev_splits_cfg]
            else:
                def_splits = [{"textcraft": "val", "alfworld": "valid_unseen", "webshop": "test"}[benchmark]]

            phys_list = cfg_obj.get("compute_physical_gpu_ids") or cfg_obj.get("physical_gpu_ids")
            phys_gpu_str = None
            dev_count = 1
            if isinstance(phys_list, list) and phys_list:
                dev_count = len(phys_list)
                phys_gpu_str = "+".join(str(x) for x in phys_list)

            if not phys_gpu_str:
                for p in procs:
                    if "runner" in (p.get("role") or "") and p.get("target_output_dir"):
                        try:
                            if pathlib.Path(p["target_output_dir"]).resolve() == study_root.resolve():
                                if p.get("gpu_key"):
                                    phys_gpu_str = p["gpu_key"]
                                    dev_count = max(1, len(p.get("gpu_ids") or []))
                                elif p.get("target_gpus"):
                                    phys_gpu_str = "+".join(p["target_gpus"])
                                    dev_count = len(p["target_gpus"])
                                break
                        except Exception:
                            pass
            if not phys_gpu_str:
                phys_gpu_str = default_gpu_str
                dev_count = 2 if "+" in default_gpu_str else 1

            base_ck = f"{suite}/{benchmark}/{method}/seed-{seed_i}"
            mp_low = str(cfg_obj.get("model_path") or "").lower()
            sr_low = study_root.name.lower()
            is_15b = ("1.5b" in mp_low) or ("1p5b" in sr_low)
            variant = ""
            ck = base_ck
            if ck in used_keys or is_15b:
                if is_15b and "joint" in sr_low:
                    variant = "1p5b-joint"
                elif is_15b:
                    variant = "1p5b"
                else:
                    variant = study_root.name
                    for pref in ("study-textcraft-", "study-", "output-"):
                        if variant.startswith(pref) and len(variant) > len(pref):
                            variant = variant[len(pref):]
                            break
                ck = f"{base_ck}@{variant}"
                n_try = 2
                while ck in used_keys:
                    ck = f"{base_ck}@{variant}-{n_try}"
                    n_try += 1
            used_keys.add(ck)

            cd = read_cell_dir(cdir, def_splits, include_du=False)
            cd["output_subdir"] = study_root.name
            discovered_map[ck] = cd
            discovered.append({
                "cell_key": ck,
                "suite": suite,
                "benchmark": benchmark,
                "method": method,
                "seed": seed_i,
                "output_subdir": study_root.name,
                "variant": variant,
                "physical_gpu_ids": phys_gpu_str,
                "device_count": dev_count,
                "total_rounds": tot_rounds,
                "horizon": horizon,
                "official_eval": "+".join(def_splits),
                "backbone": ((cd.get("cell_config") or {}).get("backbone") or ("1.5B" if is_15b else "7B")),
            })
        except Exception:
            continue
    return discovered, discovered_map
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
    out_p = pathlib.Path(out_dir) if out_dir else None
    if out_p is not None and out_p.is_dir():
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

gpus = collect_gpus()
procs = collect_fgpo_processes("Colab GPU 0", gpus=gpus)
annotate_gpu_fgpo_vram(gpus, procs)
has_runner_proc = any("runner" in (p.get("role") or "") for p in procs)
for g in gpus:
    g["role_slot"] = g["name"]
    g["active_in_experiment"] = bool(g["fgpo_mem_used_mib"] or has_runner_proc)
disks = collect_disks(["/content"])

data = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "probe_epoch": time.time(),
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
    "discovered_cells": [],
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

candidate_roots = []
if OUT_DIR and os.path.isdir(OUT_DIR):
    candidate_roots.append(pathlib.Path(OUT_DIR))
for p in procs:
    od_arg = p.get("target_output_dir") or ""
    if od_arg and os.path.isdir(od_arg):
        od_p = pathlib.Path(od_arg)
        if od_p not in candidate_roots:
            candidate_roots.append(od_p)
        for sib in sorted(od_p.parent.glob("study*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if sib.is_dir() and sib not in candidate_roots:
                candidate_roots.append(sib)
for pat in ("/apps/users/*/fgpo-v4-formal-host-a-*/study*", "/content/study*", "/content/outputs*", "/content/output*"):
    for hit in sorted(glob.glob(pat), key=lambda x: os.path.getmtime(x), reverse=True):
        hp = pathlib.Path(hit)
        if hp.is_dir() and hp not in candidate_roots:
            candidate_roots.append(hp)

explicit_subdirs = {(s.get("output_subdir") or "").strip() for s in CELL_SPECS if (s.get("output_subdir") or "").strip()}
out_name = pathlib.Path(OUT_DIR).name if OUT_DIR else ""

scheduled_dirs = set()
target_cells = {}
for spec in CELL_SPECS:
    ck = spec.get("cell_key") or f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
    sub = (spec.get("output_subdir") or "").strip()
    rel = pathlib.Path(spec["suite"]) / spec["benchmark"] / spec["method"] / f"seed-{spec['seed']}"
    chosen_dir = None
    if sub:
        for cr in candidate_roots:
            if cr.name == sub and (cr / rel).is_dir():
                chosen_dir = cr / rel
                break
            if (cr / sub / rel).is_dir():
                chosen_dir = cr / sub / rel
                break
        if chosen_dir is None:
            for cr in candidate_roots:
                if cr.name == sub:
                    chosen_dir = cr / rel
                    break
    elif OUT_DIR and out_name not in explicit_subdirs:
        cand = pathlib.Path(OUT_DIR) / rel
        if cand.is_dir():
            chosen_dir = cand
    if chosen_dir is not None and chosen_dir.exists():
        scheduled_dirs.add(str(chosen_dir.resolve()))
        scheduled_dirs.add(os.path.abspath(str(chosen_dir)))
        inc_du = (spec["suite"] == SUITE and spec["benchmark"] == BENCHMARK and spec["method"] == METHOD)
        cd_obj = read_cell_dir(chosen_dir, spec["official_eval"].split("+"), include_du=inc_du)
        if sub:
            cd_obj["output_subdir"] = sub
        data["cells"][ck] = cd_obj
        if inc_du:
            target_cells[spec["seed"]] = ck
    else:
        data["cells"][ck] = {"exists": False}

if OUT_DIR:
    out_root = pathlib.Path(OUT_DIR)
    for seed in SEEDS:
        if seed in target_cells:
            data["seeds"][str(seed)] = data["cells"][target_cells[seed]]
            continue
        ck = f"{SUITE}/{BENCHMARK}/{METHOD}/seed-{seed}"
        if ck in data["cells"]:
            data["seeds"][str(seed)] = data["cells"][ck]
            continue
        cdir = out_root / SUITE / BENCHMARK / METHOD / f"seed-{seed}"
        scheduled_dirs.add(str(cdir.resolve()))
        scheduled_dirs.add(os.path.abspath(str(cdir)))
        data["seeds"][str(seed)] = read_cell_dir(cdir, eval_splits, include_du=True)
        data["cells"][ck] = data["seeds"][str(seed)]

disc_globs = [
    "/apps/users/*/fgpo-v4-formal-host-a-*/study*/*/*/*/seed-*",
    "/content/study*/*/*/*/seed-*",
    "/content/outputs*/*/*/*/seed-*",
    "/content/outputs/*/*/*/*/seed-*",
]
if OUT_DIR:
    disc_globs.append(str(pathlib.Path(OUT_DIR) / "*/*/*/seed-*"))
disc_list, disc_map = discover_unscheduled_cells(
    disc_globs,
    procs,
    scheduled_dirs,
    set(),
    set(data["cells"].keys()),
    "0",
)
data["discovered_cells"] = disc_list
data["cells"].update(disc_map)

resolved_log = LOG_PATH
active_spec = None
for spec in CELL_SPECS:
    ck = spec.get("cell_key") or f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
    cd = data["cells"].get(ck) or {}
    if cd.get("exists") and not cd.get("cell_summary"):
        active_spec = spec
        break

if not resolved_log or resolved_log == "auto":
    fd_logs = []
    for p in procs:
        if "runner" in (p.get("role") or "") and p.get("pid"):
            for fp_s in read_proc_log_fds(p["pid"]):
                if fp_s not in fd_logs:
                    fd_logs.append(fp_s)
    if fd_logs:
        resolved_log = fd_logs[0]
    else:
        log_cands = glob.glob("/content/fgpo-*/*.log") + glob.glob("/content/logs/*.log")
        if OUT_DIR:
            log_cands += glob.glob(f"{OUT_DIR}/**/*.log", recursive=True)
        freshest_logs = sorted((p for p in log_cands if os.path.isfile(p)), key=lambda p: os.path.getmtime(p), reverse=True)
        if freshest_logs:
            resolved_log = freshest_logs[0]
        elif active_spec:
            resolved_log = f"/content/logs/fgpo_{active_spec['suite']}_{active_spec['method']}_{active_spec['benchmark']}_s{active_spec['seed']}.log"
        else:
            resolved_log = f"/content/logs/fgpo_{SUITE}_{METHOD}_{BENCHMARK}_s0.log"

seed_tail = sh(f"tail -n 50 '{resolved_log}' 2>/dev/null || true")
ev_lines = []
if active_spec:
    ack = active_spec.get("cell_key") or f"{active_spec['suite']}/{active_spec['benchmark']}/{active_spec['method']}/seed-{active_spec['seed']}"
    acd = data["cells"].get(ack) or {}
    for r in (acd.get("training_rounds") or [])[-3:]:
        m = r.get("metrics") or {}
        ev_lines.append(
            f"[training R{r.get('round')}] succ={r.get('successes')}/{r.get('trajectories')} opt={r.get('optimizer_steps')} loss={m.get('loss')} kl={m.get('reference_kl')}"
        )
    for ev in (acd.get("recent_cost_events") or [])[-8:]:
        ev_lines.append(
            f"[costs #{ev.get('event_index')} t={ev.get('elapsed')}s] role={ev.get('role')} sec={ev.get('seconds')} steps={ev.get('env_steps') or 0} group={ev.get('group_id') or '-'}"
        )
combined_tail = "\n".join([x for x in (seed_tail, "\n".join(ev_lines)) if x])
sup_log = "/content/logs/fgpo_comparison_fgpo_supervisor.log"
sup_tail = sh(f"tail -n 35 '{sup_log}' 2>/dev/null || true")
data["log_path"] = resolved_log
data["log_tail"] = combined_tail or sup_tail
data["log_tails"]["Colab GPU 0 · Active Seed"] = {"path": resolved_log, "text": combined_tail or "(暂无日志输出)"}
if sup_tail:
    data["log_tails"]["Colab GPU 0 · GCS Supervisor"] = {"path": sup_log, "text": sup_tail}

print("__FGPO_JSON_START__")
print(json.dumps(data, ensure_ascii=False))
print("__FGPO_JSON_END__")
'''

HOST_A_PROBE_BODY = r'''
DEPLOY_HINT = os.environ.get("FGPO_DEPLOY_DIR", "auto")
CELL_SPECS = json.loads(os.environ["FGPO_CELL_SPECS"])
relocated_specs = [spec for spec in CELL_SPECS if spec.get("status_snapshot") == "relocated"]
CELL_SPECS = [spec for spec in CELL_SPECS if spec.get("status_snapshot") != "relocated"]

all_gpus = collect_gpus()
procs = collect_fgpo_processes("", gpus=all_gpus)
annotate_gpu_fgpo_vram(all_gpus, procs)

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

proc_gpu_set = set()
proc_gpu_keys = set()
for p in procs:
    for gid in (p.get("gpu_ids") or []):
        proc_gpu_set.add(gid)
    if p.get("gpu_key"):
        proc_gpu_keys.add(p["gpu_key"])
single_gpu_active_specs = [
    s for s in CELL_SPECS
    if s.get("status_snapshot") != "deferred" and "+" not in str(s.get("physical_gpu_ids") or "")
]
single_gpu_set = {
    int(s["physical_gpu_ids"]) for s in single_gpu_active_specs
    if str(s.get("physical_gpu_ids") or "").isdigit()
}
gpu3_single_planned = any(
    "gpu 3" in str(s.get("sampling_layout") or "").lower()
    or "gpu 3" in str(s.get("dependency") or "").lower()
    for s in CELL_SPECS if s.get("status_snapshot") != "deferred"
)

usable_gpus = []
excluded_gpus = []
for g in all_gpus:
    idx = g["index"]
    if idx in (0, 2):
        g["role_slot"] = "Excluded (0, 2)"
        g["active_in_experiment"] = False
        excluded_gpus.append(g)
    else:
        has_live = (g.get("fgpo_mem_used_mib") or 0) > 0 or idx in proc_gpu_set
        if idx == 3:
            if "3" in proc_gpu_keys:
                g["role_slot"] = "GPU 3 (1.5B 单卡)"
            elif gpu3_single_planned:
                g["role_slot"] = "Queue A1 (3+4 / GPU 3)"
            else:
                g["role_slot"] = "Queue A1 (3+4)"
            g["sampling_layout"] = "dual_or_single_l40"
            g["active_in_experiment"] = True
        elif idx == 4:
            g["role_slot"] = "Queue A1 (3+4)"
            g["sampling_layout"] = "gpu4_backbone_and_head"
            g["active_in_experiment"] = True
        elif idx == 7:
            if "7" in proc_gpu_keys or 7 in single_gpu_set:
                g["role_slot"] = "GPU 7 (1.5B 单卡)"
            else:
                g["role_slot"] = "Queue A2 (1+7)"
            g["sampling_layout"] = "single_or_dual_l40"
            g["active_in_experiment"] = True
        elif idx == 1:
            g["role_slot"] = "Queue A2 (1+7)"
            g["sampling_layout"] = "tuned_dual_l40"
            g["active_in_experiment"] = True
        else:
            g["role_slot"] = "Active (Shared)" if has_live else "Idle/Shared"
            g["active_in_experiment"] = bool(has_live)
        usable_gpus.append(g)

numa_nodes = {}
for np in sorted(glob.glob("/sys/devices/system/node/node*/cpulist")):
    try:
        numa_nodes[pathlib.Path(np).parent.name] = pathlib.Path(np).read_text().strip()
    except Exception:
        pass

data = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "probe_epoch": time.time(),
    "deployment_dir": str(deploy_dir) if deploy_dir else None,
    "frozen_author_eval": read_frozen_author_evaluation(deploy_dir),
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
    "discovered_cells": [],
    "archived_cells": {},
    "log_tails": {},
    "config": {"protocol": "fgpo-r3-published-v4", "groups": 16, "trajectories": 8},
}

study_roots = []
for p in procs:
    od = p.get("target_output_dir")
    if od and os.path.isdir(od) and pathlib.Path(od) not in study_roots:
        study_roots.append(pathlib.Path(od))
if deploy_dir and deploy_dir.is_dir():
    for sp in sorted(deploy_dir.glob("study*"), key=lambda x: x.stat().st_mtime, reverse=True):
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

scheduled_dirs = set()
cell_study_roots = {}
for spec in CELL_SPECS:
    ck = spec["cell_key"]
    bm = spec["benchmark"]
    def_splits = spec["official_eval"].split("+")
    ordered_roots = ([deploy_dir / spec["output_subdir"]]
                     if deploy_dir and spec.get("output_subdir") else list(study_roots))
    if bm == "webshop":
        ordered_roots.sort(key=lambda r: 0 if "webshop" in r.name else 1)
    elif bm == "textcraft":
        ordered_roots.sort(key=lambda r: 0 if "textcraft" in r.name else 1)
    else:
        ordered_roots.sort(key=lambda r: 1 if ("webshop" in r.name or "textcraft" in r.name) else 0)
    chosen_dir = None
    chosen_root = None
    for sr in ordered_roots:
        cand = sr / spec["suite"] / bm / spec["method"] / f"seed-{spec['seed']}"
        if cand.exists():
            chosen_dir = cand
            chosen_root = sr
            break
    if chosen_dir is None and ordered_roots:
        chosen_root = ordered_roots[0]
        chosen_dir = chosen_root / spec["suite"] / bm / spec["method"] / f"seed-{spec['seed']}"
    if chosen_dir is not None:
        scheduled_dirs.add(str(chosen_dir.resolve()))
        scheduled_dirs.add(os.path.abspath(str(chosen_dir)))
        cell_study_roots[ck] = chosen_root
        data["cells"][ck] = read_cell_dir(chosen_dir, def_splits, include_du=False)
    else:
        data["cells"][ck] = {"exists": False}

# Archived Host A 7B GiGPO cells in <deploy>/study/comparison/alfworld/gigpo/seed-*
archived_dirs = set()
if deploy_dir and (deploy_dir / "study").is_dir():
    arch_base = deploy_dir / "study" / "comparison" / "alfworld" / "gigpo"
    if arch_base.is_dir():
        for sd_p in sorted(arch_base.glob("seed-*")):
            archived_dirs.add(str(sd_p.resolve()))
            archived_dirs.add(os.path.abspath(str(sd_p)))
            ack = f"comparison/alfworld/gigpo/{sd_p.name}"
            acell = read_cell_dir(sd_p, ["valid_unseen"], include_du=False)
            if acell.get("exists"):
                data["archived_cells"][ack] = acell

host_a_globs = [str(deploy_dir / "study*/*/*/*/seed-*")] if deploy_dir and deploy_dir.is_dir() else []
if deploy_dir and relocated_specs:
    for spec in relocated_specs:
        cdir = deploy_dir / spec["output_subdir"] / spec["cell_key"].split("@", 1)[0]
        scheduled_dirs.update((str(cdir.resolve()), os.path.abspath(str(cdir))))
disc_list, disc_map = discover_unscheduled_cells(
    host_a_globs,
    procs,
    scheduled_dirs,
    archived_dirs,
    set(data["cells"].keys()),
    "3+4",
)
data["discovered_cells"] = disc_list
data["cells"].update(disc_map)

# Dynamic log tails per Host A lane (dual-GPU and single-GPU slots, open FDs + launch/logs)
if deploy_dir and deploy_dir.is_dir():
    all_specs = list(CELL_SPECS) + list(disc_list)
    explicit_single_keys = {
        str(s.get("physical_gpu_ids")) for s in all_specs
        if s.get("status_snapshot") != "deferred" and "+" not in str(s.get("physical_gpu_ids") or "") and s.get("physical_gpu_ids")
    }
    lane_keys_ordered = []
    for cand_k in (
        ["3+4", "1+7", "7", "3"]
        + [str(s.get("physical_gpu_ids")) for s in all_specs if s.get("physical_gpu_ids")]
        + [p.get("gpu_key") for p in procs if p.get("gpu_key")]
    ):
        if cand_k and cand_k not in lane_keys_ordered:
            lane_keys_ordered.append(cand_k)

    for gpu_key in lane_keys_ordered:
        gpu_id_set = {int(x) for x in gpu_key.split("+") if x.isdigit()}
        if not gpu_id_set:
            continue
        is_dual = len(gpu_id_set) >= 2
        lane_specs = [s for s in all_specs if s.get("physical_gpu_ids") == gpu_key]
        lane_procs = [
            p for p in procs
            if p.get("gpu_key") == gpu_key
            or (is_dual and p.get("gpu_ids") and set(p["gpu_ids"]).issubset(gpu_id_set)
                and p.get("gpu_key") not in explicit_single_keys
                and any(matches_scheduled_probe(s, p) for s in lane_specs))
        ]
        # Skip idle synthetic keys unless standard 3+4/1+7 or has live process / active-plan cell
        has_non_deferred = any(s.get("status_snapshot") != "deferred" for s in lane_specs)
        if gpu_key not in ("3+4", "1+7") and not lane_procs and not has_non_deferred:
            continue

        lane_label = f"Host A · L40 {gpu_key}" if is_dual else f"Host A · GPU {gpu_key}"
        lane_runner = next((p for p in lane_procs if "runner" in (p.get("role") or "")), {})
        lane_sup = next((p for p in lane_procs if p.get("role") in ("queue_controller", "supervisor")), {})

        fd_logs = []
        for lp_proc in (lane_runner, lane_sup):
            if lp_proc.get("pid"):
                for fp_s in read_proc_log_fds(lp_proc["pid"]):
                    fp_o = pathlib.Path(fp_s)
                    if fp_o not in fd_logs:
                        fd_logs.append(fp_o)

        active_spec = None
        if lane_runner:
            active_spec = next((s for s in lane_specs if matches_scheduled_probe(s, lane_runner)), None)
            if active_spec is None:
                active_spec = next((s for s in all_specs if matches_scheduled_probe(s, lane_runner)), None)
        if active_spec is None and lane_sup and lane_sup.get("target_benchmark"):
            active_spec = next(
                (s for s in lane_specs if s.get("benchmark") == lane_sup.get("target_benchmark")
                 and s.get("status_snapshot") not in ("completed", "deferred")),
                None,
            )
        if active_spec is None:
            incomplete = [
                s for s in lane_specs
                if (data["cells"].get(s["cell_key"]) or {}).get("exists")
                and not (data["cells"].get(s["cell_key"]) or {}).get("cell_summary")
            ]
            incomplete.sort(key=lambda s: float((data["cells"].get(s["cell_key"]) or {}).get("last_mtime") or 0.0), reverse=True)
            active_spec = incomplete[0] if incomplete else (
                next((s for s in lane_specs if s.get("status_snapshot") not in ("completed", "deferred")), None)
            )

        search_log_dirs = []
        if lane_runner.get("target_output_dir"):
            rod = pathlib.Path(lane_runner["target_output_dir"])
            for sub in ("logs", "launch", ""):
                d_cand = rod / sub if sub else rod
                if d_cand.is_dir() and d_cand not in search_log_dirs:
                    search_log_dirs.append(d_cand)
        if active_spec and active_spec["cell_key"] in cell_study_roots and cell_study_roots[active_spec["cell_key"]]:
            asr = cell_study_roots[active_spec["cell_key"]]
            for sub in ("logs", "launch", ""):
                d_cand = asr / sub if sub else asr
                if d_cand.is_dir() and d_cand not in search_log_dirs:
                    search_log_dirs.append(d_cand)
        for sr in study_roots:
            for sub in ("logs", "launch"):
                d_cand = sr / sub
                if d_cand.is_dir() and d_cand not in search_log_dirs:
                    search_log_dirs.append(d_cand)
        for sub in ("logs", "launch"):
            d_cand = deploy_dir / sub
            if d_cand.is_dir() and d_cand not in search_log_dirs:
                search_log_dirs.append(d_cand)

        scored_logs = []
        seen_log_paths = set()
        for fp_o in fd_logs:
            try:
                st_m = fp_o.stat().st_mtime
                scored_logs.append((4, st_m, fp_o))
                seen_log_paths.add(str(fp_o.resolve()))
            except OSError:
                pass

        t_bm = (lane_runner.get("target_benchmark") or lane_sup.get("target_benchmark") or (active_spec or {}).get("benchmark") or "").lower()
        t_md = (lane_runner.get("target_method") or (active_spec or {}).get("method") or "").lower()
        raw_seeds = lane_runner.get("target_seeds") or [(active_spec or {}).get("seed")]
        t_sd = str(raw_seeds[0]) if raw_seeds and raw_seeds[0] is not None else ""
        runner_od_str = str(pathlib.Path(lane_runner["target_output_dir"]).resolve()) if lane_runner.get("target_output_dir") and os.path.isdir(lane_runner["target_output_dir"]) else ""
        spec_od_str = ""
        if active_spec and active_spec["cell_key"] in cell_study_roots and cell_study_roots[active_spec["cell_key"]]:
            try:
                spec_od_str = str(pathlib.Path(cell_study_roots[active_spec["cell_key"]]).resolve())
            except OSError:
                spec_od_str = ""

        lane_toks = tuple({gpu_key, gpu_key.replace("+", "-"), gpu_key.replace("+", "_"), f"gpu-{gpu_key}", f"gpu{gpu_key}"} if not is_dual else (gpu_key.replace("+", "-"), gpu_key.replace("+", "_"), gpu_key))
        other_toks = ("1-7", "1_7", "1+7") if gpu_key == "3+4" else (("3-4", "3_4", "3+4") if gpu_key == "1+7" else ())

        for ldir in search_log_dirs:
            for lf in ldir.glob("*.log"):
                try:
                    r_str = str(lf.resolve())
                    if r_str in seen_log_paths or not lf.is_file() or lf.stat().st_size == 0:
                        continue
                    nm = lf.name.lower()
                    if other_toks and any(ot in nm for ot in other_toks) and not any(lt in nm for lt in lane_toks):
                        continue
                    prio = 0
                    in_target_od = (runner_od_str and r_str.startswith(runner_od_str + "/")) or (spec_od_str and r_str.startswith(spec_od_str + "/"))
                    seed_hit = bool(t_sd and (f"seed-{t_sd}" in nm or f"s{t_sd}" in nm or f"_{t_sd}" in nm or f"-{t_sd}" in nm))
                    if in_target_od:
                        prio = 2
                        if seed_hit or f"queue-gpu-{gpu_key}" in nm:
                            prio = 3
                        elif t_md and t_md in nm:
                            prio = 3
                    elif t_bm and t_md and t_bm in nm and t_md in nm and seed_hit:
                        prio = 2
                    elif any(lt in nm for lt in lane_toks):
                        prio = 1
                    if prio > 0:
                        seen_log_paths.add(r_str)
                        scored_logs.append((prio, lf.stat().st_mtime, lf))
                except OSError:
                    continue

        scored_logs.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen_logs = [item[2] for item in scored_logs[:2]]

        text_parts = []
        for lf in chosen_logs:
            tail_txt = sh(f"tail -n 20 '{lf}' 2>/dev/null || true")
            if tail_txt:
                prefix = f"=== {lf.name} ===" if len(chosen_logs) > 1 else ""
                text_parts.append("\n".join([x for x in (prefix, tail_txt) if x]))

        ev_lines = []
        if active_spec:
            cd = data["cells"].get(active_spec["cell_key"]) or {}
            for r in (cd.get("training_rounds") or [])[-3:]:
                m = r.get("metrics") or {}
                ev_lines.append(
                    f"[training R{r.get('round')}] succ={r.get('successes')}/{r.get('trajectories')} opt={r.get('optimizer_steps')} loss={m.get('loss')} kl={m.get('reference_kl')}"
                )
            for ev in (cd.get("recent_cost_events") or [])[-8:]:
                ev_lines.append(
                    f"[costs #{ev.get('event_index')} t={ev.get('elapsed')}s] role={ev.get('role')} sec={ev.get('seconds')} steps={ev.get('env_steps') or 0} group={ev.get('group_id') or '-'}"
                )
        if ev_lines:
            text_parts.append("\n".join(ev_lines))

        primary_path = chosen_logs[0] if chosen_logs else (search_log_dirs[0] if search_log_dirs else (deploy_dir / "logs"))
        data["log_tails"][lane_label] = {
            "path": str(primary_path),
            "text": "\n".join([x for x in text_parts if x]) or "(暂无日志输出)",
        }

print("__FGPO_JSON_START__")
print(json.dumps(data, ensure_ascii=False))
print("__FGPO_JSON_END__")
'''

HOST_B_PROBE_BODY = r'''
DEPLOY_HINT = os.environ.get("FGPO_DEPLOY_DIR", "auto")
CELL_SPECS = json.loads(os.environ["FGPO_CELL_SPECS"])

all_gpus = collect_gpus()
procs = collect_fgpo_processes("", gpus=all_gpus)
annotate_gpu_fgpo_vram(all_gpus, procs)

deploy_dir = None
if DEPLOY_HINT and DEPLOY_HINT != "auto" and os.path.isdir(DEPLOY_HINT):
    deploy_dir = pathlib.Path(DEPLOY_HINT)
else:
    for p in procs:
        out_arg = p.get("target_output_dir") or ""
        if out_arg:
            cand = pathlib.Path(out_arg)
            if cand.name.startswith("output") and cand.parent.is_dir():
                deploy_dir = cand.parent
                break
    if deploy_dir is None:
        home_ws = str(pathlib.Path.home() / "workspace" / "fgpo-v4-formal-host-b-*")
        cands = sorted(glob.glob(home_ws) + glob.glob("/home/*/workspace/fgpo-v4-formal-host-b-*"), key=lambda x: os.path.getmtime(x), reverse=True)
        if cands:
            deploy_dir = pathlib.Path(cands[0])

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

data = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "probe_epoch": time.time(),
    "deployment_dir": str(deploy_dir) if deploy_dir else None,
    "frozen_author_eval": {
        **read_frozen_author_evaluation(deploy_dir),
        **read_frozen_author_evaluation(deploy_dir, "1p5b"),
    },
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
    "cells": {},
    "discovered_cells": [],
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
    webshop_original = deploy_dir / "output-webshop-menu-fixed-20261119"
    webshop_paired = deploy_dir / "output-webshop150-paired-20261119"
    lane_runners = {p["lane"]: p for p in procs if "runner" in (p.get("role") or "")}
    cell_roots = {}
    scheduled_dirs = set()
    for spec in CELL_SPECS:
        ck = f"{spec['suite']}/{spec['benchmark']}/{spec['method']}/seed-{spec['seed']}"
        study_root = deploy_dir / spec["output_subdir"] if spec.get("output_subdir") else out_dir
        stage_rounds = None
        if study_root == webshop_paired and ck.startswith("comparison/webshop/fgpo/"):
            runner = lane_runners.get(f"GPU {spec['physical_gpu_ids']}", {})
            runner_root = None
            if (runner.get("target_suite") == spec["suite"]
                    and runner.get("target_benchmark") == spec["benchmark"]
                    and runner.get("target_method") == spec["method"]
                    and spec["seed"] in (runner.get("target_seeds") or [])
                    and runner.get("target_output_dir")):
                runner_root = pathlib.Path(runner["target_output_dir"])
            if runner_root == webshop_original or (
                    runner_root is None and not (webshop_paired / ck).is_dir()
                    and (webshop_original / ck).is_dir()):
                study_root = webshop_original
                stage_rounds = 60
        cell_roots[spec["cell_key"]] = study_root
        target_cell_dir = study_root / ck
        scheduled_dirs.add(str(target_cell_dir.resolve()))
        scheduled_dirs.add(os.path.abspath(str(target_cell_dir)))
        cd = read_cell_dir(target_cell_dir, spec["official_eval"].split("+"), include_du=False)
        if stage_rounds is not None:
            cd["stage_rounds"] = stage_rounds
            cd["output_subdir"] = study_root.name
        if ck.startswith("comparison/webshop/fgpo/") and study_root.name in (
                webshop_original.name, webshop_paired.name,
                "output-webshop150-v4-joint-20261120", "output-webshop150-v4-radius-20261120"):
            original_cell = webshop_original / ck
            scheduled_dirs.add(str(original_cell.resolve()))
            scheduled_dirs.add(os.path.abspath(str(original_cell)))
            if (original_cell / ".finalization_complete.json").is_file():
                try:
                    endpoint = json.loads((original_cell / "test/evaluation_summary.json").read_text())
                    cd["webshop_v4_60"] = {
                        key: endpoint[key] for key in ("SR", "Score", "success_count", "evaluated_count")
                    }
                except (OSError, ValueError, KeyError):
                    pass
        data["cells"][spec["cell_key"]] = cd

    disc_list, disc_map = discover_unscheduled_cells(
        [str(deploy_dir / "output*/*/*/*/seed-*")],
        procs,
        scheduled_dirs,
        set(),
        set(data["cells"].keys()),
        "0",
    )
    data["discovered_cells"] = disc_list
    data["cells"].update(disc_map)

    launch_dir = out_dir / "launch"
    all_specs = list(CELL_SPECS) + list(disc_list)
    lane_sups = {}
    for p in procs:
        if p.get("role") in ("queue_controller", "supervisor") and p.get("lane"):
            lane_sups.setdefault(p["lane"], p)
    for lane_tag, prefix, gpu_id in [("Host B · GPU 0", "gpu0", "0"), ("Host B · GPU 2", "gpu2", "2")]:
        matches = sorted(launch_dir.glob(f"{prefix}*.log"), key=lambda p: p.stat().st_mtime, reverse=True) if launch_dir.is_dir() else []
        lane_specs = [spec for spec in all_specs if spec.get("physical_gpu_ids") == gpu_id]
        current_logs = [deploy_dir / spec["output_subdir"] / "launch" / name
                        for spec in lane_specs if spec.get("output_subdir")
                        for name in (f"seed-{spec['seed']}.log",
                                     f"queue-gpu-{gpu_id}.log",
                                     f"{spec['benchmark']}-{spec['method']}-seed-{spec['seed']}.log",
                                     f"{spec['benchmark']}-seed-{spec['seed']}.log" if spec["method"] == "fgpo"
                                     else f"{spec['benchmark']}-gpu-{gpu_id}.queue.log")]
        existing_cur_logs = sorted((p for p in current_logs if p.is_file() and p.stat().st_size > 0), key=lambda p: p.stat().st_mtime, reverse=True)
        if existing_cur_logs:
            matches = existing_cur_logs
        lpath = matches[0] if matches else None
        extra_lines = []
        active_ck = None
        lane_runner = lane_runners.get("GPU " + gpu_id)
        lane_sup = lane_sups.get("GPU " + gpu_id)
        if lane_runner and lane_runner.get("target_seeds"):
            ts = lane_runner["target_seeds"][0]
            suite = lane_runner.get("target_suite") or "comparison"
            bench = lane_runner.get("target_benchmark")
            method = lane_runner.get("target_method")
            active_ck = next((s["cell_key"] for s in lane_specs if matches_scheduled_probe(s, lane_runner)),
                             f"{suite}/{bench}/{method}/seed-{ts}")
            fd_cands = read_proc_log_fds(lane_runner["pid"]) if lane_runner.get("pid") else []
            if fd_cands:
                lpath = pathlib.Path(fd_cands[0])
            elif lane_runner.get("target_output_dir"):
                rdir = pathlib.Path(lane_runner["target_output_dir"]) / "launch"
                for cand_nm in (f"seed-{ts}.log", f"{bench}-{method}-seed-{ts}.log", f"{bench}-seed-{ts}.log"):
                    cand_p = rdir / cand_nm
                    if cand_p.is_file():
                        lpath = cand_p
                        break
            elif active_ck in cell_roots:
                for cand_nm in (f"seed-{ts}.log", f"{bench}-{method}-seed-{ts}.log", f"{bench}-seed-{ts}.log"):
                    active_log = cell_roots[active_ck] / "launch" / cand_nm
                    if active_log.is_file():
                        lpath = active_log
                        break
        elif lane_sup and lane_sup.get("pid"):
            sup_fds = read_proc_log_fds(lane_sup["pid"])
            if sup_fds:
                lpath = pathlib.Path(sup_fds[0])
        ltail = sh(f"tail -n 25 '{lpath}' 2>/dev/null || true") if lpath else ""
        for spec in lane_specs:
            ck = spec["cell_key"]
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

def _file_sig(path: Path | None) -> tuple[int, int, int] | None:
    if path is None:
        return None
    try:
        st = path.stat()
        return (int(st.st_ino), int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


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
            if len(user) >= 8:
                pairs.append((user[:7] + "+", "<user>"))
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
    phys = str(physical_gpu_ids or "").strip()
    if hc in ("colab", "pro6000"):
        return "colab", "colab_gpu_0", "Colab · GPU 0"
    if hc in ("host a", "host_a", "host_a_l40"):
        if phys == "3+4":
            return "host_a", "host_a_l40_3_4", "Host A · L40 3+4 (Queue A1)"
        if phys == "1+7":
            return "host_a", "host_a_l40_1_7", "Host A · L40 1+7 (Queue A2)"
        if phys == "7":
            return "host_a", "host_a_gpu_7", "Host A · GPU 7 (单卡)"
        if phys == "3":
            return "host_a", "host_a_gpu_3", "Host A · GPU 3 (单卡)"
        slug = re.sub(r"[^0-9A-Za-z]+", "_", phys).strip("_") or "0"
        return "host_a", f"host_a_gpu_{slug}", f"Host A · GPU {phys}"
    if phys == "0":
        return "host_b", "host_b_gpu_0", "Host B · GPU 0 (Queue B0)"
    if phys == "2":
        return "host_b", "host_b_gpu_2", "Host B · GPU 2 (Queue B2)"
    slug = re.sub(r"[^0-9A-Za-z]+", "_", phys).strip("_") or "0"
    return "host_b", f"host_b_gpu_{slug}", f"Host B · GPU {phys}"


def _classify_error(err_clean: str) -> str:
    low = (err_clean or "").lower()
    if not low:
        return ""
    if "timed out" in low or "timeout" in low:
        return "SSH 连接或探针超时"
    if "password" in low or "authentication" in low or "permission denied" in low or "ssh_document" in low:
        return "SSH 凭据认证失败"
    if "connection refused" in low or "unreachable" in low or "could not resolve" in low or "ssh exit 255" in low:
        return "SSH 网络不可达"
    if "traceback" in low or "keyerror" in low or "missing json marker" in low or "ssh exit 1" in low:
        return "远端探针脚本异常"
    if "no active sessions" in low:
        return "Colab 会话未启动"
    return "探测异常"


def _extract_local_cell_config(cfg_f: Path, bm_hint: str = "") -> dict[str, Any] | None:
    if not cfg_f.is_file():
        return None
    try:
        cfg_obj = json.loads(cfg_f.read_text(encoding="utf-8"))
        if not isinstance(cfg_obj, dict):
            return None
        mp = str(cfg_obj.get("model_path") or "")
        mp_low = mp.lower()
        backbone = "1.5B" if "1.5b" in mp_low else ("7B" if "7b" in mp_low else None)
        bms = cfg_obj.get("benchmarks")
        bm_cfg = bm_hint or (bms[0] if isinstance(bms, list) and bms else "")
        out_tok_map = cfg_obj.get("output_tokens_by_benchmark") or {}
        out_tok = out_tok_map.get(bm_cfg) if isinstance(out_tok_map, dict) else None
        if out_tok is None:
            out_tok = cfg_obj.get("output_tokens")
        hor_map = cfg_obj.get("horizons") or {}
        horizon = hor_map.get(bm_cfg) if isinstance(hor_map, dict) else cfg_obj.get("horizon")
        if horizon is None:
            horizon = cfg_obj.get("horizon")
        return {
            "model_name": Path(mp).name if mp else None,
            "backbone": backbone,
            "learning_rate": cfg_obj.get("learning_rate"),
            "credit_radius": cfg_obj.get("credit_radius"),
            "credit_prior_weight": cfg_obj.get("credit_prior_weight"),
            "credit_transfer_max": cfg_obj.get("credit_transfer_max"),
            "entropy_coef": cfg_obj.get("entropy_coef"),
            "kl_coef": cfg_obj.get("kl_coef"),
            "rounds": cfg_obj.get("rounds"),
            "horizon": horizon,
            "output_tokens": out_tok,
            "study_protocol": cfg_obj.get("study_protocol"),
        }
    except Exception:
        return None


def _clean_local_eval_summary(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if k in ("task_types", "recipe_depths") and isinstance(v, dict):
            sub_out: dict[str, Any] = {}
            for sk, sv in v.items():
                if isinstance(sv, dict):
                    sub_out[sk] = {
                        kk: vv for kk, vv in sv.items()
                        if kk not in ("missing_tasks", "unexpected_tasks")
                    }
                else:
                    sub_out[sk] = sv
            out[k] = sub_out
        elif k in ("missing_tasks", "unexpected_tasks"):
            continue
        else:
            out[k] = v
    return out


def _slim_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Produce a lightweight copy of snap without bulky per-cell event tails for /api/state?slim=1."""
    slim = dict(snap)
    servers_in = snap.get("servers") or {}
    servers_out: dict[str, Any] = {}
    for sid, srv in servers_in.items():
        if not isinstance(srv, dict):
            servers_out[sid] = srv
            continue
        s_cp = dict(srv)
        tel = srv.get("telemetry")
        if isinstance(tel, dict) and isinstance(tel.get("cells"), dict):
            tel_cp = dict(tel)
            cells_cp: dict[str, Any] = {}
            for ck, cell_obj in tel["cells"].items():
                if not isinstance(cell_obj, dict):
                    cells_cp[ck] = cell_obj
                    continue
                c_cp = dict(cell_obj)
                c_cp.pop("recent_cost_events", None)
                c_cp.pop("du", None)
                evals = cell_obj.get("evaluations")
                if isinstance(evals, dict):
                    evals_cp: dict[str, Any] = {}
                    for sp, ev_v in evals.items():
                        if isinstance(ev_v, dict) and "recent" in ev_v:
                            ev_sub = dict(ev_v)
                            ev_sub.pop("recent", None)
                            evals_cp[sp] = ev_sub
                        else:
                            evals_cp[sp] = ev_v
                    c_cp["evaluations"] = evals_cp
                cells_cp[ck] = c_cp
            tel_cp["cells"] = cells_cp
            s_cp["telemetry"] = tel_cp
        servers_out[sid] = s_cp
    slim["servers"] = servers_out
    return slim


def _merge_frozen_author_rows(previous, observed, snapshot):
    rows = {}
    for batch in (previous, observed):
        for row in batch:
            if (snapshot is not None and row.get("model_snapshot") == snapshot
                    and row.get("interface_protocol") == FROZEN_AUTHOR_INTERFACE):
                rows.setdefault((row.get("panel", 0), row["ordinal"]), row)
    return [rows[key] for key in sorted(rows)]


def _paired_frozen_webshop(original, adapted):
    original_by_ordinal = {row["ordinal"]: row for row in original}
    pairs = []
    for row in adapted:
        ref = original_by_ordinal.get(row["ordinal"])
        if ref and all(row[key] is not None and row[key] == ref[key]
                       for key in ("panel", "task_id", "worker_seed", "session_index")):
            pairs.append((ref, row))
    if not pairs:
        return {"matched": 0, "gained": None, "lost": None, "delta_sr_pp": None, "delta_score_points": None}
    gained = sum(not before["success"] and after["success"] for before, after in pairs)
    lost = sum(before["success"] and not after["success"] for before, after in pairs)
    return {
        "matched": len(pairs),
        "gained": gained,
        "lost": lost,
        "delta_sr_pp": 100 * (gained - lost) / len(pairs),
        "delta_score_points": 100 * sum(after["score"] - before["score"] for before, after in pairs) / len(pairs),
    }


def _default_endpoint_tasks(benchmark: str) -> int | None:
    if benchmark == "alfworld":
        return 134
    if benchmark == "webshop":
        return 128
    if benchmark == "textcraft":
        return 79
    return None


def _load_schedule_18(csv_path: Path) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    if csv_path.is_file():
        try:
            with csv_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    bm = row["benchmark"].strip()
                    if bm not in ("alfworld", "textcraft", "webshop"):
                        continue
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
                        "total_training_trajectories": (
                            int(row["training_trajectories"]) if row.get("training_trajectories")
                            else None if row["method"].strip() == "fgpo_tree_mc"
                            else int(row.get("total_training_trajectories") or (tot_r * 128))
                        ),
                        "horizon": int(row["horizon"]) if row.get("horizon") not in (None, "") else (30 if bm == "textcraft" else (50 if bm == "alfworld" else (15 if bm == "webshop" else 100))),
                        "official_eval": (row.get("official_eval") or {"textcraft": "val", "alfworld": "valid_unseen", "webshop": "graphgpo_released_validation"}[bm]).strip(),
                        "official_endpoint_tasks": int(row["official_endpoint_tasks"]) if row.get("official_endpoint_tasks") not in (None, "") else _default_endpoint_tasks(bm),
                        "status_snapshot": (row.get("status_snapshot") or "").strip(),
                        "sampling_layout": (row.get("sampling_layout") or "").strip(),
                        "dependency": (row.get("dependency") or "").strip(),
                        "output_subdir": (row.get("output_subdir") or "").strip(),
                        "variant": (row.get("variant") or "").strip(),
                        "auto_discovered": False,
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
        if r["variant"]:
            ck += "@" + r["variant"]
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


def _compute_round_velocity_sec(wall_times: list[float]) -> float | None:
    if not wall_times:
        return None
    if len(wall_times) == 1:
        return round(float(wall_times[0]), 2) if wall_times[0] > 0 else None
    deltas = []
    tail = wall_times[-7:]
    for i in range(1, len(tail)):
        dt = float(tail[i]) - float(tail[i - 1])
        if dt > 1.0:
            deltas.append(dt)
    if deltas:
        return round(sum(deltas[-5:]) / len(deltas[-5:]), 2)
    return round(float(wall_times[-1]) / len(wall_times), 2) if wall_times[-1] > 0 else None


# ---------------------------------------------------------------------------
# Unified Multi-Server ClusterPoller
# ---------------------------------------------------------------------------

class ClusterPoller:
    """Thread-safe async poller with SSE state notifications, local hot-reload, and <1ms snapshot cache."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = Path(__file__).resolve().parents[1]
        self.colab_bin = os.path.expanduser(args.colab_bin)
        self.env = _build_env()
        self.ssh_doc_path = Path(os.path.expanduser(args.ssh_doc)).resolve()
        self.schedule_csv_path = Path(os.path.expanduser(args.schedule_csv)).resolve()
        self._meta_lock = threading.Lock()

        self._schedule_sig = _file_sig(self.schedule_csv_path)
        self.schedule_18 = _load_schedule_18(self.schedule_csv_path)
        self.schedule_mtime = self._format_file_mtime(self.schedule_csv_path)

        sess_str = (getattr(args, "session", "") or "").strip()
        self.colab_archive_only = (not sess_str) and (not getattr(args, "disable_colab", False))
        self.r4_receipt_dir = (
            self.repo_root / "outputs" / "fgpo-r4-colab-20261119"
            if (args.method == "fgpo_tree_mc" and bool(sess_str)) else None
        )

        self._ssh_doc_sig = _file_sig(self.ssh_doc_path)
        self.ssh_creds = _parse_ssh_doc(self.ssh_doc_path)
        self.secret_replacements = _build_secret_replacements(self.ssh_creds)
        self._batch_mode_cache: dict[str, bool] = {}

        # Local disk caches keyed by file signatures
        self._valid_seen_sig: tuple[Any, ...] | None = None
        self._valid_seen_cache: dict[int, dict[str, Any]] = {}
        self._archived_colab_sig: tuple[Any, ...] | None = None
        self._archived_colab_cache: dict[str, dict[str, Any]] = {}
        self._colab_mirror_sig: tuple[Any, ...] | None = None
        self._colab_mirror_cache: dict[str, dict[str, Any]] = {}
        self._lit_sig: tuple[Any, ...] | None = None
        self._lit_cache: list[dict[str, Any]] = []
        self._paired_sig: tuple[Any, ...] | None = None
        self._paired_cache: dict[str, Any] = {}
        self._r4_arch_sig: tuple[Any, ...] | None = None
        self._r4_arch_cache: dict[str, Any] | None = None
        self._r4_sig: tuple[Any, ...] = self._compute_r4_sig()
        self._r4_raw_cache: tuple[dict[str, Any], list[dict[str, Any]], str | None] | None = None
        self._last_watched_ui_mtime: int = _get_ui_mtime()

        self.enabled: dict[str, bool] = {
            "colab": not getattr(args, "disable_colab", False),
            "host_a": not getattr(args, "disable_host_a", False),
            "host_b": not getattr(args, "disable_host_b", False),
        }

        colab_title = (
            "Colab · RTX PRO 6000 (已归档)"
            if self.colab_archive_only
            else f"Colab (Session {sess_str})"
        )
        self.server_titles: dict[str, str] = {
            "colab": colab_title,
            "host_a": "Host A · mgt01 (6× NVIDIA L40)",
            "host_b": "Host B · Dual RTX 6000D (GPUs 0 & 2)",
        }

        self.state_locks: dict[str, threading.Lock] = {k: threading.Lock() for k in ("colab", "host_a", "host_b")}
        self.poll_locks: dict[str, threading.Lock] = {k: threading.Lock() for k in ("colab", "host_a", "host_b")}
        self._wake_events: dict[str, threading.Event] = {k: threading.Event() for k in ("colab", "host_a", "host_b")}
        self.histories: dict[str, deque[dict[str, Any]]] = {k: deque(maxlen=120) for k in ("colab", "host_a", "host_b")}
        self.last_colab_meta_ts = 0.0

        self.server_states: dict[str, dict[str, Any]] = {}
        for sid in ("colab", "host_a", "host_b"):
            is_arch_colab = (sid == "colab" and self.colab_archive_only)
            init_status = "disabled" if not self.enabled[sid] else ("archived" if is_arch_colab else "initializing")
            self.server_states[sid] = {
                "server_id": sid,
                "display_name": self.server_titles[sid],
                "enabled": self.enabled[sid],
                "archived_only": is_arch_colab,
                "status": init_status,
                "polling": False,
                "last_ok_at": "本地归档" if is_arch_colab else None,
                "last_attempt_at": None,
                "latency_ms": None,
                "last_error": None,
                "error_summary": None,
                "transport": "local_archive" if is_arch_colab else None,
                "colab": {} if sid == "colab" else None,
                "telemetry": {"cells": {}} if is_arch_colab else None,
                "history": [],
            }

        self.meta = {
            "session": sess_str,
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
            "colab_archive_only": self.colab_archive_only,
        }

        # SSE condition & cached snapshot state
        self.state_cond = threading.Condition()
        self.state_seq: int = 0
        self.last_changed_server: str = "init"
        self.latest_event: dict[str, Any] | None = None
        self.events_feed: deque[dict[str, Any]] = deque(maxlen=40)
        self._prev_cell_states: dict[str, dict[str, Any]] = {}
        self._prev_server_statuses: dict[str, str] = {}
        self._seen_discovered_ids: set[str] = set()
        self._cached_snapshot: dict[str, Any] | None = None
        self._cached_slim_snapshot: dict[str, Any] | None = None

        self._rebuild_and_publish("init", is_init=True)

    @staticmethod
    def _format_file_mtime(path: Path) -> str:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            return time.strftime("%Y-%m-%d %H:%M:%S")

    def stop(self) -> None:
        for ev in self._wake_events.values():
            ev.set()
        with self.state_cond:
            self.state_cond.notify_all()

    def _reload_ssh_creds_if_needed(self) -> bool:
        sig = _file_sig(self.ssh_doc_path)
        with self._meta_lock:
            if sig == self._ssh_doc_sig:
                return False
            self._ssh_doc_sig = sig
            fresh = _parse_ssh_doc(self.ssh_doc_path)
            if fresh:
                self.ssh_creds = fresh
                self.secret_replacements = _build_secret_replacements(fresh)
                self._valid_seen_sig = None
                self._archived_colab_sig = None
                self._colab_mirror_sig = None
                return True
        return False

    def _reload_schedule_if_needed(self) -> tuple[bool, int]:
        sig = _file_sig(self.schedule_csv_path)
        with self._meta_lock:
            if sig == self._schedule_sig:
                return False, len(self.schedule_18)
            try:
                fresh_sched = _load_schedule_18(self.schedule_csv_path)
            except Exception:
                return False, len(self.schedule_18)
            self._schedule_sig = sig
            self.schedule_18 = fresh_sched
            self.schedule_mtime = self._format_file_mtime(self.schedule_csv_path)
            self._archived_colab_sig = None
            self._colab_mirror_sig = None
            return True, len(fresh_sched)

    def _compute_valid_seen_sig(self) -> tuple[Any, ...]:
        vs_root = self.repo_root / "outputs" / "exp-plan-20261118" / "alfworld-valid-seen"
        if not vs_root.is_dir():
            return ()
        return tuple((str(sp), _file_sig(sp)) for sp in sorted(vs_root.glob("seed-*/evaluation_summary.json")))

    def _load_local_valid_seen(self) -> dict[int, dict[str, Any]]:
        """Read and cache local ALFWorld valid_seen replay summaries by (ino, mtime_ns, size)."""
        cur_sig = self._compute_valid_seen_sig()
        with self._meta_lock:
            if cur_sig == self._valid_seen_sig:
                return dict(self._valid_seen_cache)
        out: dict[int, dict[str, Any]] = {}
        vs_root = self.repo_root / "outputs" / "exp-plan-20261118" / "alfworld-valid-seen"
        if vs_root.is_dir():
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
        with self._meta_lock:
            self._valid_seen_sig = cur_sig
            self._valid_seen_cache = out
            return dict(out)

    def _compute_archived_colab_sig(self, schedule_rows: list[dict[str, Any]]) -> tuple[Any, ...]:
        archive = self.repo_root / "outputs" / "fgpo-colab-956ceab-ag-20261116" / "gdrive_staging"
        if not archive.is_dir():
            return ()
        items = []
        for scheduled in schedule_rows:
            if scheduled["server_id"] != "colab":
                continue
            key = scheduled["cell_key"]
            d = archive / key
            items.append((
                key,
                _file_sig(d / "cell_summary.json"),
                _file_sig(d / "reporting_summary.json"),
                _file_sig(d / "config.json"),
                _file_sig(d / "training.jsonl"),
            ))
        return tuple(items)

    def _load_archived_colab_cells(self, schedule_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        cur_sig = self._compute_archived_colab_sig(schedule_rows)
        with self._meta_lock:
            if cur_sig == self._archived_colab_sig:
                return copy.deepcopy(self._archived_colab_cache)

        archive = self.repo_root / "outputs" / "fgpo-colab-956ceab-ag-20261116" / "gdrive_staging"
        cells: dict[str, dict[str, Any]] = {}
        for scheduled in schedule_rows:
            if scheduled["server_id"] != "colab":
                continue
            key = scheduled["cell_key"]
            directory = archive / key
            cs_path = directory / "cell_summary.json"
            rs_path = directory / "reporting_summary.json"
            try:
                summary = json.loads(cs_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if summary.get("status") != "complete" or summary.get("finalization_stage") != "finalized":
                continue
            if isinstance(summary.get("evaluation"), dict):
                summary["evaluation"] = {
                    ek: _clean_local_eval_summary(ev) for ek, ev in summary["evaluation"].items()
                }
            if rs_path.is_file():
                try:
                    rs_obj = json.loads(rs_path.read_text(encoding="utf-8"))
                    if isinstance(rs_obj, dict):
                        if isinstance(rs_obj.get("evaluation"), dict):
                            ev_tgt = summary.setdefault("evaluation", {})
                            for ek, ev in rs_obj["evaluation"].items():
                                ce = _clean_local_eval_summary(ev)
                                if ek not in ev_tgt or not isinstance(ev_tgt[ek], dict):
                                    ev_tgt[ek] = ce
                                elif isinstance(ce, dict):
                                    for bk in ("task_types", "recipe_depths", "SR", "Score", "score"):
                                        if bk not in ev_tgt[ek] and bk in ce:
                                            ev_tgt[ek][bk] = ce[bk]
                        if isinstance(rs_obj.get("costs"), dict) and "costs" not in summary:
                            summary["costs"] = rs_obj["costs"]
                except (OSError, ValueError):
                    pass

            cell_cfg = _extract_local_cell_config(directory / "config.json", scheduled["benchmark"])
            tr_path = directory / "training.jsonl"
            try:
                raw_rounds = [json.loads(line) for line in tr_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, ValueError):
                raw_rounds = []
            wall_times = [
                float((r.get("costs") or {}).get("wall_seconds"))
                for r in raw_rounds if (r.get("costs") or {}).get("wall_seconds") is not None
            ]
            last_costs = (raw_rounds[-1].get("costs") or {}) if raw_rounds else {}
            last_opt = raw_rounds[-1].get("optimizer_steps") if raw_rounds else 0
            mtimes = [p.stat().st_mtime for p in (cs_path, rs_path, tr_path) if p.is_file()]
            alloc_dev_sec = (
                ((summary.get("costs") or {}).get("allocated_device_seconds"))
                or (((summary.get("training") or {}).get("costs") or {}).get("allocated_device_seconds"))
                or last_costs.get("allocated_device_seconds")
            )
            entry: dict[str, Any] = {
                "exists": True,
                "archived_colab": True,
                "probe_epoch": time.time(),
                "last_mtime": max(mtimes) if mtimes else None,
                "recent_sec_per_round": _compute_round_velocity_sec(wall_times),
                "cell_summary": summary,
                "training_rounds": [project_training_round(row, scheduled["benchmark"]) for row in raw_rounds],
                "costs_summary": {
                    "total_events": last_costs.get("events"),
                    "session_elapsed_seconds": last_costs.get("wall_seconds"),
                    "env_steps": last_costs.get("env_steps"),
                    "input_tokens": last_costs.get("input_tokens"),
                    "output_tokens": last_costs.get("output_tokens"),
                    "forward_tokens": last_costs.get("forward_tokens"),
                    "successful_adamw_count": last_opt,
                    "last_phase_role": "archived_complete",
                },
                "evaluations": {
                    split: {"summary": result, "completed_count": result.get("evaluated_count"),
                            "success_count": result.get("success_count")}
                    for split, result in (summary.get("evaluation") or {}).items()
                },
            }
            if cell_cfg:
                entry["cell_config"] = cell_cfg
            if alloc_dev_sec is not None:
                entry["allocated_device_seconds"] = round(float(alloc_dev_sec), 2)
            cells[key] = _sanitize_secrets(entry, self.secret_replacements)

        with self._meta_lock:
            self._archived_colab_sig = cur_sig
            self._archived_colab_cache = cells
            return copy.deepcopy(cells)

    @staticmethod
    def _mirror_dir_for_colab_row(repo_root: Path, scheduled: dict[str, Any]) -> Path | None:
        if scheduled.get("server_id") != "colab" or scheduled.get("benchmark") != "alfworld":
            return None
        var_low = str(scheduled.get("variant") or "").lower()
        sub_low = str(scheduled.get("output_subdir") or "").lower()
        if "1p5b" not in var_low and "1p5b" not in sub_low:
            return None
        seed_i = scheduled.get("seed")
        if seed_i is None:
            return None
        d = repo_root / "archive" / "fgpo-1p5b-alfworld-colab-20261121" / "off-vm" / f"seed-{seed_i}"
        return d if d.is_dir() else None

    def _compute_local_colab_mirror_sig(self, schedule_rows: list[dict[str, Any]]) -> tuple[Any, ...]:
        items = []
        for scheduled in schedule_rows:
            d = self._mirror_dir_for_colab_row(self.repo_root, scheduled)
            if d is None:
                continue
            key = scheduled["cell_key"]
            items.append((
                key,
                _file_sig(d / "training.jsonl"),
                _file_sig(d / "training_summary.json"),
                _file_sig(d / "cell_summary.json"),
                _file_sig(d / "config.json"),
                _file_sig(d / "checkpoint" / "CURRENT"),
                _file_sig(d / "valid_unseen" / "evaluation_summary.json"),
                _file_sig(d / "costs.jsonl"),
            ))
        return tuple(items)

    def _load_local_colab_mirrors(self, schedule_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        cur_sig = self._compute_local_colab_mirror_sig(schedule_rows)
        with self._meta_lock:
            if cur_sig == self._colab_mirror_sig:
                return copy.deepcopy(self._colab_mirror_cache)

        cells: dict[str, dict[str, Any]] = {}
        for scheduled in schedule_rows:
            directory = self._mirror_dir_for_colab_row(self.repo_root, scheduled)
            if directory is None:
                continue
            key = scheduled["cell_key"]
            tr_path = directory / "training.jsonl"
            ts_path = directory / "training_summary.json"
            cs_path = directory / "cell_summary.json"
            cj_path = directory / "costs.jsonl"
            ckpt_path = directory / "checkpoint" / "CURRENT"
            ev_path = directory / "valid_unseen" / "evaluation_summary.json"

            try:
                raw_rounds = [json.loads(line) for line in tr_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            except (OSError, ValueError):
                raw_rounds = []

            ts_obj: dict[str, Any] | None = None
            if ts_path.is_file():
                try:
                    loaded = json.loads(ts_path.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(loaded, dict):
                        ts_obj = {k: v for k, v in loaded.items() if k not in ("rounds", "history")}
                except (OSError, ValueError):
                    ts_obj = None

            cs_obj: dict[str, Any] | None = None
            if cs_path.is_file():
                try:
                    loaded_cs = json.loads(cs_path.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(loaded_cs, dict):
                        cs_obj = {k: v for k, v in loaded_cs.items() if k not in ("rounds", "history")}
                        if isinstance(cs_obj.get("evaluation"), dict):
                            cs_obj["evaluation"] = {
                                ek: _clean_local_eval_summary(ev) for ek, ev in cs_obj["evaluation"].items()
                            }
                except (OSError, ValueError):
                    cs_obj = None

            if not raw_rounds and ts_obj is None and cs_obj is None and not cj_path.is_file():
                continue

            projected_rounds = [project_training_round(row, scheduled["benchmark"]) for row in raw_rounds]
            wall_times = [
                float((r.get("costs") or {}).get("wall_seconds"))
                for r in raw_rounds if (r.get("costs") or {}).get("wall_seconds") is not None
            ]
            last_costs = (
                (raw_rounds[-1].get("costs") or {}) if raw_rounds
                else (((ts_obj or {}).get("costs") or {}) if isinstance((ts_obj or {}).get("costs"), dict) else {})
            )
            last_opt = max(
                int((raw_rounds[-1].get("optimizer_steps") or 0)) if raw_rounds else 0,
                int((ts_obj or {}).get("optimizer_steps") or 0),
            )
            mtimes = [p.stat().st_mtime for p in (cj_path, tr_path, ts_path, cs_path, ckpt_path, ev_path) if p.is_file()]
            alloc_dev_sec = (
                ((cs_obj or {}).get("costs") or {}).get("allocated_device_seconds")
                or ((ts_obj or {}).get("costs") or {}).get("allocated_device_seconds")
                or last_costs.get("allocated_device_seconds")
            )
            evals_map: dict[str, Any] = {}
            if isinstance((cs_obj or {}).get("evaluation"), dict):
                for sp, ev in cs_obj["evaluation"].items():
                    if isinstance(ev, dict):
                        evals_map[sp] = {
                            "summary": ev,
                            "completed_count": ev.get("evaluated_count"),
                            "success_count": ev.get("success_count"),
                        }
            elif ev_path.is_file():
                try:
                    ev_raw = _clean_local_eval_summary(json.loads(ev_path.read_text(encoding="utf-8", errors="replace")))
                    if isinstance(ev_raw, dict):
                        all_b = (ev_raw.get("by_task_type") or {}).get("All") or {}
                        evals_map["valid_unseen"] = {
                            "summary": ev_raw,
                            "completed_count": ev_raw.get("evaluated_count") or all_b.get("episode_count"),
                            "success_count": ev_raw.get("success_count") or all_b.get("success_count"),
                        }
                except (OSError, ValueError):
                    pass

            entry: dict[str, Any] = {
                "exists": True,
                "local_mirror": True,
                "probe_epoch": time.time(),
                "last_mtime": max(mtimes) if mtimes else None,
                "recent_sec_per_round": _compute_round_velocity_sec(wall_times),
                "training_rounds": projected_rounds,
                "costs_summary": {
                    "total_events": last_costs.get("events"),
                    "session_elapsed_seconds": last_costs.get("wall_seconds"),
                    "env_steps": last_costs.get("env_steps"),
                    "input_tokens": last_costs.get("input_tokens"),
                    "output_tokens": last_costs.get("output_tokens"),
                    "forward_tokens": last_costs.get("forward_tokens"),
                    "successful_adamw_count": last_opt,
                    "last_phase_role": (ts_obj or {}).get("phase") or "local_mirror",
                },
                "evaluations": evals_map,
            }
            if (scheduled.get("output_subdir") or "").strip():
                entry["output_subdir"] = scheduled["output_subdir"]
            cell_cfg = _extract_local_cell_config(directory / "config.json", scheduled["benchmark"])
            if cell_cfg:
                entry["cell_config"] = cell_cfg
            if ts_obj is not None:
                entry["training_summary"] = ts_obj
            if cs_obj is not None:
                entry["cell_summary"] = cs_obj
            if ckpt_path.is_file():
                try:
                    entry["checkpoint_current"] = json.loads(ckpt_path.read_text(encoding="utf-8", errors="replace"))
                except (OSError, ValueError):
                    pass
            if alloc_dev_sec is not None:
                entry["allocated_device_seconds"] = round(float(alloc_dev_sec), 2)
            cells[key] = _sanitize_secrets(entry, self.secret_replacements)

        with self._meta_lock:
            self._colab_mirror_sig = cur_sig
            self._colab_mirror_cache = cells
            return copy.deepcopy(cells)

    def _compute_lit_sig(self) -> tuple[Any, ...]:
        csv_p = self.repo_root / "outputs" / "exp-plan-20261118" / "graphgpo_table1_reported.csv"
        return (_file_sig(csv_p),)

    def _load_literature_baselines(self) -> list[dict[str, Any]]:
        cur_sig = self._compute_lit_sig()
        with self._meta_lock:
            if cur_sig == self._lit_sig:
                return copy.deepcopy(self._lit_cache)

        items: list[dict[str, Any]] = []
        csv_p = self.repo_root / "outputs" / "exp-plan-20261118" / "graphgpo_table1_reported.csv"
        if csv_p.is_file():
            try:
                with csv_p.open("r", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        bb_raw = (row.get("backbone") or "").strip()
                        bb_low = bb_raw.lower()
                        if "1.5b" in bb_low:
                            bb = "1.5B"
                        elif "7b" in bb_low:
                            bb = "7B"
                        elif "closed" in bb_low:
                            bb = "Closed"
                        else:
                            continue
                        meth = (row.get("method") or "").strip()
                        mtype = (row.get("method_type") or "").strip()
                        if not meth:
                            continue
                        src_table = f"Table {(row.get('paper_table') or '1').strip()}"
                        if bb == "Closed":
                            alf_note = "闭源模型零样本 Prompting 参考（GraphGPO 表 1 / BiPACE 表 2 引用自 Feng et al. 2025）"
                            ws_note = "闭源模型 Prompting 参考（GraphGPO 表 1 / BiPACE 表 2 引用自 Feng et al. 2025）"
                        elif mtype != "RL Training":
                            alf_note = "Prompting 参考（GraphGPO 表 1 / BiPACE 表 2 引用自 Feng et al. 2025）"
                            ws_note = "Prompting 参考（GraphGPO 表 1 / BiPACE 表 2 引用自 Feng et al. 2025）"
                        elif meth in ("PPO", "RLOO"):
                            alf_note = "150 轮 · 引用自 GiGPO 论文（原环境，与本地 60 轮协议不同）"
                            ws_note = "150 轮 · 引用自 GiGPO 论文（原环境 15 步）"
                        else:
                            alf_note = "150 轮 · GraphGPO 增强观测环境复现（追加位置/手持状态，与原环境及本地 60 轮协议不同）"
                            ws_note = "150 轮 · GraphGPO text-rich 环境复现（max_prompt=5120，与原 4096 环境不同）"
                        # ALFWorld row
                        alf_all = row.get("alf_all")
                        if alf_all not in (None, ""):
                            tt = {}
                            for col_k, cat_name in (
                                ("alf_pick", "Pick"), ("alf_look", "Look"), ("alf_clean", "Clean"),
                                ("alf_heat", "Heat"), ("alf_cool", "Cool"), ("alf_pick2", "Pick2"), ("alf_all", "All"),
                            ):
                                if row.get(col_k) not in (None, ""):
                                    tt[cat_name] = {"sr": float(row[col_k])}
                            items.append({
                                "benchmark": "alfworld",
                                "backbone": bb,
                                "method": meth,
                                "method_type": mtype,
                                "sr": float(alf_all),
                                "sd": float(row["alf_all_reported_dispersion"]) if row.get("alf_all_reported_dispersion") not in (None, "") else None,
                                "score": None,
                                "score_sd": None,
                                "task_types": tt,
                                "source": f"GraphGPO 论文 {src_table}",
                                "protocol_note": alf_note,
                            })
                        # WebShop row
                        ws_sr = row.get("webshop_success")
                        ws_sc = row.get("webshop_score")
                        if ws_sr not in (None, ""):
                            items.append({
                                "benchmark": "webshop",
                                "backbone": bb,
                                "method": meth,
                                "method_type": mtype,
                                "sr": float(ws_sr),
                                "sd": float(row["webshop_success_reported_dispersion"]) if row.get("webshop_success_reported_dispersion") not in (None, "") else None,
                                "score": float(ws_sc) if ws_sc not in (None, "") else None,
                                "score_sd": float(row["webshop_score_reported_dispersion"]) if row.get("webshop_score_reported_dispersion") not in (None, "") else None,
                                "source": f"GraphGPO 论文 {src_table}",
                                "protocol_note": ws_note,
                            })
            except Exception:
                pass

        for extra in LITERATURE_EXTRA_BASELINES:
            items.append(dict(extra))

        with self._meta_lock:
            self._lit_sig = cur_sig
            self._lit_cache = items
            return copy.deepcopy(items)

    def _compute_paired_sig(self) -> tuple[Any, ...]:
        rp = self.repo_root / "outputs" / "webshop150-paired-20261119" / "webshop-paired-report" / "report.json"
        return (_file_sig(rp),)

    def _load_webshop_paired(self) -> dict[str, Any]:
        cur_sig = self._compute_paired_sig()
        with self._meta_lock:
            if cur_sig == self._paired_sig:
                return copy.deepcopy(self._paired_cache)

        out: dict[str, Any] = {}
        rp = self.repo_root / "outputs" / "webshop150-paired-20261119" / "webshop-paired-report" / "report.json"
        if rp.is_file():
            try:
                raw = json.loads(rp.read_text(encoding="utf-8"))
                comp = (raw.get("comparisons") or {}).get("v4_150_vs_v4_60") or {}
                per_seed: dict[str, Any] = {}
                for p in (comp.get("pairs") or []):
                    sd_k = str(p.get("seed"))
                    if p.get("status") != "complete":
                        continue
                    per_seed[sd_k] = {
                        "seed": p.get("seed"),
                        "outcomes": p.get("outcomes"),
                        "delta": (p.get("metrics") or {}).get("delta"),
                    }
                out = {
                    "title": comp.get("title"),
                    "status": comp.get("status"),
                    "complete_seeds": comp.get("complete_seeds") or [],
                    "per_seed": per_seed,
                    "aggregate": comp.get("aggregate"),
                }
            except Exception:
                out = {}

        with self._meta_lock:
            self._paired_sig = cur_sig
            self._paired_cache = out
            return copy.deepcopy(out)

    def _compute_r4_arch_sig(self) -> tuple[Any, ...]:
        rp = self.repo_root / "outputs" / "fgpo-r4-colab-20261119" / "retirement_summary.json"
        return (_file_sig(rp),)

    def _load_r4_archive(self) -> dict[str, Any] | None:
        cur_sig = self._compute_r4_arch_sig()
        with self._meta_lock:
            if cur_sig == self._r4_arch_sig:
                return copy.deepcopy(self._r4_arch_cache)
        out = None
        rp = self.repo_root / "outputs" / "fgpo-r4-colab-20261119" / "retirement_summary.json"
        if rp.is_file():
            try:
                out = _sanitize_secrets(json.loads(rp.read_text(encoding="utf-8")), self.secret_replacements)
            except Exception:
                out = None
        with self._meta_lock:
            self._r4_arch_sig = cur_sig
            self._r4_arch_cache = out
            return copy.deepcopy(out)

    def _compute_r4_sig(self) -> tuple[Any, ...]:
        if self.r4_receipt_dir is None or not self.r4_receipt_dir.is_dir():
            return ()
        return (
            _file_sig(self.r4_receipt_dir / "direct_restore_receipt.json"),
            _file_sig(self.r4_receipt_dir / "pi-monitor-observations.jsonl"),
            _file_sig(self.r4_receipt_dir / "colab-direct-credential.log"),
        )

    def check_and_reload_local_files(self) -> bool:
        """Check local CSV / SSH_Document / valid_seen / gdrive_staging / literature / paired / UI HTML changes."""
        sched_changed, sched_count = self._reload_schedule_if_needed()
        ssh_changed = self._reload_ssh_creds_if_needed()

        with self._meta_lock:
            cur_sched = list(self.schedule_18)
            vs_changed = self._compute_valid_seen_sig() != self._valid_seen_sig
            arch_changed = self._compute_archived_colab_sig(cur_sched) != self._archived_colab_sig
            mirror_changed = self._compute_local_colab_mirror_sig(cur_sched) != self._colab_mirror_sig
            lit_changed = self._compute_lit_sig() != self._lit_sig
            paired_changed = self._compute_paired_sig() != self._paired_sig
            r4_arch_changed = self._compute_r4_arch_sig() != self._r4_arch_sig
            cur_r4_sig = self._compute_r4_sig()
            r4_changed = cur_r4_sig != self._r4_sig
            if r4_changed:
                self._r4_sig = cur_r4_sig
                self._r4_raw_cache = None
            cur_ui_mtime = _get_ui_mtime()
            ui_changed = cur_ui_mtime != self._last_watched_ui_mtime
            if ui_changed:
                self._last_watched_ui_mtime = cur_ui_mtime

        if not (sched_changed or ssh_changed or vs_changed or arch_changed or mirror_changed or lit_changed or paired_changed or r4_arch_changed or r4_changed or ui_changed):
            return False

        extra_evs: list[dict[str, Any]] = []
        if sched_changed:
            extra_evs.append({
                "ts": time.strftime("%H:%M:%S"),
                "kind": "schedule",
                "server_id": "all",
                "global_cell_id": None,
                "message": f"排程表已热重载 ({sched_count} 个实验单元)",
            })
        changed_tag = "schedule" if sched_changed else ("ui" if ui_changed else "local")
        self._rebuild_and_publish(changed_tag, extra_events=extra_evs)
        return True

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
        common_sh = _SCRIPT_DIR / "colab_common.sh"
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

    def _poll_colab_once(self, *, force_meta: bool = False, publish: bool = True) -> None:
        sid = "colab"
        if not self.enabled[sid] or self.colab_archive_only:
            return
        if not self.poll_locks[sid].acquire(blocking=False):
            return
        t0 = time.monotonic()
        try:
            with self.state_locks[sid]:
                self.server_states[sid]["polling"] = True
                self.server_states[sid]["last_attempt_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            self._reload_schedule_if_needed()
            with self._meta_lock:
                cur_sched = list(self.schedule_18)

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
                except Exception:
                    if not colab_info:
                        colab_info = {"sessions": {"active": True, "meta_fallback": True}, "usage": {}}

            sessions_info = colab_info.get("sessions") or {"active": True}
            colab_rows = [c for c in cur_sched if c["server_id"] == "colab"]
            output_subdir = Path(self.args.output_dir).name if self.args.output_dir else ""
            if output_subdir:
                live_rows = [c for c in colab_rows if c.get("output_subdir") == output_subdir] or colab_rows
            else:
                live_rows = colab_rows
            spec_fields = ("suite", "benchmark", "method", "seed", "cell_key", "official_eval", "output_subdir", "variant", "status_snapshot")
            colab_specs = [{key: c.get(key) for key in spec_fields} for c in live_rows]
            preamble = (
                "import os\n"
                f"os.environ['FGPO_CFG_PATH'] = {self.args.config_path!r}\n"
                f"os.environ['FGPO_OUT_DIR'] = {self.args.output_dir!r}\n"
                f"os.environ['FGPO_LOG_PATH'] = {self.args.log_path!r}\n"
                f"os.environ['FGPO_SUITE'] = {self.args.suite!r}\n"
                f"os.environ['FGPO_METHOD'] = {self.args.method!r}\n"
                f"os.environ['FGPO_BENCHMARK'] = {self.args.benchmark!r}\n"
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
                st["error_summary"] = None
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
                if "No active sessions" in err_clean:
                    st["status"] = "session_inactive"
                else:
                    st["status"] = "stale" if st.get("telemetry") else "error"
                st["last_error"] = err_clean
                st["error_summary"] = _classify_error(err_clean)
        finally:
            self.poll_locks[sid].release()
            if publish:
                self._rebuild_and_publish(sid)

    def _poll_ssh_server_once(self, sid: str, *, publish: bool = True) -> None:
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
            self._reload_schedule_if_needed()
            with self._meta_lock:
                conn = self.ssh_creds.get(sid)
                cur_sched = list(self.schedule_18)
            if not conn:
                raise RuntimeError(f"Connection entry for {sid} not found in {self.ssh_doc_path.name}")

            specs = [
                {
                    "suite": c["suite"],
                    "benchmark": c["benchmark"],
                    "method": c["method"],
                    "seed": c["seed"],
                    "variant": c.get("variant"),
                    "physical_gpu_ids": c["physical_gpu_ids"],
                    "official_eval": c["official_eval"],
                    "status_snapshot": c.get("status_snapshot"),
                    "sampling_layout": c.get("sampling_layout"),
                    "dependency": c.get("dependency"),
                    "output_subdir": c.get("output_subdir"),
                    "cell_key": c["cell_key"],
                }
                for c in cur_sched if c["server_id"] == sid
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
                st["error_summary"] = None
                st["transport"] = used_transport
                previous = (st.get("telemetry") or {}).get("frozen_author_eval") or {}
                observed = telemetry.setdefault("frozen_author_eval", {})
                for name, saved in previous.items():
                    if saved.get("interface_protocol") != FROZEN_AUTHOR_INTERFACE:
                        continue
                    current = observed.get(name)
                    if current is None:
                        observed[name] = saved
                        continue
                    snapshot = current.get("model_snapshot")
                    if snapshot == saved.get("model_snapshot"):
                        current["rows"] = _merge_frozen_author_rows(saved["rows"], current["rows"], snapshot)
                        current["state"] = current["state"] or saved["state"]
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
                st["error_summary"] = _classify_error(err_clean)
        finally:
            self.poll_locks[sid].release()
            if publish:
                self._rebuild_and_publish(sid)

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

    def trigger_async_refresh(self, *, target_server: str = "all", force_meta: bool = True) -> int:
        """Mark target server(s) polling=True immediately, notify SSE, and spawn independent background probes."""
        targets = ["colab", "host_a", "host_b"] if target_server == "all" else [target_server]
        to_spawn: list[str] = []
        for sid in targets:
            if not self.enabled.get(sid):
                continue
            if sid == "colab" and (self.colab_archive_only or self.r4_receipt_dir is not None):
                continue
            if not self.poll_locks[sid].locked():
                with self.state_locks[sid]:
                    self.server_states[sid]["polling"] = True
                to_spawn.append(sid)

        if to_spawn:
            self._rebuild_and_publish(target_server)
            for sid in to_spawn:
                if sid == "colab":
                    threading.Thread(
                        target=self._poll_colab_once,
                        kwargs={"force_meta": force_meta, "publish": True},
                        daemon=True,
                    ).start()
                else:
                    threading.Thread(
                        target=self._poll_ssh_server_once,
                        args=(sid,),
                        kwargs={"publish": True},
                        daemon=True,
                    ).start()
        else:
            self.check_and_reload_local_files()
        return self.state_seq

    def poll_once(self, *, target_server: str = "all", force_meta: bool = False) -> dict[str, Any]:
        targets = ["colab", "host_a", "host_b"] if target_server == "all" else [target_server]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = []
            for sid in targets:
                if sid == "colab" and self.enabled.get("colab") and not self.colab_archive_only and self.r4_receipt_dir is None:
                    futs.append(pool.submit(self._poll_colab_once, force_meta=force_meta, publish=True))
                elif sid in ("host_a", "host_b") and self.enabled.get(sid):
                    futs.append(pool.submit(self._poll_ssh_server_once, sid, publish=True))
            for f in futs:
                f.result()
        self.check_and_reload_local_files()
        return self.get_snapshot()

    def start_background_threads(self, *, interval: int, stop_event: threading.Event) -> list[threading.Thread]:
        threads: list[threading.Thread] = []

        def _watcher() -> None:
            while not stop_event.wait(2.0):
                try:
                    self.check_and_reload_local_files()
                except Exception:
                    pass

        t_watch = threading.Thread(target=_watcher, name="fgpo-local-watcher", daemon=True)
        t_watch.start()
        threads.append(t_watch)

        for sid in ("colab", "host_a", "host_b"):
            if not self.enabled.get(sid):
                continue
            if sid == "colab" and (self.colab_archive_only or self.r4_receipt_dir is not None):
                continue

            def _server_loop(server_id: str = sid) -> None:
                wake = self._wake_events[server_id]
                if server_id == "colab":
                    self._poll_colab_once(force_meta=True, publish=True)
                else:
                    self._poll_ssh_server_once(server_id, publish=True)
                if interval <= 0:
                    return
                while not stop_event.is_set():
                    wait_sec = float(interval)
                    if server_id == "colab":
                        with self.state_locks["colab"]:
                            st_status = self.server_states["colab"].get("status")
                            last_err = self.server_states["colab"].get("last_error") or ""
                        if st_status == "session_inactive" or "No active sessions" in last_err:
                            wait_sec = float(max(interval * 4, 180))
                    wake.wait(timeout=wait_sec)
                    wake.clear()
                    if stop_event.is_set():
                        break
                    if server_id == "colab":
                        self._poll_colab_once(force_meta=False, publish=True)
                    else:
                        self._poll_ssh_server_once(server_id, publish=True)

            t_srv = threading.Thread(target=_server_loop, name=f"fgpo-poll-{sid}", daemon=True)
            t_srv.start()
            threads.append(t_srv)

        return threads

    def _apply_r4_colab_receipts(self, server: dict[str, Any], schedule_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        directory = self.r4_receipt_dir
        if directory is None or not server.get("enabled"):
            return None
        cur_sig = self._compute_r4_sig()
        cached_raw = None
        with self._meta_lock:
            if cur_sig == self._r4_sig and self._r4_raw_cache is not None:
                cached_raw = self._r4_raw_cache
        if cached_raw is None:
            try:
                recovery = json.loads((directory / "direct_restore_receipt.json").read_text(encoding="utf-8"))["recovery"]
                if recovery["provider_session"] != self.args.session:
                    return None
                records = []
                for line in (directory / "pi-monitor-observations.jsonl").read_text(encoding="utf-8").splitlines():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                successful_renewal = (recovery.get("automatic_credential_renewal") or {}).get("first_success_utc")
                try:
                    log = (directory / "colab-direct-credential.log").read_text(encoding="utf-8")
                    successful_renewal = next((line.split()[0] for line in reversed(log.splitlines())
                                               if line.endswith(" credential renewed")), successful_renewal)
                except OSError:
                    pass
                cached_raw = (recovery, records, successful_renewal)
                with self._meta_lock:
                    self._r4_sig = cur_sig
                    self._r4_raw_cache = cached_raw
            except (OSError, ValueError, KeyError):
                return None

        recovery, records, successful_renewal = cached_raw
        observed = next((r for r in reversed(records) if r.get("state") == "observed"), {})
        latest = next((r for r in reversed(records) if r.get("state") != "monitor_started"), {})
        training_metrics = {}
        for record in records:
            if record.get("state") == "observed":
                training_metrics.update(record.get("training_metrics") or {})
        worker = next(iter(observed.get("workers") or []), {})
        observed_at = observed.get("observed_at")
        monitor = recovery.get("pi_monitoring") or {}
        renewal = recovery.get("automatic_credential_renewal") or {}
        interval = monitor.get("interval_seconds") or 1800
        status = latest.get("state")
        fresh = bool(observed_at and time.time() - datetime.fromisoformat(observed_at.replace("Z", "+00:00")).timestamp() < 2 * interval + 300)
        active = fresh and bool(worker) and status not in ("alert", "complete", "lease_deadline_reached")

        server.update(status="online" if active else "error" if status == "alert"
                      else "session_inactive" if status in ("complete", "lease_deadline_reached") else "stale",
                      transport="PI receipts",
                      last_ok_at=datetime.fromisoformat(observed_at.replace("Z", "+00:00")).astimezone().strftime(
                          "%Y-%m-%d %H:%M:%S") if observed_at else None,
                      last_error=latest.get("reason") if status == "alert" else None)
        telemetry = server.get("telemetry") or {}
        cells = telemetry.setdefault("cells", {})
        completed_cells = 9 if status == "complete" else (observed.get("completed_cells") or 0)
        queue = sorted((row for row in schedule_rows
                        if row["server_id"] == "colab" and row["method"] == "fgpo_tree_mc"),
                       key=lambda row: row["lane_order"])
        for row in queue[:completed_cells]:
            key = row["cell_key"]
            cells[key] = {**training_metrics.get(key, {}), "exists": True, "completed_rounds": row["total_rounds"],
                          "cell_summary": {"status": "complete"}}
        if worker:
            key = f"comparison/{worker['benchmark']}/fgpo_tree_mc/seed-{worker['seed']}"
            if key not in cells:
                metrics = training_metrics.get(key, {})
                cells[key] = {
                    **metrics,
                    "exists": True, "completed_rounds": worker.get("completed_rounds") or 0,
                    "costs_summary": {**(metrics.get("costs_summary") or {}),
                                      "successful_adamw_count": worker.get("optimizer_steps_in_last_round"),
                                      "last_phase_role": worker.get("last_cost_role")},
                }
            if active:
                telemetry["proc_list"] = [{
                    "lane": "Colab GPU 0", "role": "fgpo.tree_runner", "pid": worker.get("pid"),
                    "target_method": "fgpo_tree_mc", "target_suite": "comparison",
                    "target_benchmark": worker["benchmark"], "target_seeds": [worker["seed"]],
                }]
        server["telemetry"] = telemetry
        return {"state": server["status"], "observed_at": observed_at, "worker": worker,
                "completed_cells": completed_cells, "interval_seconds": interval, "model": monitor.get("model"),
                "effort": monitor.get("effort"), "renewed_at": successful_renewal,
                "renewal_deadline": renewal.get("deadline_utc"), "alert": server["last_error"]}

    def _summarize_cell_live(
        self,
        sched: dict[str, Any],
        raw_cell: dict[str, Any] | None,
        *,
        is_active_runner: bool,
        lane_has_queue: bool,
        valid_seen_map: dict[int, dict[str, Any]],
        webshop_paired_map: dict[str, Any],
        actual_gpu_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        merged = dict(sched)
        merged["auto_discovered"] = bool(sched.get("auto_discovered", False))
        cd = raw_cell or {}
        exists = bool(cd.get("exists"))
        merged["exists"] = exists
        rounds = cd.get("training_rounds") or []
        rounds_done = int(cd.get("completed_rounds", len(rounds)) or 0)
        csum = cd.get("costs_summary") or {}
        g_done = max(int(csum.get("current_round_groups_done") or 0), int(cd.get("current_round_group_files") or 0))
        cell_sum = cd.get("cell_summary") or {}
        evals = dict(cd.get("evaluations") or {})
        stage_rounds = cd.get("stage_rounds")
        if stage_rounds is not None:
            merged["planned_total_rounds"] = sched["total_rounds"]
            merged["total_rounds"] = stage_rounds
            if sched.get("total_training_trajectories") is not None and sched.get("total_rounds"):
                merged["total_training_trajectories"] = (
                    sched["total_training_trajectories"] * stage_rounds // sched["total_rounds"]
                )
            merged["output_subdir"] = cd["output_subdir"]

        eval_valid_seen_sr_pct = None
        if (sched["server_id"] == "colab" and sched["suite"] == "comparison" and sched["benchmark"] == "alfworld"
                and sched["method"] == "fgpo" and not sched.get("variant")):
            vs_entry = valid_seen_map.get(int(sched["seed"]))
            if vs_entry:
                evals["valid_seen"] = {"summary": vs_entry, "completed_count": vs_entry.get("evaluated_count"), "success_count": vs_entry.get("success_count")}
                eval_valid_seen_sr_pct = vs_entry.get("SR")

        stage_complete = cell_sum.get("status") == "complete" or bool(cell_sum.get("evaluation"))
        eval_started = any((v.get("completed_count") or 0) > 0 for k, v in evals.items() if k != "valid_seen")
        has_progress = exists and (rounds_done > 0 or int(csum.get("total_events") or 0) > 0 or eval_started)

        if stage_complete:
            live_status = "completed" if stage_rounds is None else "queued"
        elif is_active_runner:
            live_status = "evaluating" if eval_started else "running"
        elif has_progress:
            live_status = "paused"
        else:
            live_status = "queued" if lane_has_queue and sched.get("status_snapshot") != "deferred" else "planned"

        cfg_meta = cd.get("cell_config") or {}
        var_low = str(sched.get("variant") or "").lower()
        sub_low = str(sched.get("output_subdir") or "").lower()
        backbone = (
            cfg_meta.get("backbone")
            or sched.get("backbone")
            or ("1.5B" if ("1p5b" in var_low or "1.5b" in var_low or "1p5b" in sub_low) else "7B")
        )
        hparams = {
            "lr": cfg_meta.get("learning_rate"),
            "radius": cfg_meta.get("credit_radius"),
            "kappa": cfg_meta.get("credit_prior_weight"),
            "transfer_max": cfg_meta.get("credit_transfer_max"),
            "entropy_coef": cfg_meta.get("entropy_coef"),
            "kl_coef": cfg_meta.get("kl_coef"),
            "output_tokens": cfg_meta.get("output_tokens"),
            "model_name": cfg_meta.get("model_name"),
            "study_protocol": cfg_meta.get("study_protocol"),
        } if cfg_meta else None

        last_r = rounds[-1] if rounds else None
        last_m = (last_r or {}).get("metrics") or {}
        last_optimization = (last_r or {}).get("optimization") or {}
        rooted_sr = ((last_r or {}).get("tree") or {}).get("rooted_success")
        latest_sr = None
        latest_succ = None
        latest_traj = None
        if last_r and last_r.get("trajectories"):
            latest_succ = int(last_r.get("successes") or 0)
            latest_traj = int(last_r.get("trajectories") or 128)
            latest_sr = round(100.0 * latest_succ / latest_traj, 2)

        is_tree = sched["method"].startswith("fgpo_tree_")
        sr_series: list[tuple[int, float]] = []
        for r in rounds:
            if is_tree:
                rooted = (r.get("tree") or {}).get("rooted_success")
                if rooted is not None:
                    sr_series.append((r.get("round"), 100.0 * rooted))
            elif r.get("trajectories"):
                sr_series.append((r.get("round"), 100.0 * (r.get("successes") or 0) / r["trajectories"]))
        peak_point = max(sr_series, key=lambda item: item[1]) if sr_series else None
        recent5 = sr_series[-5:]
        v4_60 = cd.get("webshop_v4_60") or {}

        eval_sr_pct = None
        eval_metric_name = "SR"
        eval_score_pct = None
        primary_split = None
        primary_eval_obj: dict[str, Any] = {}
        if cell_sum.get("evaluation"):
            ev_map = cell_sum["evaluation"]
            if "valid_unseen" in ev_map and ev_map["valid_unseen"].get("SR") is not None:
                primary_split = "valid_unseen"
                primary_eval_obj = ev_map["valid_unseen"]
                eval_sr_pct = round(float(primary_eval_obj["SR"]), 2)
                eval_metric_name = "SR"
            else:
                for sp_name in ("test", "test500", "val79", "val", "valid", "graphgpo_released_validation"):
                    if sp_name in ev_map and isinstance(ev_map[sp_name], dict):
                        wb = ev_map[sp_name]
                        primary_split = sp_name
                        primary_eval_obj = wb
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

        eval_breakdown: dict[str, Any] = {}
        ev_sum_fallback = ((evals.get(primary_split) or {}).get("summary") or {}) if primary_split else {}
        tt_src = primary_eval_obj.get("task_types") or ev_sum_fallback.get("task_types")
        if isinstance(tt_src, dict) and tt_src:
            eval_breakdown["task_types"] = {
                k: {
                    "sr": round(float(v["SR"]), 2) if v.get("SR") is not None else None,
                    "success_count": v.get("success_count"),
                    "evaluated_count": v.get("evaluated_count"),
                    "official_task_count": v.get("official_task_count"),
                }
                for k, v in tt_src.items() if isinstance(v, dict)
            }
        rd_src = primary_eval_obj.get("recipe_depths") or ev_sum_fallback.get("recipe_depths")
        if isinstance(rd_src, dict) and rd_src:
            eval_breakdown["recipe_depths"] = {
                str(k): {
                    "sr": round(float(v["SR"]), 2) if v.get("SR") is not None else None,
                    "success_count": v.get("success_count"),
                    "evaluated_count": v.get("evaluated_count"),
                    "official_task_count": v.get("official_task_count"),
                }
                for k, v in rd_src.items() if isinstance(v, dict)
            }
        if sched["benchmark"] == "webshop":
            ws_sp = primary_split or "test"
            ws_ev = evals.get(ws_sp) or {}
            if ws_ev.get("zero_score_count") is not None:
                eval_breakdown["zero_score_count"] = ws_ev.get("zero_score_count")
            if ws_ev.get("nonzero_score_count") is not None:
                eval_breakdown["nonzero_score_count"] = ws_ev.get("nonzero_score_count")
            if ws_ev.get("purchased_count") is not None:
                eval_breakdown["purchased_count"] = ws_ev.get("purchased_count")
            if (sched["method"] == "fgpo" and not sched.get("variant")
                    and backbone == "7B" and int(merged.get("total_rounds") or 0) == 150):
                p_seed = webshop_paired_map.get(str(sched["seed"]))
                if isinstance(p_seed, dict):
                    eval_breakdown["paired_vs_60"] = p_seed

        # Compute idle_seconds, sec_per_round, eta_seconds, and gpu_hours
        last_mtime = cd.get("last_mtime")
        probe_epoch = cd.get("probe_epoch")
        idle_seconds: float | None = None
        if last_mtime is not None:
            ref_t = float(probe_epoch) if probe_epoch is not None else time.time()
            idle_seconds = round(max(0.0, ref_t - float(last_mtime)), 1)

        sess_el = csum.get("session_elapsed_seconds")
        sec_per_round: float | None = cd.get("recent_sec_per_round")
        if sec_per_round is None and sess_el is not None and float(sess_el) > 0:
            if rounds_done > 0:
                sec_per_round = round(float(sess_el) / rounds_done, 2)
            elif g_done > 0:
                sec_per_round = round(float(sess_el) * (16.0 / min(g_done, 16)), 2)

        total_rounds = int(merged.get("total_rounds") or 0)
        eta_seconds: float | None = None
        if live_status == "completed":
            eta_seconds = 0.0
        elif sec_per_round is not None and sec_per_round > 0 and total_rounds > 0:
            intra = (min(g_done, 15) / 16.0) if live_status == "running" else 0.0
            rem_rounds = max(0.0, float(total_rounds - rounds_done) - intra)
            eta_seconds = round(rem_rounds * float(sec_per_round), 1)

        alloc_dev_sec = cd.get("allocated_device_seconds")
        if alloc_dev_sec is None:
            cs_blk = cell_sum if isinstance(cell_sum, dict) else {}
            alloc_dev_sec = (
                ((cs_blk.get("costs") or {}).get("allocated_device_seconds"))
                or (((cs_blk.get("training") or {}).get("costs") or {}).get("allocated_device_seconds"))
                or ((last_r or {}).get("costs") or {}).get("allocated_device_seconds")
            )
        gpu_hours: float | None = None
        gpu_hours_estimated = False
        if alloc_dev_sec is not None:
            gpu_hours = round(float(alloc_dev_sec) / 3600.0, 2)
        elif sess_el is not None:
            dev_cnt = len(actual_gpu_ids) if actual_gpu_ids else int(sched.get("device_count") or 1)
            gpu_hours = round(float(sess_el) * max(1, dev_cnt) / 3600.0, 2)
            gpu_hours_estimated = True

        tot_zero_g = sum(int((r.get("diagnostics") or {}).get("all_zero_groups") or 0) for r in rounds) if rounds else 0
        tot_mixed_g = sum(int((r.get("diagnostics") or {}).get("mixed_groups") or 0) for r in rounds) if rounds else 0
        tot_one_g = sum(int((r.get("diagnostics") or {}).get("all_one_groups") or 0) for r in rounds) if rounds else 0

        merged.update({
            "live_status": live_status,
            "backbone": backbone,
            "hparams": hparams,
            "actual_gpu_ids": actual_gpu_ids if actual_gpu_ids else _parse_gpu_indices(sched.get("physical_gpu_ids") or ""),
            "rounds_done": rounds_done,
            "current_round_groups_done": g_done if live_status != "completed" else 16,
            "groups_per_round": 16,
            "latest_sr_pct": latest_sr,
            "latest_successes": latest_succ,
            "latest_trajectories": latest_traj,
            "latest_rooted_sr_pct": round(100 * rooted_sr, 2) if rooted_sr is not None else None,
            "peak_sr_pct": round(peak_point[1], 2) if peak_point else None,
            "peak_sr_round": peak_point[0] if peak_point else None,
            "recent5_sr_pct": round(sum(v for _, v in recent5) / len(recent5), 2) if recent5 else None,
            "first_sr_pct": round(sr_series[0][1], 2) if sr_series else None,
            "first_sr_round": sr_series[0][0] if sr_series else None,
            "trajectories_done": sum(int(r.get("trajectories") or 0) for r in rounds),
            "v4_60_sr_pct": v4_60.get("SR"),
            "v4_60_score": v4_60.get("Score"),
            "v4_60_success_count": v4_60.get("success_count"),
            "v4_60_evaluated_count": v4_60.get("evaluated_count"),
            "optimizer_steps": max(int((last_r or {}).get("optimizer_steps") or 0),
                                   int(csum.get("successful_adamw_count") or 0)),
            "latest_loss": last_optimization.get("ppo_loss_per_weighted_token", last_m.get("loss")),
            "latest_kl": last_optimization.get("reference_capped_k3_per_weighted_token", last_m.get("reference_kl")),
            "latest_entropy": last_optimization.get("entropy_per_weighted_token", last_m.get("entropy")),
            "latest_grad_norm": last_m.get("gradient_norm"),
            "latest_credit": (last_r or {}).get("credit"),
            "latest_admissible_frac": ((last_r or {}).get("diagnostics") or {}).get("admissible_valid_fraction"),
            "latest_clip_frac": last_m.get("ppo_clip_fraction"),
            "total_all_zero_groups": tot_zero_g,
            "total_mixed_groups": tot_mixed_g,
            "total_all_one_groups": tot_one_g,
            "env_steps": int(csum.get("env_steps") or 0),
            "input_tokens": int(csum.get("input_tokens") or 0),
            "output_tokens": int(csum.get("output_tokens") or 0),
            "forward_tokens": int(csum.get("forward_tokens") or 0),
            "total_events": int(csum.get("total_events") or 0),
            "session_elapsed_seconds": sess_el,
            "allocated_device_seconds": round(float(alloc_dev_sec), 2) if alloc_dev_sec is not None else None,
            "gpu_hours": gpu_hours,
            "gpu_hours_estimated": gpu_hours_estimated,
            "last_group_id": csum.get("last_group_id"),
            "last_phase_role": csum.get("last_phase_role"),
            "checkpoint_snapshot": (cd.get("checkpoint_current") or {}).get("snapshot"),
            "eval_sr_pct": eval_sr_pct,
            "eval_metric_name": eval_metric_name,
            "eval_score_pct": eval_score_pct,
            "eval_valid_seen_sr_pct": eval_valid_seen_sr_pct,
            "eval_breakdown": eval_breakdown or None,
            "evaluations": evals,
            "sec_per_round": round(float(sec_per_round), 1) if sec_per_round is not None else None,
            "eta_seconds": eta_seconds,
            "idle_seconds": idle_seconds,
        })
        return merged

    def _build_discovered_schedule_rows(self, servers_snap: dict[str, Any], base_sched: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        existing_gids = {c["global_cell_id"] for c in base_sched}
        next_idx = len(base_sched) + 1
        host_labels = {"colab": "Colab", "host_a": "Host A", "host_b": "Host B"}
        lane_next_order: dict[str, int] = {}
        for c in base_sched:
            lid = c["lane_id"]
            lane_next_order[lid] = max(lane_next_order.get(lid, 0), int(c.get("lane_order") or 0) + 1)

        for sid in ("colab", "host_a", "host_b"):
            tel = (servers_snap.get(sid) or {}).get("telemetry") or {}
            for disc in (tel.get("discovered_cells") or []):
                ck = disc.get("cell_key")
                if not ck:
                    continue
                gid = f"{sid}:{ck}"
                if gid in existing_gids:
                    continue
                existing_gids.add(gid)
                phys = str(disc.get("physical_gpu_ids") or ("0" if sid != "host_a" else "3+4"))
                _, lane_id, lane_title = _map_row_to_lane(host_labels[sid], phys)
                lo = lane_next_order.get(lane_id, 100)
                lane_next_order[lane_id] = lo + 1
                bm = str(disc.get("benchmark") or "alfworld")
                tot_r = int(disc.get("total_rounds") or 60)
                meth = str(disc.get("method") or "fgpo")
                dev_cnt = int(disc.get("device_count") or (2 if "+" in phys else 1))
                rows.append({
                    "host_or_card": host_labels[sid],
                    "physical_gpu_ids": phys,
                    "device_count": dev_cnt,
                    "physical_gpu_count": dev_cnt,
                    "lane_order": lo,
                    "index": next_idx,
                    "server_id": sid,
                    "lane_id": lane_id,
                    "lane_title": lane_title,
                    "suite": str(disc.get("suite") or "comparison"),
                    "benchmark": bm,
                    "method": meth,
                    "seed": int(disc.get("seed") or 0),
                    "total_rounds": tot_r,
                    "total_training_trajectories": None if meth == "fgpo_tree_mc" else tot_r * 128,
                    "horizon": int(disc.get("horizon") or (30 if bm == "textcraft" else (50 if bm == "alfworld" else (15 if bm == "webshop" else 100)))),
                    "official_eval": str(disc.get("official_eval") or "valid_unseen"),
                    "official_endpoint_tasks": _default_endpoint_tasks(bm),
                    "status_snapshot": "auto_discovered",
                    "sampling_layout": "auto_discovered",
                    "dependency": f"自动发现 ({disc.get('output_subdir') or 'live'})",
                    "output_subdir": str(disc.get("output_subdir") or ""),
                    "variant": str(disc.get("variant") or ""),
                    "backbone": disc.get("backbone"),
                    "cell_key": ck,
                    "global_cell_id": gid,
                    "auto_discovered": True,
                })
                next_idx += 1
        return rows

    def _build_lane_definitions(
        self,
        combined_schedule: list[dict[str, Any]],
        r4_colab: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        preset = {ld["lane_id"]: dict(ld) for ld in LANE_DEFINITIONS}
        seen_ids: set[str] = set()
        ordered: list[dict[str, Any]] = []

        def _register_lane(sid: str, lid: str, title: str, phys: str) -> None:
            if lid in seen_ids:
                return
            seen_ids.add(lid)
            if lid in preset:
                ordered.append(dict(preset[lid]))
                return
            gids = _parse_gpu_indices(phys)
            if sid == "host_a":
                short_ln = f"L40 {phys}" if len(gids) >= 2 else f"GPU {phys}"
                role_sl = f"Host A {short_ln}"
            elif sid == "host_b":
                short_ln = f"GPU {phys}"
                role_sl = f"Queue B{phys} (GPU {phys} · 80 GiB cap)"
            else:
                short_ln = "Colab GPU 0"
                role_sl = "Colab G4 / PRO 6000"
            ordered.append({
                "lane_id": lid,
                "server_id": sid,
                "lane_title": title,
                "short_lane": short_ln,
                "role_slot": role_sl,
                "gpu_indices": gids or [0],
            })

        show_colab_lane = (not self.colab_archive_only) or (bool(r4_colab) and r4_colab.get("state") == "online")
        for sc in combined_schedule:
            sid = sc["server_id"]
            if sid == "colab" and not show_colab_lane:
                continue
            _register_lane(sid, sc["lane_id"], sc["lane_title"], sc.get("physical_gpu_ids") or "0")

        # Ensure live Host A 3+4 / Host B 0, 2 lanes appear even when filtered
        for must_id in ("host_a_l40_3_4", "host_b_gpu_0", "host_b_gpu_2"):
            if must_id in preset and must_id not in seen_ids:
                seen_ids.add(must_id)
                ordered.append(dict(preset[must_id]))

        srv_rank = {"host_a": 0, "host_b": 1, "colab": 2}
        lane_rank = {ld["lane_id"]: idx for idx, ld in enumerate(LANE_DEFINITIONS)}
        ordered.sort(key=lambda d: (srv_rank.get(d["server_id"], 9), lane_rank.get(d["lane_id"], 50), d["lane_id"]))
        return ordered

    def _build_snapshot_internal(self) -> dict[str, Any]:
        servers_snap: dict[str, Any] = {}
        for sid in ("colab", "host_a", "host_b"):
            with self.state_locks[sid]:
                servers_snap[sid] = copy.deepcopy(self.server_states[sid])

        with self._meta_lock:
            base_sched = [dict(c) for c in self.schedule_18 if c.get("status_snapshot") != "relocated"]
            sched_mtime = self.schedule_mtime
            replacements = list(self.secret_replacements)

        valid_seen_map = self._load_local_valid_seen()
        lit_baselines = self._load_literature_baselines()
        webshop_paired = self._load_webshop_paired()
        r4_archive = self._load_r4_archive()
        r4_colab = self._apply_r4_colab_receipts(servers_snap["colab"], base_sched)
        archived_colab = self._load_archived_colab_cells(base_sched)
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
        local_mirrors = self._load_local_colab_mirrors(base_sched)
        if local_mirrors:
            colab_tel = servers_snap["colab"].get("telemetry") or {}
            colab_cells = colab_tel.setdefault("cells", {})
            for key, mirror in local_mirrors.items():
                live = colab_cells.get(key)
                if not isinstance(live, dict) or not live.get("exists"):
                    colab_cells[key] = mirror
                elif len(live.get("training_rounds") or []) < len(mirror.get("training_rounds") or []):
                    merged_cell = dict(live)
                    for mk in (
                        "training_rounds", "recent_sec_per_round", "allocated_device_seconds",
                        "cell_config", "training_summary", "cell_summary",
                        "checkpoint_current", "evaluations", "output_subdir",
                    ):
                        if mk in mirror and mirror[mk] is not None:
                            merged_cell[mk] = mirror[mk]
                    if not merged_cell.get("costs_summary"):
                        merged_cell["costs_summary"] = mirror.get("costs_summary")
                    colab_cells[key] = merged_cell
                elif mirror.get("cell_config") and not live.get("cell_config"):
                    live["cell_config"] = mirror["cell_config"]
            servers_snap["colab"]["telemetry"] = colab_tel
        if isinstance(colab_tel, dict) and isinstance(colab_tel.get("cells"), dict):
            for seed_i, vs_entry in valid_seen_map.items():
                ck0 = f"comparison/alfworld/fgpo/seed-{seed_i}"
                if ck0 in colab_tel["cells"] and colab_tel["cells"][ck0].get("exists"):
                    colab_tel["cells"][ck0].setdefault("evaluations", {})["valid_seen"] = {
                        "summary": vs_entry,
                        "completed_count": vs_entry.get("evaluated_count"),
                        "success_count": vs_entry.get("success_count"),
                    }

        discovered_rows = self._build_discovered_schedule_rows(servers_snap, base_sched)
        combined_schedule = base_sched + discovered_rows
        lanes_def = self._build_lane_definitions(combined_schedule, r4_colab)

        # Direct runner-to-cell matching across each server (supports subset GPU execution e.g. GPU 3 inside 3+4 queue)
        active_runner_by_gid: dict[str, dict[str, Any]] = {}
        active_runner_by_lid: dict[str, dict[str, Any]] = {}
        for sid in ("colab", "host_a", "host_b"):
            srv = servers_snap[sid]
            if srv.get("status") != "online":
                continue
            tel = srv.get("telemetry") or {}
            procs = tel.get("proc_list") or []
            srv_cells = [c for c in combined_schedule if c["server_id"] == sid]
            for rproc in procs:
                if "runner" not in (rproc.get("role") or "") or not rproc.get("target_method"):
                    continue
                t_suite = rproc.get("target_suite") or "comparison"
                t_bench = rproc.get("target_benchmark")
                t_meth = rproc.get("target_method")
                t_seeds = rproc.get("target_seeds") or []
                output_subdir = Path(rproc.get("target_output_dir") or "").name
                matched_cell = None
                for sd in t_seeds:
                    exact = next(
                        (c for c in srv_cells
                         if c["suite"] == t_suite and c["benchmark"] == t_bench
                         and c["method"] == t_meth and c["seed"] == sd
                         and output_subdir and c.get("output_subdir") == output_subdir),
                        None,
                    )
                    fallback = next(
                        (c for c in srv_cells
                         if c["suite"] == t_suite and c["benchmark"] == t_bench
                         and c["method"] == t_meth and c["seed"] == sd
                         and not c.get("output_subdir")),
                        None,
                    )
                    cand = exact or fallback
                    if cand is None:
                        continue
                    raw_c = (tel.get("cells") or {}).get(cand["cell_key"]) or {}
                    cs_obj = raw_c.get("cell_summary") or {}
                    if not (cs_obj.get("status") == "complete" or cs_obj.get("evaluation")):
                        matched_cell = cand
                        break
                if matched_cell is not None:
                    active_runner_by_gid[matched_cell["global_cell_id"]] = rproc
                    active_runner_by_lid[matched_cell["lane_id"]] = rproc

        lane_has_queue_map: dict[str, bool] = {}
        for ldef in lanes_def:
            lid = ldef["lane_id"]
            sid = ldef["server_id"]
            srv = servers_snap[sid]
            tel = (srv.get("telemetry") or {}) if srv.get("status") == "online" else {}
            l_gpu_key = "+".join(str(x) for x in ldef["gpu_indices"])
            l_procs = [
                p for p in (tel.get("proc_list") or [])
                if p.get("lane") == ldef["short_lane"] or p.get("gpu_key") == l_gpu_key
            ]
            lane_has_queue_map[lid] = (
                (lid in active_runner_by_lid)
                or (sid == "colab" and bool(r4_colab) and r4_colab.get("state") == "online")
                or any(p.get("role") in ("queue_controller", "supervisor") for p in l_procs)
            )

        wp_per_seed = (webshop_paired.get("per_seed") or {}) if isinstance(webshop_paired, dict) else {}
        merged_18: list[dict[str, Any]] = []
        for sc in combined_schedule:
            lid = sc["lane_id"]
            gid = sc["global_cell_id"]
            srv_tel = (servers_snap.get(sc["server_id"]) or {}).get("telemetry") or {}
            raw_cell = (srv_tel.get("cells") or {}).get(sc["cell_key"])
            rproc = active_runner_by_gid.get(gid)
            merged_18.append(self._summarize_cell_live(
                sc,
                raw_cell,
                is_active_runner=(rproc is not None),
                lane_has_queue=lane_has_queue_map.get(lid, False),
                valid_seen_map=valid_seen_map,
                webshop_paired_map=wp_per_seed,
                actual_gpu_ids=(rproc.get("gpu_ids") if rproc else None),
            ))

        lanes_out: list[dict[str, Any]] = []
        for ldef in lanes_def:
            lid = ldef["lane_id"]
            sid = ldef["server_id"]
            srv = servers_snap[sid]
            tel = (srv.get("telemetry") or {}) if srv.get("status") == "online" else {}
            lane_cells = sorted([c for c in merged_18 if c["lane_id"] == lid], key=lambda x: x["lane_order"])
            paused_cells = [c for c in lane_cells if c["live_status"] == "paused"]
            active_cell = next((c for c in lane_cells if c["live_status"] in ("running", "evaluating")), None)
            if active_cell is None:
                active_cell = next((c for c in lane_cells if c["live_status"] == "queued"), None)
            if active_cell is None:
                active_cell = min(
                    paused_cells,
                    key=lambda c: c["idle_seconds"] if c["idle_seconds"] is not None else float("inf"),
                    default=None,
                )
            if active_cell is None:
                active_cell = next((c for c in lane_cells if c["status_snapshot"] == "transferring"
                                    and c["live_status"] == "planned"), None)
            if active_cell is None:
                non_def = [c for c in lane_cells if c.get("status_snapshot") != "deferred"]
                active_cell = next((c for c in non_def if c["live_status"] == "completed"), None) or (non_def[-1] if non_def else (lane_cells[-1] if lane_cells else None))
            completed_cells = [c for c in lane_cells if c["live_status"] == "completed"]
            queued_cells = [c for c in lane_cells if c["live_status"] == "queued" and c is not active_cell]
            planned_cells = [c for c in lane_cells if c["live_status"] == "planned"]

            act_gpu_set = set(active_cell.get("actual_gpu_ids") or []) if active_cell else set()
            lane_gpu_set = set(ldef["gpu_indices"]) | act_gpu_set
            lane_gpus = [g for g in (tel.get("gpus") or []) if g.get("index") in lane_gpu_set]
            l_gpu_key = "+".join(str(x) for x in ldef["gpu_indices"])
            runner_proc = active_runner_by_lid.get(lid)
            procs = [
                p for p in (tel.get("proc_list") or [])
                if p.get("lane") == ldef["short_lane"] or p.get("gpu_key") == l_gpu_key
                or (runner_proc and (p.get("pid") == runner_proc.get("pid") or p.get("ppid") == runner_proc.get("pid")))
            ]
            if runner_proc is None:
                runner_proc = next((p for p in procs if "runner" in (p.get("role") or "")), None)
            sup_proc = next((p for p in procs if p.get("role") in ("queue_controller", "supervisor")), None)
            env_workers = sum(1 for p in procs if p.get("role") == "fgpo.envs")

            lanes_out.append({
                "lane_id": lid,
                "server_id": sid,
                "server_status": srv.get("status"),
                "lane_title": ldef["lane_title"],
                "short_lane": ldef["short_lane"],
                "role_slot": ldef["role_slot"],
                "gpu_indices": sorted(lane_gpu_set),
                "gpus": lane_gpus,
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

        enabled_list = [s for s in servers_snap.values() if s.get("enabled")]
        live_enabled_list = [s for s in enabled_list if not s.get("archived_only")]
        archived_servers = [s for s in enabled_list if s.get("archived_only")]
        online_count = sum(1 for s in live_enabled_list if s.get("status") == "online")
        usable_pool_gpus = 0
        active_gpus: list[dict[str, Any]] = []
        total_pids = 0
        for s in live_enabled_list:
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
        gpu_hours_cum = round(sum(float(c.get("gpu_hours") or 0.0) for c in merged_18), 2)

        host_a_arch = ((servers_snap["host_a"].get("telemetry") or {}).get("archived_cells") or {})
        colab_usage = ((servers_snap["colab"].get("colab") or {}).get("usage") or {})

        active_plan_cells = [c for c in merged_18 if c.get("status_snapshot") != "deferred"]
        deferred_cells = [c for c in merged_18 if c.get("status_snapshot") == "deferred"]

        cluster_summary = {
            "servers_enabled": len(live_enabled_list),
            "servers_live_enabled": len(live_enabled_list),
            "servers_archived": len(archived_servers),
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
            "cells_active_plan": len(active_plan_cells),
            "cells_deferred": len(deferred_cells),
            "cells_discovered": sum(1 for c in merged_18 if c.get("auto_discovered")),
            "cells_completed": sum(1 for c in active_plan_cells if c["live_status"] == "completed"),
            "cells_running": sum(1 for c in merged_18 if c["live_status"] in ("running", "evaluating")),
            "cells_paused": sum(1 for c in merged_18 if c["live_status"] == "paused"),
            "cells_queued": sum(1 for c in active_plan_cells if c["live_status"] == "queued"),
            "cells_planned": sum(1 for c in active_plan_cells if c["live_status"] == "planned"),
            "archived_cells_count": len(host_a_arch),
            "env_steps_cum": env_steps_cum,
            "input_tokens_cum": in_tok_cum,
            "output_tokens_cum": out_tok_cum,
            "gpu_hours_cum": gpu_hours_cum,
            "active_pids_count": total_pids,
            "colab_balance": colab_usage.get("balance"),
            "colab_rate": colab_usage.get("rate"),
        }

        colab_st = servers_snap["colab"]
        frozen_author_eval = {
            name: {"expected": count, "interface_protocol": FROZEN_AUTHOR_INTERFACE,
                   "model_snapshot": FROZEN_AUTHOR_SNAPSHOTS.get(name), "rows": [], "sources": {}}
            for _, jobs in FROZEN_AUTHOR_JOBS.values() for name, count in jobs.items()
        }
        for sid in ("host_a", "host_b"):
            server = servers_snap[sid]
            streams = (server.get("telemetry") or {}).pop("frozen_author_eval", {})
            for name, stream in streams.items():
                if stream.get("interface_protocol") != FROZEN_AUTHOR_INTERFACE:
                    continue
                result = frozen_author_eval[name]
                snapshot = stream.get("model_snapshot")
                if snapshot is not None and result["model_snapshot"] not in (None, snapshot):
                    continue
                result["model_snapshot"] = result["model_snapshot"] or snapshot
                result["rows"] = _merge_frozen_author_rows(result["rows"], stream["rows"], result["model_snapshot"])
                result["sources"][sid] = {"connection": server["status"],
                                          "interface_protocol": FROZEN_AUTHOR_INTERFACE,
                                          "model_snapshot": snapshot, **stream["state"]}
        payload = {
            "protocol": "fgpo-r3-published-v4",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "schedule_mtime": sched_mtime,
            "servers": servers_snap,
            "r4_colab": r4_colab,
            "r4_archive": r4_archive,
            "literature_baselines": lit_baselines,
            "webshop_paired": webshop_paired,
            "frozen_author_eval": {name: frozen_author_eval[name] for name in FROZEN_AUTHOR_JOBS["7b"][1]},
            "frozen_author_eval_1p5b": {name: frozen_author_eval[name] for name in FROZEN_AUTHOR_JOBS["1p5b"][1]},
            "frozen_author_eval_1p5b_webshop_pair": {
                "interface_protocol": FROZEN_AUTHOR_INTERFACE,
                "original_model_snapshot": frozen_author_eval["webshop-original"]["model_snapshot"],
                "adapted_model_snapshot": frozen_author_eval["webshop-adapted"]["model_snapshot"],
                **_paired_frozen_webshop(
                    frozen_author_eval["webshop-original"]["rows"], frozen_author_eval["webshop-adapted"]["rows"]
                ),
            },
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
        return _sanitize_secrets(payload, replacements)

    def _diff_and_collect_events(self, snap: dict[str, Any], *, is_init: bool = False) -> list[dict[str, Any]]:
        ts_now = time.strftime("%H:%M:%S")
        events: list[dict[str, Any]] = []

        for sid, srv in (snap.get("servers") or {}).items():
            if not srv.get("enabled"):
                continue
            cur_st = str(srv.get("status") or "")
            prev_st = self._prev_server_statuses.get(sid)
            self._prev_server_statuses[sid] = cur_st
            if not is_init and prev_st is not None and prev_st != cur_st:
                if not (prev_st == "initializing" and cur_st == "initializing"):
                    events.append({
                        "ts": ts_now,
                        "kind": "status",
                        "server_id": sid,
                        "global_cell_id": None,
                        "message": f"{srv.get('display_name') or sid} 状态更新：{cur_st}",
                    })

        for c in (snap.get("schedule") or []):
            gid = c["global_cell_id"]
            sid = c["server_id"]
            srv_status = ((snap.get("servers") or {}).get(sid) or {}).get("status")
            # Wait until server has completed its first probe (or local archive exists) before baseline-locking cell state
            if srv_status == "initializing" and not c.get("exists"):
                continue

            label = f"{c['benchmark']} · {c['method']} s{c['seed']}" + (f"@{c['variant']}" if c.get("variant") else "")
            if c.get("auto_discovered") and gid not in self._seen_discovered_ids:
                self._seen_discovered_ids.add(gid)
                events.append({
                    "ts": ts_now,
                    "kind": "discovered",
                    "server_id": sid,
                    "global_cell_id": gid,
                    "message": f"自动发现实验 {label} (R{c['rounds_done']}/{c['total_rounds']})",
                })

            prev = self._prev_cell_states.get(gid)
            cur_rec = {
                "rounds_done": int(c.get("rounds_done") or 0),
                "live_status": c.get("live_status"),
                "eval_sr_pct": c.get("eval_sr_pct"),
            }
            self._prev_cell_states[gid] = cur_rec
            if is_init or prev is None:
                continue

            if cur_rec["rounds_done"] > prev["rounds_done"]:
                sr_val = c.get("latest_rooted_sr_pct") if c.get("latest_rooted_sr_pct") is not None else c.get("latest_sr_pct")
                sr_txt = f" · SR {sr_val:.2f}%" if sr_val is not None else ""
                events.append({
                    "ts": ts_now,
                    "kind": "round",
                    "server_id": sid,
                    "global_cell_id": gid,
                    "message": f"{label} 完成 R{cur_rec['rounds_done']}/{c['total_rounds']}{sr_txt}",
                })

            if (cur_rec["eval_sr_pct"] is not None and prev["eval_sr_pct"] is None) or (
                cur_rec["live_status"] == "completed" and prev["live_status"] != "completed"
            ):
                if cur_rec["eval_sr_pct"] is not None:
                    m_name = c.get("eval_metric_name") or "SR"
                    ev_txt = f" · {m_name} {cur_rec['eval_sr_pct']:.2f}%"
                else:
                    ev_txt = ""
                events.append({
                    "ts": ts_now,
                    "kind": "eval",
                    "server_id": sid,
                    "global_cell_id": gid,
                    "message": f"{label} 完成官方终评{ev_txt}",
                })

        return events

    def _rebuild_and_publish(
        self,
        changed_server: str,
        *,
        is_init: bool = False,
        extra_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        snap = self._build_snapshot_internal()
        with self.state_cond:
            self.state_seq += 1
            seq = self.state_seq
            diff_evs = self._diff_and_collect_events(snap, is_init=is_init)
            all_new = list(extra_events or []) + diff_evs
            for ev in all_new:
                ev["seq"] = seq
                self.events_feed.appendleft(ev)
            ui_mt = _get_ui_mtime()
            snap["seq"] = seq
            snap["ui_mtime"] = ui_mt
            snap["events_feed"] = list(self.events_feed)
            self._cached_snapshot = snap
            self._cached_slim_snapshot = _slim_snapshot(snap)
            self.last_changed_server = changed_server
            self.latest_event = all_new[-1] if all_new else None
            self.state_cond.notify_all()
            return snap

    def get_snapshot_with_seq(self, *, slim: bool = False) -> tuple[int, dict[str, Any]]:
        self.check_and_reload_local_files()
        with self.state_cond:
            if self._cached_snapshot is None:
                self._rebuild_and_publish("init", is_init=True)
            assert self._cached_snapshot is not None
            if slim and self._cached_slim_snapshot is not None:
                return self.state_seq, self._cached_slim_snapshot
            return self.state_seq, self._cached_snapshot

    def get_cell_detail(self, global_id: str) -> dict[str, Any] | None:
        seq, snap = self.get_snapshot_with_seq(slim=False)
        cell = next((c for c in (snap.get("schedule") or []) if c.get("global_cell_id") == global_id), None)
        if cell is None:
            return None
        sid = cell.get("server_id") or ""
        ck = cell.get("cell_key") or ""
        tel = ((snap.get("servers") or {}).get(sid) or {}).get("telemetry") or {}
        raw = (tel.get("cells") or {}).get(ck) or {"exists": False}
        return {
            "cell": cell,
            "raw": raw,
            "seq": seq,
            "ui_mtime": _get_ui_mtime(),
        }

    def get_snapshot(self, *, slim: bool = False) -> dict[str, Any]:
        seq, snap = self.get_snapshot_with_seq(slim=slim)
        return {**snap, "seq": seq, "ui_mtime": _get_ui_mtime()}


ColabPoller = ClusterPoller


# ---------------------------------------------------------------------------
# HTTP Server, SSE Push Stream, Graceful Port Replacement & CLI Entrypoint
# ---------------------------------------------------------------------------

_CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
    TimeoutError,
    socket.timeout,
)


class MonitorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        _, exc_val, _ = sys.exc_info()
        if isinstance(exc_val, _CLIENT_DISCONNECT_ERRORS):
            return
        super().handle_error(request, client_address)


class MonitorHandler(BaseHTTPRequestHandler):
    poller: ClusterPoller
    stop_event: threading.Event

    def log_message(self, format: str, *args: Any) -> None:
        return

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True

    def finish(self) -> None:
        try:
            super().finish()
        except OSError:
            pass

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        accept_enc = self.headers.get("Accept-Encoding", "")
        use_gzip = len(raw_body) > 4096 and "gzip" in accept_enc.lower()
        body = gzip.compress(raw_body, compresslevel=5) if use_gzip else raw_body
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            if use_gzip:
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _handle_sse(self) -> None:
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            with self.poller.state_cond:
                last_seq = self.poller.state_seq
            hello_payload = json.dumps({"seq": last_seq, "ui_mtime": _get_ui_mtime()}, ensure_ascii=False)
            self.wfile.write(f"event: hello\ndata: {hello_payload}\n\n".encode("utf-8"))
            self.wfile.flush()

            stop_ev = getattr(self, "stop_event", None)
            while not (stop_ev is not None and stop_ev.is_set()):
                with self.poller.state_cond:
                    if self.poller.state_seq == last_seq and not (stop_ev is not None and stop_ev.is_set()):
                        self.poller.state_cond.wait(timeout=15.0)
                    cur_seq = self.poller.state_seq
                    changed_srv = self.poller.last_changed_server
                    latest_ev = self.poller.latest_event

                if stop_ev is not None and stop_ev.is_set():
                    break
                if cur_seq != last_seq:
                    last_seq = cur_seq
                    upd_payload = json.dumps({
                        "seq": cur_seq,
                        "changed_server": changed_srv,
                        "ui_mtime": _get_ui_mtime(),
                        "latest_event": latest_ev,
                    }, ensure_ascii=False)
                    self.wfile.write(f"event: state_updated\ndata: {upd_payload}\n\n".encode("utf-8"))
                else:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html_body = _load_dashboard_html().encode("utf-8")
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Content-Length", str(len(html_body)))
                self.end_headers()
                self.wfile.write(html_body)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        elif parsed.path == "/api/health":
            self._send_json({
                "ok": True,
                "service": "fgpo-monitor-ui",
                "pid": os.getpid(),
                "seq": self.poller.state_seq,
                "ui_mtime": _get_ui_mtime(),
            })
        elif parsed.path == "/api/state":
            qs = urllib.parse.parse_qs(parsed.query)
            since_raw = (qs.get("since_seq") or [None])[0]
            slim_flag = (qs.get("slim") or ["0"])[0].lower() in ("1", "true", "yes")
            cur_seq, snap = self.poller.get_snapshot_with_seq(slim=slim_flag)
            ui_mt = _get_ui_mtime()
            if since_raw is not None:
                try:
                    if int(since_raw) == cur_seq:
                        self._send_json({
                            "unchanged": True,
                            "seq": cur_seq,
                            "ui_mtime": ui_mt,
                            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        })
                        return
                except ValueError:
                    pass
            self._send_json({**snap, "seq": cur_seq, "ui_mtime": ui_mt})
        elif parsed.path == "/api/cell":
            qs = urllib.parse.parse_qs(parsed.query)
            cid = (qs.get("id") or [None])[0]
            if not cid:
                self._send_json({"error": "missing_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            detail = self.poller.get_cell_detail(cid)
            if detail is None:
                self._send_json({"error": "cell_not_found", "id": cid}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
        elif parsed.path == "/api/events":
            self._handle_sse()
        else:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/shutdown":
            client_ip = self.client_address[0] if self.client_address else ""
            if client_ip not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
                self._send_json({"ok": False, "error": "forbidden"}, status=HTTPStatus.FORBIDDEN)
                return
            self._send_json({"ok": True, "pid": os.getpid()})
            stop_ev = getattr(self, "stop_event", None)
            if stop_ev is not None:
                stop_ev.set()
            self.poller.stop()
            srv_ref = self.server

            def _shutdown_and_close() -> None:
                try:
                    srv_ref.shutdown()
                finally:
                    try:
                        srv_ref.server_close()
                    except Exception:
                        pass

            threading.Thread(target=_shutdown_and_close, daemon=True).start()
        elif parsed.path == "/api/refresh":
            qs = urllib.parse.parse_qs(parsed.query)
            srv = (qs.get("server") or ["all"])[0]
            if srv not in ("all", "colab", "host_a", "host_b"):
                self._send_json({"error": "invalid_server"}, status=HTTPStatus.BAD_REQUEST)
                return
            async_flag = (qs.get("async") or ["0"])[0].lower() in ("1", "true", "yes")
            if async_flag:
                seq = self.poller.trigger_async_refresh(target_server=srv, force_meta=(srv in ("all", "colab")))
                self._send_json({"accepted": True, "target": srv, "seq": seq})
            else:
                self._send_json(self.poller.poll_once(target_server=srv, force_meta=(srv in ("all", "colab"))))
        else:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)


def _background_loop(poller: ClusterPoller, interval: int, stop_event: threading.Event) -> None:
    poller.start_background_threads(interval=interval, stop_event=stop_event)
    stop_event.wait()


def _bind_or_replace_server(
    host: str,
    port: int,
    handler_cls: type[BaseHTTPRequestHandler],
    *,
    no_replace: bool = False,
) -> MonitorHTTPServer:
    try:
        return MonitorHTTPServer((host, port), handler_cls)
    except OSError as exc:
        if exc.errno not in (errno.EADDRINUSE, 48, 98):
            raise

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    url_base = f"http://[{probe_host}]:{port}" if ":" in probe_host else f"http://{probe_host}:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    existing_health: dict[str, Any] | None = None
    try:
        req = urllib.request.Request(f"{url_base}/api/health", method="GET")
        with opener.open(req, timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                if isinstance(data, dict) and data.get("service") == "fgpo-monitor-ui":
                    existing_health = data
    except Exception:
        existing_health = None

    if existing_health is not None and not no_replace and probe_host in ("127.0.0.1", "::1", "localhost"):
        try:
            sreq = urllib.request.Request(f"{url_base}/api/shutdown", data=b"", method="POST")
            with opener.open(sreq, timeout=2.0) as _:
                pass
        except Exception:
            pass
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            time.sleep(0.1)
            try:
                srv = MonitorHTTPServer((host, port), handler_cls)
                print(f"Replaced existing fgpo-monitor-ui instance (pid={existing_health.get('pid')}) on {host}:{port}")
                return srv
            except OSError as bind_err:
                if bind_err.errno not in (errno.EADDRINUSE, 48, 98):
                    raise

    if existing_health is not None and no_replace:
        reason = f"already running fgpo-monitor-ui (pid={existing_health.get('pid')}, --no-replace set)"
    elif existing_health is not None:
        reason = f"existing fgpo-monitor-ui (pid={existing_health.get('pid')}) did not release port in time"
    else:
        reason = "held by a pre-upgrade monitor or another process"
    print(f"Port {host}:{port} is unavailable ({reason}).", file=sys.stderr)
    print("Stop the existing monitor with: pkill -f colab_monitor_ui.py", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_ssh_doc = str(repo_root / "SSH_Document.md")
    default_sched_csv = str(repo_root / "outputs" / "exp-plan-20261118" / "compute_schedule_current.csv")

    parser = argparse.ArgumentParser(description="Unified live monitor for F-GPO experiments across Colab (archived/live), Host A, and Host B.")
    # Preserved backward-compatible CLI flags (Colab defaults to archived read-only mode unless --session is specified)
    parser.add_argument("--session", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--log-path", default="auto")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--suite", default="comparison")
    parser.add_argument("--benchmark", default="alfworld")
    parser.add_argument("--method", default="fgpo")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=int, default=20, help="Auto-poll interval in seconds (0 = manual only)")
    parser.add_argument("--no-replace", action="store_true", help="Do not replace an existing fgpo-monitor-ui instance on the same port")
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

    class _Handler(MonitorHandler):
        pass

    _Handler.poller = poller
    _Handler.stop_event = stop_event

    # Bind (or gracefully replace existing monitor) BEFORE firing remote probes
    server = _bind_or_replace_server(args.host, args.port, _Handler, no_replace=args.no_replace)
    poller.start_background_threads(interval=args.interval, stop_event=stop_event)

    enabled_names = [
        f"{k}(archived)" if (k == "colab" and poller.colab_archive_only) else k
        for k, v in poller.enabled.items() if v
    ]
    print(
        f"F-GPO Unified Cluster Monitor UI running at http://{args.host}:{args.port} "
        f"(servers={','.join(enabled_names)}, interval={args.interval}s)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        poller.stop()
        server.server_close()


if __name__ == "__main__":
    main()
