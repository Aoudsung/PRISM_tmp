#!/usr/bin/env python3
"""Read-only comparison across r7 (fgpo-r7-token-mean-v1), r6, r5, and V4."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("rb") as handle:
        for raw in handle:
            if raw.endswith(b"\n") and raw.strip():
                rows.append(json.loads(raw))
    return rows


def _ent_per_tok(row: dict) -> float | None:
    metrics = row.get("metrics") or {}
    mixed = int(metrics.get("mixed_groups") or 0)
    ent = metrics.get("entropy")
    if mixed <= 0 or ent is None:
        return None
    if metrics.get("entropy_normalization") == "token":
        denom = int(metrics.get("minibatch_count") or metrics.get("steps") or mixed)
        return float(ent) / denom if denom > 0 else None
    eval_tokens = int(metrics.get("evaluated_response_tokens") or 0)
    if eval_tokens <= 0:
        return None
    denom = float(metrics.get("group_trajectory_denominator") or 8.0)
    return float(ent) * denom / eval_tokens


def _kl_per_batch(row: dict) -> float | None:
    metrics = row.get("metrics") or {}
    mixed = int(metrics.get("mixed_groups") or 0)
    kl = metrics.get("reference_kl")
    if mixed <= 0 or kl is None:
        return None
    denom = int(metrics.get("minibatch_count") or metrics.get("steps") or mixed)
    return float(kl) / denom if denom > 0 else None


def _mean_pending_weight(row: dict) -> float | None:
    diag = row.get("diagnostics") or {}
    groups = diag.get("credit_groups") or {}
    vals = [
        float(item["mean_pending_weight"])
        for item in groups.values()
        if not item.get("homogeneous_group") and item.get("mean_pending_weight") is not None
    ]
    return sum(vals) / len(vals) if vals else None


def _row_stats(row: dict) -> dict:
    diag = row.get("diagnostics") or {}
    costs = row.get("costs") or {}
    metrics = row.get("metrics") or {}
    credit_groups = diag.get("credit_groups") or {}
    trajs = int(row.get("trajectories") or 0)
    wins = int(row.get("successes") or 0)
    syn = diag.get("syntax_valid_fraction")
    adm = diag.get("admissible_valid_fraction")
    tot_steps = int(diag.get("total_steps") or 0)
    adm_steps = int(diag.get("admissible_valid_steps") or 0)
    adm_obs = diag.get("admissible_observed_steps")
    if adm_obs is None and diag.get("admissible_valid_steps") is not None:
        adm_obs = round(adm_steps / float(adm)) if adm and float(adm) > 0 else tot_steps
    norms = [float(v) for v in (metrics.get("batch_gradient_norms") or ()) if v is not None]
    mean_norm = sum(norms) / len(norms) if norms else None
    return {
        "round": int(row["round"]),
        "protocol": row.get("protocol"),
        "weighting": metrics.get("loss_weighting"),
        "wins": wins,
        "trajs": trajs,
        "sr": (100.0 * wins / trajs) if trajs else 0.0,
        "mixed": int(diag.get("mixed_groups") or 0),
        "all_one": int(diag.get("all_one_groups") or 0),
        "syn_steps": int(diag.get("syntax_valid_steps") or 0),
        "tot_steps": tot_steps,
        "syn_pct": 100.0 * float(syn) if syn is not None else None,
        "adm_steps": adm_steps,
        "adm_obs": int(adm_obs or 0),
        "adm_pct": 100.0 * float(adm) if adm is not None else None,
        "unterm": int(diag.get("unterminated_response_steps") or 0),
        "ent_tok": _ent_per_tok(row),
        "kl_batch": _kl_per_batch(row),
        "omega": _mean_pending_weight(row),
        "opt_steps": int(row.get("optimizer_steps") or 0),
        "opt_in_round": int(diag.get("optimizer_steps_in_round") or 0),
        "mean_grad_norm": mean_norm,
        "clip_frac": float(metrics["ppo_clip_fraction"]) if metrics.get("ppo_clip_fraction") is not None else None,
        "cum_wall_s": float(costs.get("wall_seconds") or 0.0),
    }


def _round_wall_map(rows: list[dict]) -> dict[int, float]:
    ordered = sorted(rows, key=lambda r: int(r["round"]))
    out: dict[int, float] = {}
    prev = 0.0
    for r in ordered:
        cum = float((r.get("costs") or {}).get("wall_seconds") or 0.0)
        delta = (cum - prev) if cum >= prev else cum
        out[int(r["round"])] = delta
        prev = cum
    return out


def _fmt(val: float | None, spec: str = ".2f", suffix: str = "") -> str:
    return "—" if val is None else f"{val:{spec}}{suffix}"


def _window_summary(rows: list[dict], wall_by_r: dict[int, float], lo: int, hi: int) -> dict | None:
    sub = [r for r in rows if lo <= int(r["round"]) <= hi]
    if not sub:
        return None
    stats = [_row_stats(r) for r in sub]
    wins = sum(s["wins"] for s in stats)
    trajs = sum(s["trajs"] for s in stats)
    tot_steps = sum(s["tot_steps"] for s in stats)
    syn_steps = sum(s["syn_steps"] for s in stats)
    adm_obs = sum(s["adm_obs"] for s in stats)
    adm_steps = sum(s["adm_steps"] for s in stats)
    mixed = sum(s["mixed"] for s in stats)
    unterm = sum(s["unterm"] for s in stats)
    opt_steps = sum(s["opt_in_round"] for s in stats)
    wall_h = sum(wall_by_r.get(s["round"], 0.0) for s in stats) / 3600.0
    return {
        "rounds": f"R{lo}-R{sub[-1]['round']}",
        "wins": wins,
        "trajs": trajs,
        "sr": 100.0 * wins / trajs if trajs else 0.0,
        "mixed": mixed,
        "opt_steps": opt_steps,
        "syn_pct": 100.0 * syn_steps / tot_steps if tot_steps else None,
        "adm_pct": 100.0 * adm_steps / adm_obs if adm_obs else None,
        "unterm": unterm,
        "wall_h": wall_h,
    }


def _live_costs_progress(cell: Path) -> str:
    costs = cell / "costs.jsonl"
    if not costs.exists():
        return "not started"
    step_counts: collections.Counter[str] = collections.Counter()
    group_sec: dict[str, float] = {}
    max_round = -1
    for line in costs.open(encoding="utf-8"):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        role = ev.get("role")
        det = ev.get("details") or {}
        gid = det.get("group_id")
        if role == "rollout.step_native" and gid:
            step_counts[gid] += 1
            r = int(gid.split(":")[2])
            if r > max_round:
                max_round = r
        elif role == "rollout.group" and gid:
            group_sec[gid] = float(ev.get("seconds") or 0.0)
    if not step_counts:
        return "costs initialized, 0 env steps"
    cur_gids = [g for g in step_counts if int(g.split(":")[2]) == max_round]
    cur_gids.sort(key=lambda g: int(g.split(":")[-1]))
    done = [g for g in cur_gids if g in group_sec]
    avg_s = (sum(group_sec[g] for g in done) / len(done)) if done else 0.0
    return f"R{max_round}: {len(done)}/{len(cur_gids)} groups finished (avg {avg_s:.1f}s/group)"


def _summarize_arm(label: str, cell: Path, jsonl_override: Path | None = None) -> None:
    rows = _read_jsonl(jsonl_override if jsonl_override is not None else (cell / "training.jsonl"))
    print(f"  [{label}] live: {_live_costs_progress(cell) if jsonl_override is None else 'snapshot'}")
    if not rows:
        print(f"  [{label}] 0 completed rounds")
        return
    wall = _round_wall_map(rows)
    w_all = _window_summary(rows, wall, 0, 999)
    proto = rows[-1].get("protocol", "?")
    wgt = (rows[-1].get("metrics") or {}).get("loss_weighting", "?")
    print(
        f"  [{label}] proto={proto} weighting={wgt} {w_all['rounds']}: "
        f"cum={w_all['wins']}/{w_all['trajs']} ({w_all['sr']:.2f}%), mix={w_all['mixed']}, "
        f"adamw={w_all['opt_steps']}, syn={_fmt(w_all['syn_pct'], '.1f', '%')}, "
        f"adm={_fmt(w_all['adm_pct'], '.1f', '%')}, unterm={w_all['unterm']}, wall={w_all['wall_h']:.2f}h"
    )
    for r in rows[-5:]:
        s = _row_stats(r)
        print(
            f"    R{s['round']:02d}: SR={s['wins']:3d}/{s['trajs']} ({s['sr']:5.2f}%) mix={s['mixed']:2d} "
            f"adamw=+{s['opt_in_round']}({s['opt_steps']}) ent={_fmt(s['ent_tok'], '.3f')} "
            f"kl={_fmt(s['kl_batch'], '.5f')} syn={_fmt(s['syn_pct'], '5.1f', '%')} "
            f"adm={_fmt(s['adm_pct'], '5.1f', '%')} unt={s['unterm']} "
            f"gnorm={_fmt(s['mean_grad_norm'], '.2f')} clip={_fmt(s['clip_frac'], '.3f')} "
            f"wall={wall.get(s['round'], 0.0):.0f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", nargs=2, metavar=("LABEL", "PATH"), default=[])
    args = parser.parse_args()
    for label, path_str in args.cell:
        p = Path(path_str)
        if p.is_file():
            _summarize_arm(label, p.parent, jsonl_override=p)
        else:
            _summarize_arm(label, p)


if __name__ == "__main__":
    main()
