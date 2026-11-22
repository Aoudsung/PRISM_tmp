#!/usr/bin/env python3
"""Read-only paired comparison of r5 (fgpo-r5-symmetric-v1) vs. V4 (fgpo-r3-published-v4)."""

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


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ent_per_tok(row: dict) -> float | None:
    metrics = row.get("metrics") or {}
    mixed = int(metrics.get("mixed_groups") or 0)
    ent = metrics.get("entropy")
    if mixed <= 0 or ent is None:
        return None
    if metrics.get("entropy_normalization") == "token":
        denom = int(metrics.get("minibatch_count") or mixed)
        return float(ent) / denom if denom > 0 else None
    eval_tokens = int(metrics.get("evaluated_response_tokens") or 0)
    if eval_tokens <= 0:
        return None
    denom = float(metrics.get("group_trajectory_denominator") or 8.0)
    return float(ent) * denom / eval_tokens


def _mean_pending_weight(row: dict) -> float | None:
    diag = row.get("diagnostics") or {}
    groups = diag.get("credit_groups") or {}
    vals = []
    for item in groups.values():
        if not item.get("homogeneous_group") and item.get("mean_pending_weight") is not None:
            vals.append(float(item["mean_pending_weight"]))
    return sum(vals) / len(vals) if vals else None


def _row_stats(row: dict) -> dict:
    diag = row.get("diagnostics") or {}
    costs = row.get("costs") or {}
    credit_groups = diag.get("credit_groups") or {}
    mixed_gids = {
        gid for gid, info in credit_groups.items() if not info.get("homogeneous_group", True)
    }
    trajs = int(row.get("trajectories") or 0)
    wins = int(row.get("successes") or 0)
    syn = diag.get("syntax_valid_fraction")
    adm = diag.get("admissible_valid_fraction")
    tot_steps = int(diag.get("total_steps") or 0)
    adm_steps = int(diag.get("admissible_valid_steps") or 0)
    adm_obs = diag.get("admissible_observed_steps")
    if adm_obs is None and diag.get("admissible_valid_steps") is not None:
        adm_obs = round(adm_steps / float(adm)) if adm and float(adm) > 0 else tot_steps
    return {
        "round": int(row["round"]),
        "wins": wins,
        "trajs": trajs,
        "sr": (100.0 * wins / trajs) if trajs else 0.0,
        "mixed": int(diag.get("mixed_groups") or 0),
        "all_one": int(diag.get("all_one_groups") or 0),
        "total_groups": int(diag.get("total_groups") or len(credit_groups) or 16),
        "mixed_gids": mixed_gids,
        "syn_steps": int(diag.get("syntax_valid_steps") or 0),
        "tot_steps": tot_steps,
        "syn_pct": 100.0 * float(syn) if syn is not None else None,
        "adm_steps": adm_steps,
        "adm_obs": int(adm_obs or 0),
        "adm_pct": 100.0 * float(adm) if adm is not None else None,
        "unterm": int(diag.get("unterminated_response_steps") or 0),
        "ent_tok": _ent_per_tok(row),
        "omega": _mean_pending_weight(row),
        "gated": int(diag.get("positive_peer_gated_steps") or 0),
        "opt_steps": int(row.get("optimizer_steps") or 0),
        "opt_in_round": int(diag.get("optimizer_steps_in_round") or 0),
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
    wall_h = sum(wall_by_r.get(s["round"], 0.0) for s in stats) / 3600.0
    ents = [s["ent_tok"] for s in stats if s["ent_tok"] is not None]
    omegas = [s["omega"] for s in stats if s["omega"] is not None]
    return {
        "rounds": f"R{lo}-R{sub[-1]['round']}",
        "wins": wins,
        "trajs": trajs,
        "sr": 100.0 * wins / trajs if trajs else 0.0,
        "mixed": mixed,
        "syn_pct": 100.0 * syn_steps / tot_steps if tot_steps else None,
        "adm_pct": 100.0 * adm_steps / adm_obs if adm_obs else None,
        "unterm": unterm,
        "mean_ent_tok": sum(ents) / len(ents) if ents else None,
        "mean_omega": sum(omegas) / len(omegas) if omegas else None,
        "wall_h": wall_h,
    }


def _live_costs_progress(cell: Path) -> str:
    costs = cell / "costs.jsonl"
    if not costs.exists():
        return "not started"
    step_counts: collections.Counter[str] = collections.Counter()
    slot_steps: collections.Counter[tuple[str, int | None]] = collections.Counter()
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
            slot_steps[(gid, det.get("slot"))] += 1
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
    last_gid = cur_gids[-1]
    last_slots = [slot_steps[(last_gid, s)] for s in range(8)]
    return (
        f"R{max_round}: {len(done)}/{len(cur_gids)} groups finished "
        f"(avg {avg_s:.1f}s/group), latest={last_gid} slots={last_slots}"
    )


def _print_eval(cell: Path, split: str) -> None:
    summary = _read_json(cell / "cell_summary.json")
    if summary and split in (summary.get("evaluation") or {}):
        ev = summary["evaluation"][split]
        extra = ""
        if ev.get("Score") is not None:
            extra += f" Score={ev['Score']:.2f}"
        if ev.get("task_types"):
            cats = ", ".join(
                f"{k}={v['SR']:.1f}%" for k, v in ev["task_types"].items() if v.get("SR") is not None
            )
            extra += f" | types: {cats}"
        if ev.get("recipe_depths"):
            depths = ", ".join(
                f"d{k}={v['success_count']}/{v['official_task_count']}({v['SR']:.1f}%)"
                for k, v in sorted(ev["recipe_depths"].items())
                if v.get("SR") is not None
            )
            extra += f" | depths: {depths}"
        print(
            f"  OFFICIAL EVAL [{split}]: SR={ev['success_count']}/{ev['official_task_count']} "
            f"({ev['SR']:.2f}%){extra}"
        )
        return
    eval_rows = _read_jsonl(cell / f"evaluation-{split}.jsonl")
    if eval_rows:
        wins = sum(1 for r in eval_rows if r.get("success"))
        scores = [float(r["score"]) * 100.0 for r in eval_rows if r.get("score") is not None]
        score_str = f" Score={sum(scores)/len(scores):.2f}" if scores else ""
        print(f"  PARTIAL EVAL [{split}]: {wins}/{len(eval_rows)} ({100.0*wins/len(eval_rows):.2f}%){score_str}")
    else:
        print(f"  OFFICIAL EVAL [{split}]: pending")


def report_alfworld(r5_out: Path, v4_ref: Path) -> None:
    print("=== 1. ALFWorld 1.5B seed 0 (Paired Seeds, Same 512 Token Cap & Prompt Structure) ===")
    v4_rows = _read_jsonl(v4_ref / "alfworld_seed0_v4_training.jsonl")
    r5_cell = r5_out / "comparison" / "alfworld" / "fgpo" / "seed-0"
    r5_rows = _read_jsonl(r5_cell / "training.jsonl")
    print(f"  Live status: {_live_costs_progress(r5_cell)}")
    v4_by_r = {int(r["round"]): r for r in v4_rows}
    r5_by_r = {int(r["round"]): r for r in r5_rows}
    v4_wall = _round_wall_map(v4_rows)
    r5_wall = _round_wall_map(r5_rows)

    if r5_rows:
        print("  Round | r5 SR (w/128) | V4 SR (w/128) | Mix r5/V4 (+gain/-loss) | Syn% r5/V4 | Adm% r5/V4 | Unterm r5/V4 | Ent/tok r5/V4 | Omega(r5) | Gate(V4) | Wall(s) r5/V4")
        for r in sorted(r5_by_r):
            s5 = _row_stats(r5_by_r[r])
            if r in v4_by_r:
                s4 = _row_stats(v4_by_r[r])
                gain = len(s5["mixed_gids"] - s4["mixed_gids"])
                loss = len(s4["mixed_gids"] - s5["mixed_gids"])
                v4_sr = f"{s4['sr']:5.2f}% ({s4['wins']:3d})"
                mix_str = f"{s5['mixed']:2d}/{s4['mixed']:2d} (+{gain}/-{loss})"
                syn_str = f"{_fmt(s5['syn_pct'], '5.1f')}/{_fmt(s4['syn_pct'], '5.1f')}"
                adm_str = f"{_fmt(s5['adm_pct'], '5.1f')}/{_fmt(s4['adm_pct'], '5.1f')}"
                unt_str = f"{s5['unterm']:4d}/{s4['unterm']:4d}"
                ent_str = f"{_fmt(s5['ent_tok'], '.3f')}/{_fmt(s4['ent_tok'], '.3f')}"
                gate_str = f"{s4['gated']:4d}"
                wall_str = f"{r5_wall.get(r, 0.0):.0f}/{v4_wall.get(r, 0.0):.0f}"
            else:
                v4_sr = "   —   ( — )"
                mix_str = f"{s5['mixed']:2d}/ —"
                syn_str = f"{_fmt(s5['syn_pct'], '5.1f')}/  —"
                adm_str = f"{_fmt(s5['adm_pct'], '5.1f')}/  —"
                unt_str = f"{s5['unterm']:4d}/  —"
                ent_str = f"{_fmt(s5['ent_tok'], '.3f')}/  —"
                gate_str = "  —"
                wall_str = f"{r5_wall.get(r, 0.0):.0f}/ —"
            print(
                f"  R{r:02d}   | {s5['sr']:5.2f}% ({s5['wins']:3d}) | {v4_sr} | "
                f"{mix_str:21s} | {syn_str:11s} | {adm_str:11s} | {unt_str:9s} | "
                f"{ent_str:11s} | {_fmt(s5['omega'], '.3f'):9s} | {gate_str:8s} | {wall_str}"
            )
    else:
        print("  r5 R0 still sampling (no completed row in training.jsonl yet).")

    for label, lo, hi in (("R0 (Zero-Shot Tag Framing)", 0, 0), ("R0-R5", 0, 5), ("R0-R14 (Full V4 Paired)", 0, 14), ("R0-Latest", 0, 999)):
        w5 = _window_summary(r5_rows, r5_wall, lo, hi)
        w4 = _window_summary(v4_rows, v4_wall, lo, hi)
        if w5 or w4:
            p5 = f"{w5['wins']}/{w5['trajs']} ({w5['sr']:.2f}%) mix={w5['mixed']} syn={_fmt(w5['syn_pct'], '.1f', '%')} adm={_fmt(w5['adm_pct'], '.1f', '%')} unterm={w5['unterm']} h={w5['wall_h']:.2f}" if w5 else "pending"
            p4 = f"{w4['wins']}/{w4['trajs']} ({w4['sr']:.2f}%) mix={w4['mixed']} syn={_fmt(w4['syn_pct'], '.1f', '%')} adm={_fmt(w4['adm_pct'], '.1f', '%')} unterm={w4['unterm']} h={w4['wall_h']:.2f}" if w4 else "—"
            print(f"  Window {label}:")
            print(f"    r5: {p5}")
            print(f"    V4: {p4}")
    _print_eval(r5_cell, "valid_unseen")


def report_other_benchmark(name: str, cell: Path, split: str, note: str, ref_eval: dict | None = None) -> None:
    print(f"\n=== {name} ===")
    print(f"  Note: {note}")
    print(f"  Live status: {_live_costs_progress(cell)}")
    rows = _read_jsonl(cell / "training.jsonl")
    if rows:
        w = _window_summary(rows, _round_wall_map(rows), 0, 999)
        last = _row_stats(rows[-1])
        print(
            f"  Completed {len(rows)} rounds ({w['rounds']}): "
            f"cum_SR={w['wins']}/{w['trajs']} ({w['sr']:.2f}%), mix={w['mixed']}, "
            f"syn={_fmt(w['syn_pct'], '.1f', '%')}, unterm={w['unterm']}, wall={w['wall_h']:.2f}h"
        )
        print(
            f"  Latest R{last['round']:02d}: SR={last['wins']}/{last['trajs']} ({last['sr']:.2f}%), "
            f"mix={last['mixed']}, ent/tok={_fmt(last['ent_tok'], '.3f')}, "
            f"omega={_fmt(last['omega'], '.3f')}, unterm={last['unterm']}, opt_steps={last['opt_steps']}"
        )
    else:
        print("  Waiting in sequential queue (0 completed rounds).")
    _print_eval(cell, split)
    if ref_eval is not None:
        print(
            f"  V4 orig reference ({ref_eval.get('study_protocol', 'v4')}): "
            f"SR={ref_eval['success_count']}/{ref_eval['official_task_count']} ({ref_eval['SR']:.2f}%), "
            f"Score={ref_eval['Score']:.2f}, steps={ref_eval['mean_steps']:.2f}, "
            f"out_tok={ref_eval['mean_output_tokens']:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-out", type=Path, default=Path("/content/outputs/fgpo-r5-1p5b-colab-20261122"))
    parser.add_argument("--v4-ref", type=Path, default=Path("/content/outputs/fgpo-r5-1p5b-colab-20261122/v4_baseline_ref"))
    args = parser.parse_args()

    report_alfworld(args.r5_out, args.v4_ref)
    report_other_benchmark(
        "2. TextCraft 1.5B seed 1 (r5 Package vs. V4-orig 0/1920 R0-R14 & Host B V4-format)",
        args.r5_out / "comparison" / "textcraft" / "fgpo" / "seed-1",
        "val",
        "Controlled diff: r5 raises output_tokens 128->512 (matching ALFWorld/WebShop/BiPACE) and removes inventory example.",
    )
    ws_ref = _read_json(args.v4_ref / "webshop_seed1_v4_eval.json")
    report_other_benchmark(
        "3. WebShop 1.5B seed 1 (100% Pure Algorithmic Comparison: Identical Prompt, Parser, 512 Cap, Tasks)",
        args.r5_out / "comparison" / "webshop" / "fgpo" / "seed-1",
        "test",
        "Zero prompt/budget confounders vs. V4-orig (68.75% / 78.04) and V4-adapted (Host B GPU 2).",
        ref_eval=ws_ref,
    )


if __name__ == "__main__":
    main()
