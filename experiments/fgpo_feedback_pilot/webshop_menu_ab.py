"""Compare the two-line WebShop menu repair on the fixed confirm32 panel.

Select the before or after fgpo source tree with PYTHONPATH. The launcher owns
the isolated copies; the runner never changes either source tree.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
from statistics import fmean


ARMS = ("before", "after")
MENU_REPAIR = (
    (b"self.clickables = available['clickables']",
     b"self.clickables = [label for label in available['clickables'] if label != 'search']"),
    (b"(name == 'click' and argument in self.clickables and argument != 'search'))",
     b"(name == 'click' and argument in self.clickables))"),
)
SAMPLING = {"temperature": 0.4, "top_p": 1.0, "horizon": 15, "output_tokens": 512, "batch_size": 8}


def panel(path):
    manifest = json.loads(path.read_text())
    tasks = manifest["confirmation_tasks"]
    if (manifest["confirmation_seed"] != 2026111804 or [t["index"] for t in tasks] != list(range(32))
            or len({t["task_id"] for t in tasks}) != 32):
        raise ValueError("Use the frozen WebShop confirm32 manifest")
    return tasks


def source_file(arm):
    from fgpo import envs

    path = Path(envs.__file__).resolve()
    content = path.read_bytes()
    column = ARMS.index(arm)
    if any(content.count(lines[column]) != 1 for lines in MENU_REPAIR):
        raise ValueError(f"PYTHONPATH must select the {arm} WebShop envs.py source")
    return path


def verify(args, tasks):
    from fgpo.envs import make_env
    from fgpo.runner import write_json

    config = json.loads(args.config.read_text())
    source = source_file("after")
    task = tasks[0]
    env = make_env(config, "webshop", args.output / "verify-worker", seed=task["environment_seed"])
    try:
        env.reset(task["task_id"], initial_seed=task["environment_seed"])
        menu = env.action_schema["available_actions"]
        if menu != ["search[<your query>]"]:
            raise ValueError(f"Unexpected initial WebShop menu {menu!r}")
        search = "search[shirt]"
        result = env.step_native({"kind": "native_text", "text": search})
        products = sorted(command for command in env.action_schema["available_actions"]
                          if re.fullmatch(r"click\[[a-z0-9]{10}\]", command, re.IGNORECASE))
        if not result.info["valid"] or not products:
            raise ValueError("The native search must be valid and expose a clickable product")
        product = products[0]
        click = env.step_native({"kind": "native_text", "text": product})
        env.reset(task["task_id"], initial_seed=task["environment_seed"])
        invalid = "click[search]"
        wrong = env.step_native({"kind": "native_text", "text": invalid})
        if not click.info["valid"] or wrong.info["valid"]:
            raise ValueError("Native product and invalid clicks had unexpected validity")
    finally:
        env.close()
    receipt = {"source": str(source), "task_id": task["task_id"], "environment_seed": task["environment_seed"],
               "reset_menu": menu, "search": search, "search_valid": result.info["valid"],
               "clickable_products": len(products), "product_click": product, "product_click_valid": click.info["valid"],
               "invalid_click": invalid, "invalid_click_valid": wrong.info["valid"]}
    write_json(args.output / "verify.json", receipt, durable=True)
    print(json.dumps(receipt), flush=True)


class ObserveFirstAction:
    def __init__(self, env, task_id, first):
        self.env, self.task_id, self.first = env, task_id, first

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step_native(self, action):
        result = self.env.step_native(action)
        if self.task_id not in self.first:
            self.first[self.task_id] = {
                "action": action["text"], "valid": bool(result.info["valid"]),
                "click_search": action["text"] == "click[search]",
            }
        return result


def task_path(root, task):
    return root / "episodes" / f"{task['index']:03d}.json"


def task_receipts(root, arm, tasks):
    rows = {}
    for task in tasks:
        path = task_path(root, task)
        if not path.is_file():
            continue
        row = json.loads(path.read_text())
        if row["arm"] != arm or any(row[key] != task[key] for key in ("task_id", "environment_seed", "sampling_seed")):
            raise ValueError(f"Task or seed differs from the frozen panel in {path}")
        rows[task["task_id"]] = row
    return rows


def run(args, tasks):
    from fgpo.appworld_batch import RolloutGroupSpec
    from fgpo.envs import EnvironmentPool
    from fgpo.metrics import webshop_score
    from fgpo.policy import CausalPolicy
    from fgpo.public import HORIZONS, OUTPUT_LIMITS
    from fgpo.records import CostLedger
    from fgpo.rollout import collect_evaluation
    from fgpo.runner import isolate_gpus, write_json
    import torch

    source = source_file(args.arm)
    probe = json.loads((args.output / "verify.json").read_text())
    if args.arm == "after" and probe["source"] != str(source):
        raise ValueError("Use the verified repaired source tree for the after arm")
    if (HORIZONS["webshop"], OUTPUT_LIMITS["webshop"]) != (15, 512):
        raise ValueError("The rollout source must use WebShop horizon 15 and action budget 512")
    config = json.loads(args.config.read_text())
    recipe = {**SAMPLING, "gpu": args.gpu, "model": str(Path(config["model_path"]).resolve()),
              "webshop_data_root": str(Path(config["webshop_data_root"]).resolve()),
              "webshop_python_executable": config["webshop_python_executable"],
              "max_context_tokens": config["max_context_tokens"],
              "cache_initial_capacity": config["cache_initial_capacity"],
              "working_memory_gib": config["working_memory_gib"]}
    output = args.output / args.arm
    receipt_path = output / "run.json"
    identity = {"arm": args.arm, "source": str(source), "recipe": recipe}
    if receipt_path.is_file():
        if json.loads(receipt_path.read_text()) != identity:
            raise ValueError("Resume the arm with its original source and inference settings")
    else:
        write_json(receipt_path, identity, durable=True)
    completed = task_receipts(output, args.arm, tasks)
    if len(completed) == len(tasks):
        return

    logical_gpus, _ = isolate_gpus([args.gpu], config["excluded_gpu_ids"])
    ledger = CostLedger(gpu_ids=logical_gpus)
    policy = CausalPolicy(recipe["model"], gpu_ids=logical_gpus, ledger=ledger, trainable=False, benchmark="webshop",
                          max_context_tokens=recipe["max_context_tokens"], max_new_tokens=SAMPLING["output_tokens"],
                          cache_initial_capacity=recipe["cache_initial_capacity"],
                          working_memory_gib=recipe["working_memory_gib"])
    policy.sampling_temperature = SAMPLING["temperature"]
    pool = EnvironmentPool(config, "webshop", output / "environments", size=SAMPLING["batch_size"])
    first = {}
    task_by_id = {task["task_id"]: task for task in tasks}

    def env_factory(task_id, initial_seed, trajectory_id):
        return ObserveFirstAction(pool(task_id, initial_seed, trajectory_id), task_id, first)

    def publish(spec, episode):
        task = task_by_id[spec.task_id]
        official = episode.official_evaluation
        row = {**task, **asdict(episode), "arm": args.arm, "first_native_action": first.pop(spec.task_id),
               "official_success": official["success"], "native_score": webshop_score(official)}
        write_json(task_path(output, task), row, durable=True)
        completed[spec.task_id] = row
        print(f"{args.arm} completed {len(completed)}/{len(tasks)} task={spec.task_id}", flush=True)

    try:
        with torch.inference_mode():
            for offset in range(0, len(tasks), pool.size):
                specs = tuple(RolloutGroupSpec(f"webshop-menu-ab:{task['index']}", task["task_id"],
                                              task["environment_seed"], (task["sampling_seed"],))
                              for task in tasks[offset:offset + pool.size] if task["task_id"] not in completed)
                if specs:
                    collect_evaluation(policy, env_factory, specs, benchmark="webshop", on_episode_complete=publish)
    finally:
        pool.close()
    if len(completed) != len(tasks):
        raise RuntimeError("The rollout did not complete the 32-task panel")


def arm_metrics(rows):
    values = list(rows.values())
    total = len(values)
    invalid = sum(not row["first_native_action"]["valid"] for row in values)
    click_search = sum(row["first_native_action"]["click_search"] for row in values)
    successes = sum(row["official_success"] for row in values)
    return {"tasks": total, "first_invalid_count": invalid, "first_invalid_percent": 100 * invalid / total,
            "first_click_search_count": click_search, "first_click_search_percent": 100 * click_search / total,
            "full_success_count": successes, "full_success_percent": 100 * successes / total,
            "mean_score": 100 * fmean(row["native_score"] for row in values)}


def compare(args, tasks):
    from fgpo.runner import write_json

    roots = {arm: args.output / arm for arm in ARMS}
    metadata = {arm: json.loads((root / "run.json").read_text()) for arm, root in roots.items()}
    old, new = (Path(metadata[arm]["source"]) for arm in ARMS)
    expected_old = new.read_bytes()
    for before, after in MENU_REPAIR:
        expected_old = expected_old.replace(after, before, 1)
    if old == new or old.read_bytes() != expected_old or metadata["before"]["recipe"] != metadata["after"]["recipe"]:
        raise ValueError("Use isolated source trees that differ only by the menu repair and share one inference recipe")
    rows = {arm: task_receipts(roots[arm], arm, tasks) for arm in ARMS}
    if any(len(values) != len(tasks) for values in rows.values()):
        raise ValueError("The comparison requires both complete 32-task arms")
    arms = {arm: arm_metrics(values) for arm, values in rows.items()}
    changes = []
    for task in tasks:
        before, after = (rows[arm][task["task_id"]] for arm in ARMS)
        changes.append({**task, "before": {key: before[key] for key in ("first_native_action", "official_success", "native_score")},
                        "after": {key: after[key] for key in ("first_native_action", "official_success", "native_score")},
                        "first_invalid_change": int(not after["first_native_action"]["valid"]) - int(not before["first_native_action"]["valid"]),
                        "full_success_change": int(after["official_success"]) - int(before["official_success"]),
                        "score_change": 100 * (after["native_score"] - before["native_score"])})
    delta = {key: arms["after"][key] - arms["before"][key] for key in arms["before"] if key != "tasks"}
    result = {"panel": "confirm32", "recipe": metadata["after"]["recipe"], "arms": arms,
              "score_scale": [0, 100], "percentage_difference_unit": "percentage_points",
              "after_minus_before": delta, "full_success_gains": sum(row["full_success_change"] == 1 for row in changes),
              "full_success_losses": sum(row["full_success_change"] == -1 for row in changes), "tasks": changes}
    write_json(args.output / "comparison.json", result, durable=True)
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}, indent=2), flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, entry in (("verify", verify), ("run", run), ("compare", compare)):
        command = commands.add_parser(name)
        command.set_defaults(entry=entry)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name != "compare":
            command.add_argument("--config", type=Path, required=True)
        if name == "run":
            command.add_argument("--arm", choices=ARMS, required=True)
            command.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.entry(args, panel(args.manifest))


if __name__ == "__main__":
    main()
