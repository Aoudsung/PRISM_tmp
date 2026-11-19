"""Run the WebShop pilot updates and optional zero-demo control on confirm32.

Run with PYTHONPATH pointing to the original V4 WebShop adapter source. Use the
original pilot manifest and samples. Each arm starts a fresh process.
The archive_null arm also needs the directory containing fgpo_verified_feedback/learn.py on PYTHONPATH.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from math import comb
import os
from pathlib import Path
import shutil
import socket
from statistics import fmean
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from types import SimpleNamespace


REPLAY_STEPS = {"ordinary": 9, "proposed": 12, "archive_null": 13}
ARCHIVE_ARMS = ("anchor_v4", "archive_null", "archive_success", "archive_success_history")
METRICS = ("official_success", "native_score", "steps")
BASE_SOURCE = "independent_pretrained_qwen"
BASE_SOURCE_FILE = "base-source.json"
THRESHOLD_FREE_ARM = "threshold_free_success"


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def confirmation_tasks(path):
    manifest = json.loads(path.read_text())
    tasks = manifest["confirmation_tasks"]
    if (manifest["confirmation_seed"] != 2026111804
            or [task["index"] for task in tasks] != list(range(32))
            or len({task["task_id"] for task in tasks}) != 32):
        raise ValueError("Use the frozen WebShop confirmation panel with seed 2026111804")
    return tasks


def prepare_base(args):
    original = json.loads(args.pilot_manifest.read_text())
    args.pilot.mkdir(parents=True, exist_ok=True)
    destination = args.pilot / "manifest.json"
    if destination.is_file():
        if json.loads(destination.read_text())["tasks"] != original["tasks"]:
            raise ValueError("The independent collection must use the original training task rows and seeds")
    else:
        with destination.open("x") as output:
            json.dump(original, output, indent=2)
            output.write("\n")
    source = {"actor_provenance": BASE_SOURCE, "collection_model": str(args.model.resolve())}
    source_path = args.pilot / BASE_SOURCE_FILE
    if source_path.is_file():
        if json.loads(source_path.read_text()) != source:
            raise ValueError("The independent corpus is already assigned to a different model")
    else:
        with source_path.open("x") as output:
            json.dump(source, output, indent=2)
            output.write("\n")


def result_root(args):
    return args.output / "independent-base" if args.independent_base else args.output


def colab_config(args):
    base = json.loads(args.base.read_text())
    resource_keys = (
        "python_executable", "excluded_gpu_ids", "activation_offload",
        "cache_initial_capacity", "representation_token_budget", "representation_batch_size",
        "working_memory_gib", "optimizer_bucket_mib",
        "optimizer_pin_memory", "checkpoint_shard_mib",
    )
    config = {key: base[key] for key in resource_keys}
    batch_size = args.eval_batch_size if args.eval_batch_size is not None else base["environment_pool_size"]
    if batch_size < 1 or args.head_chunk_tokens < 1:
        raise ValueError("Evaluation width and head chunk size must be positive")
    reference = str(args.reference_model.absolute()) if args.reference_model else base["model_path"]
    config.update(
        protocol="fgpo-r3-published-v4", method="fgpo", benchmark="webshop", seed=0, gpu_ids=[0],
        local_files_only=True, checkpoint_backend="local", model_path=reference,
        webshop_python_executable=str(args.webshop_python.absolute()),
        webshop_data_root=str(args.webshop_root.absolute()),
        environment_pool_size=batch_size,
        environment_pool_size_source="colab_base_config" if args.eval_batch_size is None else "command_line",
        head_chunk_tokens=args.head_chunk_tokens,
        horizons={"webshop": 15}, output_tokens=512, max_context_tokens=32768,
        rounds=60, groups=16, trajectories=8, ppo_epochs=1,
        temperature=1., top_p=1., eval_temperature=.4, eval_top_p=1.,
        learning_rate=1e-6, weight_decay=.01, optimizer_betas=[.9, .999], optimizer_eps=1e-8,
        gamma=1., reward_scale=10., loss_normalization="trajectory", ppo_clip_ratio=.2,
        ppo_dual_clip=3., entropy_coef=.001, kl_coef=.01, max_grad_norm=1.,
        credit_radius=.1, credit_prior_weight=1., credit_transfer_max=.5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def credit_path(output, arm):
    filename = "fixed-archive-v4.pt" if arm == "archive_null" else "fixed-pilot-v4.pt"
    return output / "reference-precompute" / filename


def credit_source(pilot, groups, config, arm):
    def version(path):
        info = path.stat()
        return info.st_size, info.st_mtime_ns

    reference = Path(config["model_path"]).resolve()
    keys = ("protocol", "method", "benchmark", "horizons", "output_tokens", "max_context_tokens",
            "cache_initial_capacity", "head_chunk_tokens", "representation_token_budget", "representation_batch_size",
            "reward_scale", "credit_radius", "credit_prior_weight", "credit_transfer_max")
    return {
        "corpus": str(pilot.resolve()), "manifest": version(pilot / "manifest.json"), "arm": arm,
        "samples": [version(pilot / "samples" / f"{task['index']:03d}.pt") for task in groups],
        "reference": str(reference), "reference_index": version(reference / "model.safetensors.index.json"),
        "config": {key: config.get(key) for key in keys}, "attention": "sdpa", "dtype": "bfloat16",
        "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
    }


def read_precomputed(path, source):
    import torch

    if path.is_file():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache["source"] == source:
            return cache
        print(f"reference cache source differs at {path}; compute this arm with the normal recipe", flush=True)
    return None


def precompute(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    from fgpo.policy import CausalPolicy
    from fgpo.protocol import require_protocol, resolve_method_config
    from fgpo.records import CostLedger
    from fgpo.runner import _file_lock, isolate_gpus
    from update import prepare_credit, select_group

    config = json.loads(args.config.read_text())
    config.update(method="fgpo", benchmark="webshop", seed=0)
    config = resolve_method_config(config)
    require_protocol(config)
    source = json.loads((args.pilot / BASE_SOURCE_FILE).read_text())
    reference = Path(config["model_path"]).resolve()
    if (source["actor_provenance"] != BASE_SOURCE or Path(source["collection_model"]).resolve() != reference
            or config.get("checkpoint_backend", "local") != "local"):
        raise ValueError("Precomputation requires the independent base Qwen corpus and its fixed reference")
    groups = json.loads((args.pilot / "manifest.json").read_text())["tasks"]
    if [task["index"] for task in groups] != list(range(32)):
        raise ValueError("Reference precomputation requires all 32 original task rows in order")
    if not 0 < args.allocator_budget_gib <= 19.75:
        raise ValueError("Use a positive precomputation allocator budget of at most 19.75 GiB")

    root = args.output / "independent-base"
    with _file_lock(root / ".reference-precompute.lock", nonblocking=True):
        jobs = []
        for arm in dict.fromkeys(args.arm or REPLAY_STEPS):
            path = credit_path(root / arm, arm)
            identity = credit_source(args.pilot, groups, config, arm)
            cache = read_precomputed(path, identity)
            if cache is not None:
                print(json.dumps({**cache["precompute"], "already_published": True}), flush=True)
            else:
                jobs.append((arm, path, identity))
        if not jobs:
            return

        logical_ids, _ = isolate_gpus([args.gpu], config["excluded_gpu_ids"])
        gpu = logical_ids[0]
        fraction = args.allocator_budget_gib * 2**30 / torch.cuda.get_device_properties(gpu).total_memory
        torch.cuda.set_per_process_memory_fraction(fraction, device=gpu)
        torch.use_deterministic_algorithms(True)
        ledger = CostLedger(gpu_ids=logical_ids)
        policy = CausalPolicy(str(reference), gpu_ids=logical_ids, ledger=ledger, trainable=False,
                              benchmark="webshop", max_context_tokens=int(config.get("max_context_tokens", 32768)),
                              max_new_tokens=config["output_tokens"],
                              cache_initial_capacity=int(config.get("cache_initial_capacity", 1024)),
                              representation_token_budget=int(config.get("representation_token_budget", 8192)),
                              representation_batch_size=int(config.get("representation_batch_size", 8)),
                              working_memory_gib=float(config.get("working_memory_gib", 6.)))
        for arm, path, identity in jobs:
            torch.cuda.reset_peak_memory_stats(gpu)
            started = monotonic()
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"reference precompute start arm={arm}", flush=True)
            with TemporaryDirectory(dir=path.parent) as staging:
                staged = Path(staging) / path.name
                if arm == "archive_null":
                    from learn import load_groups, ordinary_credit

                    scheduled = load_groups(args.pilot, groups, corpus="archive")
                    cache, _, _ = ordinary_credit(SimpleNamespace(config=config, policy=policy), scheduled, staged, "v4")
                    del scheduled
                else:
                    selected = [select_group(args.pilot, task, arm)[1] for task in groups]
                    cache = prepare_credit(policy, config, selected, reference_loaded=True)
                    del selected
                policy.synchronize()
                cache["source"] = identity
                cache["precompute"] = {
                    "arm": arm, "cache": str(path.resolve()), "prepared_seconds": cache["prepared_seconds"],
                    "arm_seconds": monotonic() - started, "allocator_budget_gib": args.allocator_budget_gib,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(gpu) / 2**30,
                    "corpus": identity["corpus"], "reference": identity["reference"],
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                }
                torch.save(cache, staged)
                staged.replace(path)
            print(json.dumps(cache["precompute"]), flush=True)
            del cache
            ledger.begin_group()
            torch.cuda.empty_cache()


def same_trial_receipt(args, arm):
    path = result_root(args) / arm / "receipt.json"
    receipt = json.loads(path.read_text())
    expected = {
        "arm": arm, "actor_provenance": BASE_SOURCE, "exit_code": 0,
        "pilot_manifest": str((args.pilot / "manifest.json").resolve()),
        "manifest": str(args.manifest.resolve()), "config": str(args.config.resolve()),
        "optimizer_initial_step": 0, "physical_gpu_ids": args.gpus,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Expected a completed arm from this independent trial at {path}")
    return receipt


def observed_schedule(path):
    return [(row["index"], row["group_id"], row["tick"], row["marked_successes"])
            for row in read_jsonl(path) if row["tick"] != "skip"]


def matched_threshold_free_schedule(args, config, scheduled, cache_path):
    from learn import marked_responses, mixed

    control = same_trial_receipt(args, "archive_null")
    reference = Path(config["model_path"]).resolve()
    if (Path(control["actor_anchor"]).resolve() != reference or Path(control["reference"]).resolve() != reference
            or not control["training_deterministic_algorithms"]
            or not control["training"]["fixed_reference_reused"]
            or Path(control["training"]["fixed_reference_cache"]).resolve() != cache_path.resolve()):
        raise ValueError("The completed zero-demo receipt must use the same fixed cache and base Qwen")
    schedule = []
    for task, group, eligible, note in scheduled:
        if eligible:
            _, marked = marked_responses(group, note)
            if marked:
                schedule.append((task["index"], group.group_id, "supervised", [step.step_id for step, _ in marked]))
        elif mixed(group):
            schedule.append((task["index"], group.group_id, "ppo", []))
    supervised = [(index, steps) for index, _, kind, steps in schedule if kind == "supervised"]
    if (len(schedule) != 12 or [index for index, _ in supervised] != [1, 15, 23]
            or sum(len(steps) for _, steps in supervised) != 4
            or schedule != observed_schedule(result_root(args) / "archive_null" / "groups.jsonl")):
        raise ValueError("The frozen corpus must reproduce the completed zero-demo PPO and demonstration schedule")
    return schedule


def train_arm(engine, args, groups):
    active = args.arm == THRESHOLD_FREE_ARM
    cache_arm = "archive_null" if active else args.arm
    path = credit_path(result_root(args) / cache_arm, cache_arm) if active else credit_path(engine.output_dir, args.arm)
    credit = None
    if args.independent_base and path.is_file():
        credit = read_precomputed(path, credit_source(args.pilot, groups, engine.config, cache_arm))
    credit_receipt = credit["precompute"] if credit is not None else None
    if active or args.arm == "archive_null":
        from learn import load_groups, update

        if active and credit is None:
            raise FileNotFoundError("The threshold-free supplement requires the matching zero-demo precomputed cache")
        scheduled = load_groups(args.pilot, groups, corpus="archive")
        cache_path = path if credit is not None else engine.output_dir / "fixed-archive-v4.pt"
        matched = matched_threshold_free_schedule(args, engine.config, scheduled, cache_path) if active else None
        del credit
        training = update(engine, scheduled, cache_path, kernel="v4", active=active)
        training.update(corpus="archive", kernel="v4", supervised_loss_scale=int(active))
        del scheduled
        journal = engine.output_dir / "groups.jsonl"
        rows = [row for row in read_jsonl(journal) if row["tick"] != "skip"]
        demo_rows = [row for row in rows if row["tick"] == "supervised"]
        if active:
            if not training["fixed_reference_reused"] or matched != observed_schedule(journal):
                raise RuntimeError("The active demonstration arm did not replay the matched zero-demo schedule")
            demonstration = [{key: row[key] for key in (
                "index", "optimizer_steps_total", "marked_successes", "supervised_tokens", "true_nll_over_8",
                "preclip_gradient_norm", "postclip_gradient_norm",
            )} for row in demo_rows]
            nonzero = sum(row["preclip_gradient_norm"] > 0 for row in demo_rows)
            training.update(demonstration_updates=demonstration, demonstration_nonzero_steps=nonzero,
                            demonstration_zero_steps=len(demo_rows) - nonzero, shared_reference_arm="archive_null")
        elif any(row["supervised_loss_scale"] != 0 or not row["gradient_parameter_names"]
                 or row["preclip_gradient_norm"] != 0. or row["postclip_gradient_norm"] != 0. for row in demo_rows):
            raise RuntimeError("The zero-demo arm must apply zero gradients to actual AdamW parameters")
        demonstration_indices = [row["index"] for row in demo_rows]
        if not active:
            training["zero_demo_group_indices"] = demonstration_indices
        matches_archive = training["ppo_steps"] == 9 and demonstration_indices == [8, 16, 24, 28]
        fields = ("index", "tick", "optimizer_steps_total", "preclip_gradient_norm", "postclip_gradient_norm")
    else:
        from update import update

        training = update(engine, args.pilot, groups, args.arm, credit=credit)
        rows = [row for row in read_jsonl(engine.output_dir / "groups.jsonl") if row["optimizer_step"]]
        guided_indices = [row["index"] for row in rows if row["selected_arm"] == "Q"]
        training.update(ppo_steps=len(rows), weighted_guided_group_indices=guided_indices)
        matches_archive = guided_indices == ([16, 24, 28] if args.arm == "proposed" else [])
        fields = ("index", "selected_arm", "weight", "optimizer_steps_total",
                  "preclip_gradient_norm", "weighted_gradient_norm")

    actual_steps = training["optimizer_steps"]
    if [row["optimizer_steps_total"] for row in rows] != list(range(1, actual_steps + 1)):
        raise RuntimeError(f"The optimizer step journal disagrees with the actual clock for {args.arm}")
    if not args.independent_base and (not matches_archive or actual_steps != REPLAY_STEPS[args.arm]):
        raise RuntimeError(f"Replay schedule differs from the original pilot for {args.arm}")
    if args.independent_base:
        training["fixed_reference_precompute_reused"] = credit_receipt is not None
        training["fixed_reference_precompute"] = credit_receipt
    return training, [{key: row[key] for key in fields} for row in rows]


def evaluate_independent(engine, tasks, arm):
    from fgpo.appworld_batch import RolloutGroupSpec
    from fgpo.envs import EnvironmentPool
    from fgpo.rollout import collect_evaluation

    start = monotonic()
    path = engine.output_dir / "evaluation.jsonl"
    saved = []
    if path.is_file():
        with path.open("r+b") as journal:
            data = journal.read()
            committed = data.rfind(b"\n") + 1
            if committed < len(data):
                journal.truncate(committed)
        saved = [json.loads(line) for line in data[:committed].splitlines()]
    by_id = match_task_rows(saved, tasks, path)
    resumed_tasks = len(by_id)
    if resumed_tasks < len(tasks):
        config, policy = engine.config, engine.policy
        policy.model.eval()
        policy.sampling_temperature = config["eval_temperature"]
        pool = EnvironmentPool(config, "webshop", engine.output_dir / "environments",
                               size=config["environment_pool_size"])
        task_by_id = {task["task_id"]: task for task in tasks}
        try:
            with path.open("a", buffering=1) as journal:
                def publish(spec, episode):
                    row = {**task_by_id[spec.task_id], **asdict(episode), "arm": arm, "panel": "confirm32",
                           "official_success": episode.official_evaluation["success"],
                           "native_score": episode.official_evaluation["score"]}
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    by_id[spec.task_id] = row

                for offset in range(0, len(tasks), pool.size):
                    specs = tuple(RolloutGroupSpec(
                        f"pilot-evaluation:{task['index']}", task["task_id"], task["environment_seed"],
                        (task["sampling_seed"],),
                    ) for task in tasks[offset:offset + pool.size] if task["task_id"] not in by_id)
                    if specs:
                        with policy.phase("evaluation.rollout", details={"start_index": offset}):
                            collect_evaluation(policy, pool, specs, benchmark="webshop", on_episode_complete=publish)
        finally:
            pool.close()
    if len(by_id) != len(tasks):
        raise RuntimeError("The no-hint evaluation did not score every manifest entry once")
    rows = by_id.values()
    return {
        "tasks": len(by_id), "resumed_completed_tasks": resumed_tasks,
        "official_successes": sum(row["official_success"] for row in rows),
        "official_success_rate": fmean(row["official_success"] for row in rows),
        "mean_native_score": fmean(row["native_score"] for row in rows),
        "mean_steps": fmean(row["steps"] for row in rows), "seconds": monotonic() - start,
    }


def replay(args, tasks):
    if args.independent_base:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    from fgpo.runner import _bind_compute_pair_cpu_affinity, _file_lock, isolate_gpus
    from update import PilotEngine

    if args.arm in ("archive_null", THRESHOLD_FREE_ARM):
        from learn import LearningEngine

        engine_class = LearningEngine
    else:
        engine_class = PilotEngine

    started = monotonic()
    pilot_manifest = args.pilot / "manifest.json"
    pilot = json.loads(pilot_manifest.read_text())
    groups = pilot["tasks"]
    if [group["index"] for group in groups] != list(range(32)):
        raise ValueError("Use the original pilot's 32 frozen update groups in manifest order")
    config = json.loads(args.config.read_text())
    reference = Path(config["model_path"]).resolve()
    source_path = args.pilot / BASE_SOURCE_FILE
    if args.independent_base:
        source = json.loads(source_path.read_text())
        if (source["actor_provenance"] != BASE_SOURCE or Path(source["collection_model"]).resolve() != reference
                or (args.anchor is not None and args.anchor.resolve() != reference) or len(args.gpus) != 1):
            raise ValueError("Independent mode uses the same pretrained collection, actor and reference on one GPU")
        if config.get("checkpoint_backend", "local") != "local":
            raise ValueError("Independent mode uses the local experiment configuration")
        anchor = reference
    else:
        anchor = args.anchor or Path(pilot["source_checkpoint"])
        if source_path.exists() or anchor.resolve() == reference:
            raise ValueError("Archive mode requires the original corpus and a distinct final WebShop actor")
    physical_ids = args.gpus
    logical_ids, _ = isolate_gpus(physical_ids, config["excluded_gpu_ids"])
    _bind_compute_pair_cpu_affinity(config, physical_ids)
    config.update(method="fgpo", benchmark="webshop", seed=0, gpu_ids=logical_ids,
                  allocated_physical_gpu_ids=physical_ids, compute_physical_gpu_ids=physical_ids,
                  compute_logical_gpu_ids=logical_ids, model_placement_explicit=True)

    output = result_root(args) / args.arm
    output.mkdir(parents=True, exist_ok=args.independent_base)
    lock = _file_lock(output / ".run.lock", nonblocking=True) if args.independent_base else nullcontext()
    provenance = BASE_SOURCE if args.independent_base else "fresh_frozen_corpus_replay"
    identity = {
        "arm": args.arm, "actor_provenance": provenance,
        "pilot_manifest": str(pilot_manifest.resolve()), "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()), "actor_anchor": str(anchor.resolve()),
        "reference": config["model_path"], "physical_gpu_ids": physical_ids,
    }
    with lock:
        receipt_path = output / "receipt.json"
        if receipt_path.is_file():
            completed = json.loads(receipt_path.read_text())
            if any(completed[key] != value for key, value in identity.items()):
                raise ValueError("The completed arm belongs to a different experiment")
            print(json.dumps(completed), flush=True)
            return

        trained_path = output / "trained.json"
        trained = None
        saved_model = output / "model"
        if args.independent_base and trained_path.is_file():
            trained_record = json.loads(trained_path.read_text())
            if trained_record["identity"] != identity:
                raise ValueError("The saved training run belongs to a different experiment")
            if saved_model.is_dir() or trained_record["training"]["optimizer_steps"] == 0:
                trained = trained_record
        journal = output / "evaluation.jsonl"
        if args.independent_base and trained is None and journal.is_file() and journal.stat().st_size:
            raise FileNotFoundError("Restore the actor used for the existing evaluation before resuming")

        actor_path = saved_model if trained and saved_model.is_dir() else anchor
        engine = engine_class(config, output, actor_path)
        try:
            finish_arm(engine, args, tasks, groups, identity, trained, started)
        finally:
            engine.close()


def finish_arm(engine, args, tasks, groups, identity, trained, started):
    import torch

    import fgpo
    from fgpo.runner import write_json
    from update import evaluate

    output, config = engine.output_dir, engine.config
    receipt_path = output / "receipt.json"
    evaluation_config = {
        "public_prompt": "v4_raw", "temperature": config["eval_temperature"],
        "top_p": config["eval_top_p"], "horizon": config["horizons"]["webshop"],
        "response_tokens": config["output_tokens"], "hint": False,
        "batch_size": config["environment_pool_size"],
        "batch_size_source": config.get("environment_pool_size_source", "run_config"),
    }
    if (evaluation_config["temperature"] != 0.4 or evaluation_config["top_p"] != 1.0
            or evaluation_config["horizon"] != 15 or evaluation_config["response_tokens"] != 512):
        raise ValueError("Use the original V4 WebShop no-hint evaluation settings")
    ppo_config = {key: config[key] for key in (
        "learning_rate", "weight_decay", "optimizer_betas", "optimizer_eps", "reward_scale",
        "loss_normalization", "ppo_clip_ratio", "ppo_dual_clip", "entropy_coef", "kl_coef",
        "max_grad_norm", "head_chunk_tokens",
    )}
    if args.arm == THRESHOLD_FREE_ARM:
        null = same_trial_receipt(args, "archive_null")
        if (null["evaluation_config"] != evaluation_config
                or json.dumps(null["ppo_config"], sort_keys=True) != json.dumps(ppo_config, sort_keys=True)):
            raise ValueError("The supplement must use the completed zero-demo arm's PPO and evaluation settings")

    saved_model = output / "model"
    if trained is None:
        if args.independent_base:
            torch.use_deterministic_algorithms(True)
        training_deterministic = torch.are_deterministic_algorithms_enabled()
        training, steps = train_arm(engine, args, groups)
        training["deterministic_algorithms"] = training_deterministic
        if args.independent_base:
            write_json(output / "trained.json", {"identity": identity, "training": training, "steps": steps}, durable=True)
        if args.save_model or (args.independent_base and training["optimizer_steps"] > 0):
            pending = output / "model.incoming"
            if pending.is_dir():
                shutil.rmtree(pending)
            engine.policy.save_model(pending)
            pending.rename(saved_model)
    else:
        training, steps = trained["training"], trained["steps"]

    if args.independent_base:
        torch.use_deterministic_algorithms(False)
    evaluation_deterministic = torch.are_deterministic_algorithms_enabled()
    if args.independent_base:
        evaluation = evaluate_independent(engine, tasks, args.arm)
    else:
        evaluation = evaluate(engine, [{**task, "panel": "confirm32"} for task in tasks], args.arm)

    seconds = monotonic() - started
    receipt = {
        **identity, "exit_code": 0, "resumed_saved_training": trained is not None,
        "original_backward_regimen": "BF16 with observed run-to-run variation",
        "source": str(Path(fgpo.__file__).resolve().parent), "command": [sys.executable, *sys.argv],
        "optimizer_initial_step": 0, "actor_seed": 0, "confirmation_seed": 2026111804,
        "passes_over_frozen_collection": 1,
        "ppo_config": ppo_config,
        "training": training, "optimizer_step_log": steps,
        "evaluation": evaluation, "evaluation_config": evaluation_config,
        "host": socket.gethostname(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in engine.gpu_ids],
        "actor_device_map": engine.policy.device_map,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "dtype": str(next(engine.policy.model.parameters()).dtype),
        "attention": engine.policy.metadata["attention"],
        "training_deterministic_algorithms": training["deterministic_algorithms"],
        "evaluation_deterministic_algorithms": evaluation_deterministic,
        "deterministic_algorithms": evaluation_deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "saved_model": str(saved_model.resolve()) if saved_model.is_dir() else None,
        "total_seconds": seconds, "allocated_gpu_seconds": len(engine.gpu_ids) * seconds,
    }
    write_json(receipt_path, receipt, durable=args.independent_base)
    print(json.dumps(receipt), flush=True)


def match_task_rows(rows, tasks, path):
    by_id = {row["task_id"]: row for row in rows}
    expected = {task["task_id"]: task for task in tasks}
    if len(by_id) != len(rows) or by_id.keys() - expected.keys():
        raise ValueError(f"Unexpected or repeated confirm32 task in {path}")
    for task_id, row in by_id.items():
        task = expected[task_id]
        if any(row[key] != task[key] for key in ("environment_seed", "sampling_seed")):
            raise ValueError(f"Seed mismatch for {task_id} in {path}")
    return by_id


def panel_rows(path, tasks):
    rows = [row for row in read_jsonl(path) if row["panel"] == "confirm32"]
    if len(rows) != len(tasks):
        raise ValueError(f"Expected one confirm32 row per manifest task in {path}")
    return match_task_rows(rows, tasks, path)


def paired_difference(pairs):
    gains = sum(a["official_success"] and not b["official_success"] for a, b in pairs)
    losses = sum(b["official_success"] and not a["official_success"] for a, b in pairs)
    discordant = gains + losses
    lower_tail = sum(comb(discordant, i) for i in range(min(gains, losses) + 1))
    score_difference = fmean(a["native_score"] - b["native_score"] for a, b in pairs)
    return {
        "pairs": len(pairs), "full_success_gains": gains, "full_success_losses": losses,
        "success_rate_difference": (gains - losses) / len(pairs),
        "native_score_difference": score_difference, "score_difference_out_of_100": 100 * score_difference,
        "steps_difference": fmean(a["steps"] - b["steps"] for a, b in pairs),
        "exact_paired_discordance_p": min(1., 2 * lower_tail / (2 ** discordant)),
    }


def threshold_free_budget(args, tasks):
    prior = {arm: same_trial_receipt(args, arm) for arm in REPLAY_STEPS}
    collected = 8 * len(tasks) + 4 * prior["archive_null"]["training"]["eligible_groups"]
    evaluated = sum(receipt["evaluation"]["tasks"] for receipt in prior.values())
    total = collected + evaluated + len(tasks)
    if total > 480:
        raise ValueError("The supplemental panel would exceed the independent trial's 480-episode limit")
    return {"collected_training": collected, "previous_evaluation": evaluated,
            "supplemental_evaluation": len(tasks), "total": total, "limit": 480}


def compare_threshold_free(args, tasks, budget):
    from fgpo.runner import write_json

    arms = ("ordinary", "archive_null", THRESHOLD_FREE_ARM)
    root = result_root(args)
    rows = {arm: panel_rows(root / arm / "evaluation.jsonl", tasks) for arm in arms}
    receipts = {arm: same_trial_receipt(args, arm) for arm in arms}
    contrasts, summaries, paired = {}, {}, []
    for arm in arms:
        evaluation, training = receipts[arm]["evaluation"], receipts[arm]["training"]
        summaries[arm] = {
            **evaluation, "mean_score_out_of_100": 100 * evaluation["mean_native_score"],
            "optimizer_steps": training["optimizer_steps"], "ppo_steps": training["ppo_steps"],
            "supervised_ticks": training.get("supervised_ticks", 0),
        }
    for control in arms[:2]:
        pairs = [(rows[THRESHOLD_FREE_ARM][task["task_id"]], rows[control][task["task_id"]]) for task in tasks]
        contrasts[f"{THRESHOLD_FREE_ARM}_minus_{control}"] = {
            **paired_difference(pairs),
            "gained_task_ids": [a["task_id"] for a, b in pairs if a["official_success"] and not b["official_success"]],
            "lost_task_ids": [a["task_id"] for a, b in pairs if b["official_success"] and not a["official_success"]],
        }
    for task in tasks:
        paired.append({**task, "arms": {arm: {key: rows[arm][task["task_id"]][key] for key in METRICS}
                                        for arm in arms}})
    active = receipts[THRESHOLD_FREE_ARM]["training"]
    result = {"panel": "confirm32", "mode": BASE_SOURCE, "trajectory_budget": budget,
              "arms": summaries, "paired_differences": contrasts, "tasks": paired,
              "demonstration_updates": active["demonstration_updates"],
              "demonstration_nonzero_steps": active["demonstration_nonzero_steps"],
              "demonstration_zero_steps": active["demonstration_zero_steps"],
              "shared_fixed_reference_cache": active["fixed_reference_cache"]}
    write_json(root / THRESHOLD_FREE_ARM / "comparison.json", result, durable=True)
    print(json.dumps(result, allow_nan=False), flush=True)


def compare(args, tasks):
    if args.independent_base and args.v5_updates:
        raise ValueError("Compare the independent-Qwen arms within their own result directory")
    output = result_root(args)
    paths = {arm: output / arm / "evaluation.jsonl" for arm in ("ordinary", "proposed")}
    current_null = output / "archive_null" / "evaluation.jsonl"
    if args.independent_base or current_null.is_file():
        paths["archive_null"] = current_null
    historical_null = "historical_archive_null"
    if args.v5_updates:
        for arm in ARCHIVE_ARMS:
            journal = args.v5_updates / arm / "evaluation.jsonl"
            if arm == "archive_null":
                if journal.resolve() == current_null.resolve():
                    raise ValueError("Use a separate archive journal for the historical zero-demo result")
                paths[historical_null] = journal
            elif journal.is_file():
                paths[arm] = journal
    rows = {arm: panel_rows(path, tasks) for arm, path in paths.items()}
    paired = []
    for task in tasks:
        per_arm = {arm: {key: result[task["task_id"]][key] for key in METRICS}
                   for arm, result in rows.items()}
        paired.append({**task, "arms": per_arm})
    summaries = {arm: {
        "official_successes": sum(row["official_success"] for row in results.values()),
        "official_success_rate": fmean(row["official_success"] for row in results.values()),
        "mean_native_score": fmean(row["native_score"] for row in results.values()),
        "mean_score_out_of_100": 100 * fmean(row["native_score"] for row in results.values()),
        "mean_steps": fmean(row["steps"] for row in results.values()),
    } for arm, results in rows.items()}
    for arm, summary in summaries.items():
        receipt_path = paths[arm].with_name("receipt.json")
        if not args.independent_base and not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("actor_provenance") == BASE_SOURCE) != args.independent_base:
            raise ValueError(f"The comparison mode disagrees with the actor provenance for {arm}")
        if args.independent_base:
            training = receipt["training"]
            summary.update(optimizer_steps=training["optimizer_steps"], ppo_steps=training["ppo_steps"],
                           zero_demo_group_indices=training.get("zero_demo_group_indices", []),
                           fixed_reference_precompute_reused=training.get("fixed_reference_precompute_reused", False),
                           fixed_reference_precompute=training.get("fixed_reference_precompute"))
    comparisons = {}
    contrasts = [(arm, "ordinary") for arm in paths if arm != "ordinary"]
    null_controls = [arm for arm in ("archive_null", historical_null) if arm in paths]
    contrasts.extend(("proposed", arm) for arm in null_controls)
    if "archive_success" in paths:
        contrasts.append(("archive_success", historical_null))
    for first, second in contrasts:
        pairs = [(row["arms"][first], row["arms"][second]) for row in paired]
        comparisons[f"{first}_minus_{second}"] = paired_difference(pairs)
    print(json.dumps({"panel": "confirm32", "mode": BASE_SOURCE if args.independent_base else "archived_replay",
                      "journals": {arm: str(path) for arm, path in paths.items()},
                      "primary_zero_demo_control": null_controls[0] if null_controls else None,
                      "arms": summaries, "paired_differences": comparisons, "tasks": paired},
                     indent=2, allow_nan=False))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare-base", help="Record a new or existing independent collection")
    prepare_parser.add_argument("--pilot-manifest", type=Path, required=True, help="Original training task and seed manifest")
    prepare_parser.add_argument("--pilot", type=Path, required=True, help="Independent collection directory")
    prepare_parser.add_argument("--model", type=Path, required=True, help="Pretrained model used by collect.py")
    colab_parser = commands.add_parser("colab-config", help="Write a separate local WebShop config for Colab GPU 0")
    colab_parser.add_argument("--base", type=Path, required=True, help="Colab V4 resource configuration")
    colab_parser.add_argument("--webshop-python", type=Path, required=True, help="Official WebShop worker interpreter")
    colab_parser.add_argument("--webshop-root", type=Path, required=True, help="Official WebShop checkout with its data")
    colab_parser.add_argument("--reference-model", type=Path, help="Local copy of the original pretrained Qwen reference")
    colab_parser.add_argument("--eval-batch-size", type=int, help="Pilot evaluation width if recovered from the archive")
    colab_parser.add_argument("--head-chunk-tokens", type=int, default=64, help="Head chunk width, default 64 from the Host A base config")
    colab_parser.add_argument("--output", type=Path, required=True, help="New config JSON path")
    run_parser = commands.add_parser("run", help="Replay one arm and evaluate confirm32")
    run_parser.add_argument("--pilot", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--arm", choices=REPLAY_STEPS, required=True)
    run_parser.add_argument("--gpus", type=int, nargs="+", required=True)
    run_parser.add_argument("--anchor", type=Path, help="Relocated copy of the original final seed-0 actor")
    run_parser.add_argument("--save-model", action="store_true")
    supplement = commands.add_parser("threshold-free", help="Run the independent matched success-likelihood supplement")
    supplement.add_argument("--pilot", type=Path, required=True)
    supplement.add_argument("--config", type=Path, required=True)
    supplement.add_argument("--manifest", type=Path, required=True)
    supplement.add_argument("--output", type=Path, required=True)
    supplement.add_argument("--gpus", type=int, nargs="+", default=[0])
    supplement.set_defaults(arm=THRESHOLD_FREE_ARM, independent_base=True, anchor=None, save_model=True)
    credit_parser = commands.add_parser("precompute", help="Prepare fixed V4 credit for the complete independent corpus")
    credit_parser.add_argument("--pilot", type=Path, required=True)
    credit_parser.add_argument("--config", type=Path, required=True)
    credit_parser.add_argument("--output", type=Path, required=True)
    credit_parser.add_argument("--gpu", type=int, default=0)
    credit_parser.add_argument("--allocator-budget-gib", type=float, default=19.75)
    credit_parser.add_argument("--arm", choices=REPLAY_STEPS, action="append", help="Repeat to select arms; default is all three")
    compare_parser = commands.add_parser("compare", help="Join the replay journals and any V5 archive journals")
    compare_parser.add_argument("--v5-updates", type=Path, help="Historical updates directory with the archive_null journal and optional other archived arms")
    for command in (run_parser, compare_parser):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--independent-base", action="store_true", help="Use the separate base Qwen experiment")
    args = parser.parse_args()
    if args.command == "prepare-base":
        prepare_base(args)
        return
    if args.command == "colab-config":
        colab_config(args)
        return
    if args.command == "precompute":
        precompute(args)
        return
    tasks = confirmation_tasks(args.manifest)
    if args.command == "threshold-free":
        budget = threshold_free_budget(args, tasks)
        replay(args, tasks)
        compare_threshold_free(args, tasks, budget)
    elif args.command == "run":
        replay(args, tasks)
    else:
        compare(args, tasks)


if __name__ == "__main__":
    main()
