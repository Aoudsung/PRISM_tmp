"""Collect paired 4+4 WebShop pilot groups with ordinary V4 training records."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch

from fgpo.envs import EnvironmentPool
from fgpo.filtration import TokenFiltration
from fgpo.policy import CausalPolicy
from fgpo.public import (
    HORIZONS,
    IM_END,
    IM_START,
    NEWLINE,
    OUTPUT_LIMITS,
    TERMINATORS,
    pinned_prompt,
    project_public_observation,
)
from fgpo.records import CostLedger, GroupRecord, StepRecord, TrajectoryRecord
from fgpo.rollout import (
    _Cohort,
    _advance_cohort,
    _append_owned,
    _execute_response,
    _merge_cohorts,
    _sample_turn,
)


BENCHMARK = "webshop"
REVIEW_BUDGET = 4096
SUGGESTION_BUDGET = 96
MAX_SERIAL_ATTEMPTS = 32 * 12  # The independent study reserves 3 * 32 of its 480 episodes for evaluation.


@dataclass
class Slot:
    index: int
    seed: int
    environment: object
    goal: str
    initial: object
    view: TokenFiltration
    steps: list[StepRecord]
    metrics: dict | None = None


@dataclass(frozen=True)
class Review:
    turn: int
    hint_ids: tuple[int, ...]
    input_tokens: int
    output_tokens: int


def keep_ends(ids, count, omitted):
    if len(ids) <= count:
        return ids
    if count <= len(omitted):
        return ids[:count]
    left = (count - len(omitted) + 1) // 2
    right = count - len(omitted) - left
    return ids[:left] + omitted + (ids[-right:] if right else ())


def make_review(policy, donor, seed):
    encode = policy.encode
    labels = tuple(enumerate("ABCDEFGHIJKLMNO"[:len(donor.steps)], start=1))
    choices = [encode(letter) for _, letter in labels]
    if not choices or any(len(ids) != 1 for ids in choices):
        raise ValueError("Review requires one distinct token per observed step.")
    choices = [ids[0] for ids in choices]
    if len(set(choices)) != len(choices):
        raise ValueError("Review step tokens must be distinct.")
    system = (
        "Review this recorded WebShop attempt. Identify one observed decision "
        "that is worth retrying to satisfy the shopping goal. Reply with its "
        "step letter followed by one short actionable suggestion for a new "
        "attempt. Use the recorded evidence and do not assume the new attempt "
        "has reached the same page."
    )
    footer = (IM_END, NEWLINE, IM_START) + encode("assistant\n")
    head = ((IM_START,) + encode("system\n") + encode(system)
            + (IM_END, NEWLINE, IM_START) + encode("user\n") + encode(
        f"Goal: {donor.goal}\nOfficial success: {int(donor.official_success)}\n"
    ))
    headers = [encode(f"\nStep {turn} ({letter})\nAction: ") for turn, letter in labels]
    returns = encode("\nPublic return: ")
    omitted = encode(" […] ")
    suffix = encode("\nChoose a listed step letter.") + footer
    delimiter = encode("\nSuggestion: ")
    available = (REVIEW_BUDGET - SUGGESTION_BUDGET - 1 - len(delimiter)
                 - len(head) - len(suffix) - sum(map(len, headers))
                 - len(returns) * len(labels))
    if available < 2 * len(labels):
        raise ValueError("Review budget cannot hold the full goal and every step label.")
    share, remainder = divmod(available, len(labels))
    input_ids = list(head)
    for row, (header, step) in enumerate(zip(headers, donor.steps, strict=True)):
        room = share + int(row < remainder)
        action = encode(json.dumps(dict(step.action), ensure_ascii=False))
        feedback = encode(step.observation_text)
        action_room = min(len(action), max(room // 2, room - len(feedback)))
        input_ids.extend(header)
        input_ids.extend(keep_ends(action, action_room, omitted))
        input_ids.extend(returns)
        input_ids.extend(keep_ends(feedback, room - action_room, omitted))
    input_ids.extend(suffix)
    generator = policy.new_generator(seed)
    state = policy.prefill_root(input_ids, role="review.prompt")
    try:
        conditional = state.next_logits[0, choices].softmax(-1)
        picked = int(torch.multinomial(conditional, 1, generator=generator).item())
        turn = labels[picked][0]
        state = _append_owned(policy, state, [(choices[picked],) + delimiter],
                              role="review.selected_step")
        parked, responses, _ = _sample_turn(policy, _Cohort((0,), state), [generator],
                                             SUGGESTION_BUDGET)
    finally:
        policy.release_cache(state)
    for _, owned in parked:
        policy.release_cache(owned)
    suggestion = policy.tokenizer.decode(
        responses[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    if not suggestion:
        raise ValueError("The frozen reviewer produced an empty suggestion.")
    hint = encode(
        "\n\nA review of an earlier attempt suggested the following for this turn. "
        "Consider it using your current observation.\n" + suggestion
    )
    return Review(turn, hint, len(input_ids) + len(delimiter), 1 + len(responses[0]))


def score_responses(policy, prompts, responses, role):
    """Score all sampled response tokens under the counterfactual exact prefix."""
    prefixes = [prompt + tuple(response[:-1])
                for prompt, response in zip(prompts, responses, strict=True)]
    lengths = [len(prefix) for prefix in prefixes]
    width = max(lengths)
    ids = torch.tensor([
        prefix + (policy.tokenizer.pad_token_id,) * (width - len(prefix))
        for prefix in prefixes
    ], device=policy.device)
    valid = torch.arange(width, device=policy.device)[None] < torch.tensor(
        lengths, device=policy.device
    )[:, None]
    hidden, _, _ = policy.body(ids, attention_mask=valid, actual_tokens=sum(lengths),
                               logical_length=width, teacher_forced=True, role=role)
    scores = []
    for row, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        left = len(prompt) - 1
        chunks = []
        for offset in range(0, len(response), 64):
            targets = torch.tensor(response[offset:offset + 64], device=policy.device)
            logits = policy.head(hidden[row, left + offset:left + offset + len(targets)],
                                 role=role + ".head")
            chunks.append(logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1))
        scores.append(tuple(torch.cat(chunks).cpu().tolist()))
    return scores, sum(lengths)


def reset_slot(policy, tokenizer_lock, environments, task, group_id, index, scope):
    trajectory_id = f"{group_id}:{'guided:' if scope == 'guided' else ''}slot-{index}"
    environment = environments(task["task_id"], task["environment_seed"], trajectory_id)
    try:
        context = environment.reset(task["task_id"], initial_seed=task["environment_seed"])
        schema = environment.action_schema
        goal = str(context["instruction"])
        with tokenizer_lock:
            initial = project_public_observation(
                policy.tokenizer, BENCHMARK, str(context["observation"]),
                action_schema=schema, goal=goal, initial=True)
            pin = pinned_prompt(policy.tokenizer, str(schema["native_system_prompt"]), initial)
            view = TokenFiltration(policy.tokenizer, pin, trajectory_id,
                                   max_new_tokens=policy.max_new_tokens,
                                   max_context_tokens=policy.max_context_tokens)
        return Slot(index, task["slot_seeds"][index], environment, goal, initial, view, [])
    except BaseException:
        environment.close()
        raise


def guided_prefix(view, review):
    position = len(view.prompt_ids) - len(view.assistant_footer_ids)
    if view.prompt_ids[position:] != view.assistant_footer_ids:
        raise ValueError("The current actor prefix lacks the designated assistant header.")
    return view.prompt_ids[:position] + review.hint_ids + view.prompt_ids[position:]


def record_step(slot, response, old_logprobs, actual, turn):
    action, syntax_valid, result, observation = actual
    step_id = f"{slot.view.trajectory_id}:step-{turn}"
    terminal = result.done or turn + 1 == HORIZONS[BENCHMARK]
    advance = slot.view.append(step_id, response, observation, terminal=terminal)
    slot.steps.append(StepRecord(
        step_id, slot.view.trajectory_id, turn, advance.response_span.segment_id,
        advance.prompt_length, advance.response_span, advance.observation_span,
        response, tuple(old_logprobs), observation.pure_ids, observation.representation_ids,
        observation.pure_text, observation.raw_observation, dict(action), float(result.reward),
        bool(result.done), syntax_valid, advance.block, advance.response_span.segment_id,
        advance.prompt_length, advance.response_span, advance.observation_span,
        environment_info=dict(result.info),
    ))
    return advance


def collect_four(policy, environments, executor, task, group_id, indices, scope, review=None, *, on_finalized=None):
    ledger = policy.ledger
    ledger.begin_group(group_id)
    started = perf_counter()
    slots, pending, finalizers, sampled, cohorts = {}, {}, {}, [], []
    marked, scoring_tokens = [], 0
    tokenizer_lock = Lock()

    def dispatch(index, response):
        slot = slots[index]
        pending[index] = executor.submit(
            _execute_response, policy.tokenizer, tokenizer_lock, slot.environment,
            BENCHMARK, response, slot.goal,
        )

    with ExitStack() as leases:
        reset_jobs = [
            executor.submit(reset_slot, policy, tokenizer_lock, environments, task,
                            group_id, index, scope)
            for index in indices
        ]
        for job in reset_jobs:
            if job.exception() is None:
                slot = job.result()
                slots[slot.index] = slot
                leases.callback(slot.environment.close)
        for job in reset_jobs:
            job.result()
        pin = slots[indices[0]].view.pinned_ids
        if any(slot.view.pinned_ids != pin for slot in slots.values()):
            raise ValueError("The four native resets did not share the same actor root.")
        generators = {index: policy.new_generator(slot.seed) for index, slot in slots.items()}
        root = policy.prefill_root(pin, role="rollout.shared_root")
        cohorts = [_Cohort(tuple(indices), policy.fork_cache(root, copies=len(indices)))]
        try:
            for turn in range(HORIZONS[BENCHMARK]):
                is_marked = review is not None and turn + 1 == review.turn
                guided = is_marked and scope == "guided"
                for cohort in cohorts:
                    ordinary_prefixes = guided_prefixes = None
                    if is_marked:
                        views = [slots[index].view for index in cohort.slots]
                        ordinary_prefixes = [view.prompt_ids for view in views]
                        guided_prefixes = [guided_prefix(view, review) for view in views]
                        if any(len(prefix) + policy.max_new_tokens > policy.max_context_tokens
                               for prefix in guided_prefixes):
                            raise ValueError("The guided turn exceeds the exact model context.")
                    sample_cohort = cohort
                    if guided:
                        sample_cohort = _Cohort(cohort.slots, policy.prefill_batch(
                            guided_prefixes, role="rollout.guided_prefix"))
                    with ledger.aggregate_model_calls():
                        parked, responses, sampled_scores = _sample_turn(
                            policy, sample_cohort, generators, policy.max_new_tokens,
                            on_response=dispatch,
                        )
                    sample_cohort.state = None
                    sampled.append((cohort, parked, responses, sampled_scores,
                                    ordinary_prefixes, guided_prefixes))
                next_cohorts = []
                for cohort, parked, responses, sampled_scores, p_prompts, q_prompts in sampled:
                    p_scores = sampled_scores
                    if guided:
                        for _, owned in parked:
                            policy.release_cache(owned)
                        parked[:] = [(tuple(range(len(cohort.slots))), cohort.state)]
                    if is_marked:
                        other_scores, tokens = score_responses(
                            policy, p_prompts if guided else q_prompts, responses,
                            "score.ordinary" if guided else "score.guided",
                        )
                        scoring_tokens += tokens
                        p_scores, q_scores = ((other_scores, sampled_scores) if guided
                                              else (sampled_scores, other_scores))
                        for index, response, p, q in zip(
                            cohort.slots, responses, p_scores, q_scores, strict=True
                        ):
                            marked.append({
                                "slot": index, "generated_tokens": len(response),
                                "logp": math.fsum(p), "logq": math.fsum(q),
                                "sampled_terminator": response[-1] in TERMINATORS,
                            })
                    advances = {}
                    for index, response, old in zip(
                        cohort.slots, responses, p_scores, strict=True
                    ):
                        slot = slots[index]
                        advance = record_step(slot, tuple(response), old,
                                              pending.pop(index).result(), turn)
                        if advance.cache_mode == "terminal":
                            finalizers[index] = executor.submit(slot.environment.finalize_evaluation)
                        if guided and advance.cache_mode == "append":
                            # Ordinary KV still stops before the marked response.
                            advance = replace(advance, delta_ids=advance.block.token_ids)
                        advances[index] = advance
                    next_cohorts.extend(_advance_cohort(
                        policy, root, cohort, parked, advances, history_length=None))
                sampled.clear()
                for index, future in finalizers.items():
                    slots[index].metrics = dict(future.result())
                    if on_finalized is not None:
                        on_finalized()
                finalizers.clear()
                cohorts = _merge_cohorts(policy, next_cohorts, history_length=None)
                if not cohorts:
                    break
        finally:
            for future in (*pending.values(), *finalizers.values()):
                future.exception()
            for _, parked, *_ in sampled:
                for _, owned in parked:
                    policy.release_cache(owned)
            policy.release_cache(root)
    trajectories = []
    for index in indices:
        slot = slots[index]
        segments = slot.view.segments()
        trajectories.append(TrajectoryRecord(
            slot.view.trajectory_id, group_id, index, task["task_id"], BENCHMARK,
            task["environment_seed"], slot.seed, slot.goal, policy.encode(slot.goal),
            slot.initial.pure_ids, pin, tuple(slot.steps), segments, segments,
            bool(slot.metrics["success"]), slot.metrics, (), None,
        ))
    policy.synchronize()
    costs = tuple(ledger.records)
    details = dict(
        seconds=perf_counter() - started,
        input_tokens=sum(row.input_tokens or 0 for row in costs),
        output_tokens=sum(len(step.response_ids) for slot in slots.values() for step in slot.steps),
        scoring_tokens=scoring_tokens,
        environment_steps=sum(len(slot.steps) for slot in slots.values()),
    )
    return tuple(trajectories), costs, marked, details


def density_weight(marked):
    log_ratio = math.fsum(row["logq"] - row["logp"] for row in marked)
    return log_ratio, 2.0 * math.exp(min(-log_ratio, 0.0)) / (1.0 + math.exp(-abs(log_ratio)))


def collect_review(policy, group_id, donor, seed):
    started = perf_counter()
    policy.ledger.begin_group(group_id)
    with policy.ledger.aggregate_model_calls():
        review = make_review(policy, donor, seed)
    return review, tuple(policy.ledger.records), perf_counter() - started


def collect_task(policy, environments, executor, task, serial=None):
    group_id = f"feedback-pilot:{task['index']:03d}"
    started = perf_counter()
    wave = collect_four if serial is None else serial.wave
    first, first_costs, _, first_details = wave(
        policy, environments, executor, task, group_id, range(4), "first")
    eligible = not any(trajectory.success for trajectory in first)
    review, review_costs, review_seconds = None, (), 0.0
    if eligible:
        donor = first[task["donor_slot"]]
        if serial is None:
            review, review_costs, review_seconds = collect_review(policy, group_id, donor, task["review_seed"])
        else:
            review, review_costs, review_seconds = serial.review(policy, task, group_id, donor)
    ordinary, p_costs, p_marked, p_details = wave(
        policy, environments, executor, task, group_id, range(4, 8), "ordinary", review)
    guided, q_costs, q_marked, q_details = None, (), None, None
    if eligible:
        guided, q_costs, q_marked, q_details = wave(
            policy, environments, executor, task, group_id, range(4, 8), "guided", review)
    task_args = group_id, task["task_id"], BENCHMARK, task["environment_seed"]
    p_group = GroupRecord(*task_args, first + ordinary, first_costs + review_costs + p_costs)
    q_group = (GroupRecord(*task_args, first + guided, first_costs + review_costs + q_costs)
               if eligible else None)
    logl_p, weight_p = density_weight(p_marked)
    logl_q, weight_q = density_weight(q_marked) if eligible else (None, None)
    p_reached = {row["slot"] for row in p_marked}
    q_reached = {row["slot"] for row in q_marked} if eligible else None
    picked_q = eligible and task["mixture_arm"] == "Q"
    group = q_group if picked_q else p_group
    successes = sum(row.success for row in group.trajectories)
    diagnostics = {
        "index": task["index"],
        "task_id": task["task_id"],
        "eligible": eligible,
        "mixture_arm": task["mixture_arm"],
        "donor_slot": task["donor_slot"],
        "note_index": review.turn if review else None,
        "first4": [int(row.success) for row in first],
        "second4_P": [int(row.success) for row in ordinary],
        "second4_Q": [int(row.success) for row in guided] if eligible else None,
        "native_score_first4": [row.official_metrics["score"] for row in first],
        "native_score_P": [row.official_metrics["score"] for row in ordinary],
        "native_score_Q": [row.official_metrics["score"] for row in guided] if eligible else None,
        "marked_reach_P": [index in p_reached for index in range(4, 8)],
        "marked_reach_Q": [index in q_reached for index in range(4, 8)] if eligible else None,
        "marked_density_P": p_marked,
        "marked_density_Q": q_marked,
        "logL_P": logl_p,
        "logL_Q": logl_q,
        "w_P": weight_p,
        "w_Q": weight_q,
        "first4_cost": first_details,
        "P_cost": p_details,
        "Q_cost": q_details,
        "review_cost": {
            "seconds": review_seconds,
            "input_tokens": review.input_tokens if review else 0,
            "output_tokens": review.output_tokens if review else 0,
        },
        "wall_seconds": perf_counter() - started,
        "projected_group": {
            "mixture_arm": "Q" if picked_q else "P",
            "eligible": eligible,
            "successes": successes,
            "mixed": 0 < successes < len(group.trajectories),
            "marked_reach": len(q_reached if picked_q else p_reached),
            "trajectories": len(group.trajectories),
        },
    }
    if serial is not None:
        diagnostics["slot_collection"] = serial.details(task)
    return {
        "index": task["index"], "P": p_group, "Q": q_group, "w_P": weight_p,
        "w_Q": weight_q, "eligible": eligible, "mixture_arm": task["mixture_arm"],
        "diagnostics": diagnostics,
    }


def atomic_write(path, write):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    encoded = (json.dumps(payload, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(encoded))


class CollectionPaused(Exception):
    pass


def pause_at_boundary(path):
    if path is not None and path.exists():
        raise CollectionPaused


class SerialCollection:
    """Persist complete slots in order before starting another real episode."""

    def __init__(self, root, samples, args, config):
        mode_path = root / "serial-collection.json"
        mode = {
            "slot_batch_size": 1, "model": str(args.model.resolve()),
            "max_context_tokens": config["max_context_tokens"],
            "cache_initial_capacity": config["cache_initial_capacity"],
        }
        if mode_path.is_file():
            if json.loads(mode_path.read_text()) != mode:
                raise ValueError("Resume the serial corpus with its original model and sampling settings")
        else:
            if any(samples.glob("*.pt")):
                raise ValueError("Use a fresh sample directory for the serial corpus")
            write_json(mode_path, mode)
        self.root = root / "slot-progress"
        self.allocator_budget_gib = args.allocator_budget_gib
        self.pause_file = args.pause_file
        self.attempts = sum(json.loads(path.read_text())["attempts"] for path in self.root.glob("*/*.attempt.json"))

    def task_dir(self, task):
        return self.root / f"{task['index']:03d}"

    def wave(self, policy, environments, executor, task, group_id, indices, scope, review=None):
        folder = self.task_dir(task)
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for index in indices:
            slot_path = folder / f"{scope}-{index}.pt"
            if slot_path.is_file():
                records.append(torch.load(slot_path, map_location="cpu", weights_only=False))
                continue
            pause_at_boundary(self.pause_file)
            attempt_path = folder / f"{scope}-{index}.attempt.json"
            past = json.loads(attempt_path.read_text()) if attempt_path.is_file() else {}
            previous = past.get("attempts", 0)
            if self.attempts >= MAX_SERIAL_ATTEMPTS:
                raise RuntimeError("The collection has used all 384 reserved sampling attempts")
            attempt = {
                "index": task["index"], "scope": scope, "slot": index, "seed": task["slot_seeds"][index],
                "attempts": previous + 1, "prior_attempts_without_committed_record": previous,
                "prior_finalized_without_committed_record": (past.get("prior_finalized_without_committed_record", 0)
                                                               + int(past.get("native_finalized", False))),
                "allocator_budgets_gib": sorted({*past.get("allocator_budgets_gib", ()), self.allocator_budget_gib}),
                "native_finalized": False,
            }
            write_json(attempt_path, attempt)
            self.attempts += 1
            if previous:
                print(f"retry slot task={task['index']} wave={scope} slot={index} "
                      f"uncommitted_attempts={previous}", flush=True)
            def finalized():
                attempt["native_finalized"] = True
                write_json(attempt_path, attempt)

            try:
                record = collect_four(policy, environments, executor, task, group_id, (index,), scope, review,
                                      on_finalized=finalized)
                atomic_write(slot_path, lambda stream: torch.save(record, stream))
            except BaseException as error:
                print(f"slot uncommitted task={task['index']} wave={scope} slot={index} "
                      f"attempt={previous + 1} error={type(error).__name__}", flush=True)
                raise
            records.append(record)
            print(f"slot committed task={task['index']} wave={scope} slot={index} "
                  f"sampling_attempts_total={self.attempts}", flush=True)
            torch.cuda.empty_cache()
            pause_at_boundary(self.pause_file)
        trajectories = tuple(trajectory for batch, _, _, _ in records for trajectory in batch)
        costs = tuple(cost for _, events, _, _ in records for cost in events)
        marks = [mark for _, _, marked, _ in records for mark in marked]
        details = {key: sum(record[3][key] for record in records) for key in records[0][3]}
        return trajectories, costs, marks, details

    def review(self, policy, task, group_id, donor):
        path = self.task_dir(task) / "review.pt"
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=False)
        pause_at_boundary(self.pause_file)
        result = collect_review(policy, group_id, donor, task["review_seed"])
        atomic_write(path, lambda stream: torch.save(result, stream))
        torch.cuda.empty_cache()
        return result

    def details(self, task):
        rows = [json.loads(path.read_text()) for path in sorted(self.task_dir(task).glob("*.attempt.json"))]
        return {
            "slot_batch_size": 1,
            "attempts": sum(row["attempts"] for row in rows),
            "allocator_budgets_gib": sorted({budget for row in rows for budget in row["allocator_budgets_gib"]}),
            "prior_attempts_without_committed_record": [row for row in rows if row["attempts"] > 1],
        }


def run(args):
    if args.slot_batch_size == 1:
        from fgpo.runner import _file_lock

        if args.allocator_budget_gib is None:
            raise ValueError("Serial collection requires an explicit CUDA allocator budget")
        try:
            with _file_lock(args.root.resolve() / ".serial-sampler.lock", nonblocking=True):
                pause_at_boundary(args.pause_file)
                run_shard(args)
        except CollectionPaused:
            print(f"sampler paused at a saved slot boundary shard={args.shard}", flush=True)
            raise SystemExit(75) from None
    else:
        if args.pause_file is not None:
            raise ValueError("A pause file requires --slot-batch-size 1")
        run_shard(args)


@torch.inference_mode()
def run_shard(args):
    root = args.root.resolve()
    config = json.loads(args.config.read_text())
    rows = json.loads((root / "manifest.json").read_text())["tasks"]
    if sorted(row["index"] for row in rows) != list(range(32)):
        raise ValueError("The manifest must specify all 32 fixed task indices.")
    for row in rows:
        if not (row["task_id"].startswith(f"train/{row['environment_seed']}/")
                and 0 <= row["donor_slot"] < 4 and row["mixture_arm"] in ("P", "Q")
                and len(set(row["slot_seeds"])) == len(row["slot_seeds"]) == 8):
            raise ValueError(f"Invalid training manifest entry at index {row['index']}.")
    if torch.cuda.device_count() != 1:
        raise ValueError("Each isolated pilot shard requires exactly one visible CUDA device.")
    if args.allocator_budget_gib is not None:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        fraction = args.allocator_budget_gib * 2**30 / total_bytes
        if not 0. < fraction <= 1.:
            raise ValueError("The CUDA allocator budget must be positive and fit the visible device")
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    samples = root / "samples"
    samples.mkdir(exist_ok=True)
    serial = None
    if args.slot_batch_size == 1:
        serial = SerialCollection(root, samples, args, config)
    elif (root / "serial-collection.json").exists():
        raise ValueError("Continue this corpus with --slot-batch-size 1")
    ledger = CostLedger(gpu_ids=(0,))
    policy = CausalPolicy(
        str(args.model), gpu_ids=(0,), ledger=ledger, trainable=False, benchmark=BENCHMARK,
        max_context_tokens=int(config["max_context_tokens"]),
        max_new_tokens=OUTPUT_LIMITS[BENCHMARK],
        cache_initial_capacity=int(config["cache_initial_capacity"]),
        working_memory_gib=float(config["working_memory_gib"]),
    )
    policy.sampling_temperature = 1.0
    environments = EnvironmentPool(
        config, BENCHMARK, root / f"workers-{args.shard}", size=args.slot_batch_size,
    )
    print(f"sampler ready shard={args.shard} gpu={os.environ['CUDA_VISIBLE_DEVICES']} "
          f"slot_batch_size={args.slot_batch_size} allocator_budget_gib={args.allocator_budget_gib}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.slot_batch_size) as executor:
            for task in sorted(rows, key=lambda row: row["index"]):
                index = task["index"]
                if index % 2 != args.shard:
                    continue
                record = samples / f"{index:03d}.pt"
                companion = samples / f"{index:03d}.json"
                if record.exists() and companion.exists():
                    continue
                if record.exists():
                    sample = torch.load(record, map_location="cpu", weights_only=False)
                else:
                    print(f"start task index={index}", flush=True)
                    sample = collect_task(policy, environments, executor, task, serial)
                    atomic_write(record, lambda stream: torch.save(sample, stream))
                details = sample["diagnostics"]
                write_json(companion, details)
                print(f"done task index={index} eligible={int(details['eligible'])} "
                      f"seconds={details['wall_seconds']:.1f}", flush=True)
    finally:
        environments.close()
    peak_gib = torch.cuda.max_memory_reserved() / 2**30
    print(f"sampler complete shard={args.shard} peak_reserved_gib={peak_gib:.2f}", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--slot-batch-size", type=int, choices=(1, 4), default=4)
    parser.add_argument("--allocator-budget-gib", type=float, help="Hard process-local PyTorch CUDA allocator ceiling")
    parser.add_argument("--pause-file", type=Path, help="Exit at a saved slot boundary when this file appears")
    run(parser.parse_args())
