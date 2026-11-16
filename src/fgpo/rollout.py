"""Real same-root F-GPO rollout with independently seeded batched KV slots.

collect_group(policy, env_factory, *, task_id, benchmark, initial_seed,
              slot_seeds, group_id, history_length=None, ledger=None,
              on_complete=None) -> GroupRecord

env_factory(task_id, initial_seed, trajectory_id) returns a fresh isolated
official process.  Training passes eight slot seeds; evaluation passes one.
The complete-group callback is a durable collection boundary.
Independent environment requests run in a slot-bounded thread pool, then drain
at a policy barrier; only the caller writes their ordered physical-cost rows.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

from .filtration import TokenFiltration
from .public import HORIZONS, OUTPUT_LIMITS, TERMINATORS, encode_text, pinned_prompt, parse_native_response, project_public_observation
from .records import GroupRecord, StepRecord, TrajectoryRecord, monotonic_seconds


@dataclass
class _Cohort:
    slots: tuple[int, ...]
    state: object
    suffix: object | None = None


@dataclass
class _RequestResult:
    value: object
    error: BaseException | None
    seconds: float


def _environment_request(call):
    started = monotonic_seconds()
    try:
        return _RequestResult(call(), None, monotonic_seconds() - started)
    except BaseException as error:
        return _RequestResult(None, error, monotonic_seconds() - started)


def _parallel_requests(pool, jobs, ledger, role, *, env_steps=0):
    """Drain actual requests before recording their fees in stable slot order."""
    submitted = []
    submission_error = None
    try:
        for slot, call, details in jobs:
            submitted.append((slot, details, pool.submit(_environment_request, call)))
    except BaseException as error:
        submission_error = error
    completed = [(slot, details, future.result()) for slot, details, future in submitted]
    results = {}
    for slot, details, result in completed:
        receipt = dict(details)
        receipt["outcome"] = "success" if result.error is None else "error"
        if result.error is not None:
            receipt.update(error_type=type(result.error).__name__, error=str(result.error))
        cost = ledger.record("environment", role, seconds=result.seconds,
                             env_steps=env_steps, details=receipt)
        results[slot] = (result, cost)
    if submission_error is not None:
        raise submission_error.with_traceback(submission_error.__traceback__)
    return results


def _raise_request_error(results):
    for result, _ in results.values():
        if result.error is not None:
            raise result.error.with_traceback(result.error.__traceback__)


def _reset_environment(environment, task_id, initial_seed):
    context = environment.reset(task_id, initial_seed=initial_seed)
    return context, environment.action_schema


def _append_owned(policy, state, rows, *, role):
    updated = policy.append_cache(state, rows, role=role)
    if updated is not state:
        policy.release_cache(state)
    return updated


def _sample_turn(policy, cohort, generators, max_tokens):
    state = cohort.state
    ids = [[] for _ in cohort.slots]
    logprobs = [[] for _ in cohort.slots]
    active = list(range(len(cohort.slots)))
    for position in range(max_tokens):
        tokens, scores = policy.draw_tokens(state, active,
            [generators[cohort.slots[row]] for row in active])
        continuing, rows = [], [() for _ in cohort.slots]
        for row, token, score in zip(active, tokens, scores, strict=True):
            ids[row].append(int(token))
            logprobs[row].append(float(score))
            if token not in TERMINATORS and position + 1 < max_tokens:
                continuing.append(row)
                rows[row] = (int(token),)
        if not continuing:
            break
        state = _append_owned(policy, state, rows, role="rollout.decode")
        active = continuing
    return state, ids, logprobs


def _advance_cohort(policy, root, cohort, sampled_state, advances, *, history_length):
    """Consume old cohort ownership before extending exact surviving prefixes."""
    groups = {}
    for index, slot in enumerate(cohort.slots):
        advance = advances[slot]
        if advance.cache_mode != "terminal":
            groups.setdefault(advance.cache_mode, []).append(index)
    cohort.state = None
    append_state = None
    if "append" in groups:
        append_state = policy.select_cache(sampled_state, groups["append"], consume=True)
    else:
        policy.release_cache(sampled_state)
    suffix_state, cohort.suffix = cohort.suffix, None
    if "suffix" in groups:
        if suffix_state is None:
            raise RuntimeError("The exact two-turn policy needs its previously retained suffix cache.")
        suffix_state = policy.select_cache(suffix_state, groups["suffix"], consume=True)
    elif suffix_state is not None:
        policy.release_cache(suffix_state)
        suffix_state = None
    result = []
    for mode, indices in groups.items():
        slots = tuple(cohort.slots[index] for index in indices)
        if mode == "append":
            state, append_state = append_state, None
            state = _append_owned(policy, state, [advances[slot].delta_ids for slot in slots],
                                  role="rollout.turn_append")
        elif mode == "suffix":
            state, suffix_state = suffix_state, None
            state = _append_owned(policy, state, [advances[slot].block.token_ids for slot in slots],
                                  role="rollout.history2_extend")
        else:
            state = policy.fork_cache(root, copies=len(slots))
            state = _append_owned(policy, state, [advances[slot].delta_ids for slot in slots],
                                  role="rollout.segment_replay")
        state = policy.compact_cache(state)
        suffix = None
        if history_length == 2:
            # Keep root+B_t independently from the two-turn actor cache.  Its
            # hidden states never contain the evicted B_(t-1) prefix.
            suffix = policy.fork_cache(root, copies=len(slots))
            suffix = _append_owned(policy, suffix, [advances[slot].block.token_ids for slot in slots],
                                   role="rollout.history2_suffix")
            suffix = policy.compact_cache(suffix)
        result.append(_Cohort(slots, state, suffix))
    return result


def collect_group(policy, env_factory, *, task_id, benchmark, initial_seed, slot_seeds,
                  group_id, history_length=None, ledger=None, on_complete=None):
    if benchmark not in HORIZONS or history_length not in (None, 2):
        raise ValueError("Select the declared F-GPO benchmark and full/two-turn filtration.")
    slot_seeds = tuple(int(seed) for seed in slot_seeds)
    if len(slot_seeds) not in (1, 8) or len(set(slot_seeds)) != len(slot_seeds):
        raise ValueError("Training has eight independently seeded slots; evaluation has one actual trajectory.")
    ledger = policy.ledger if ledger is None else ledger
    if ledger is not policy.ledger:
        raise ValueError("Actor and model forwards share the same actual cost ledger.")
    ledger.begin_group(group_id)
    count, horizon, output_limit = len(slot_seeds), HORIZONS[benchmark], OUTPUT_LIMITS[benchmark]
    ids = tuple(f"{group_id}:slot-{slot}" for slot in range(count))
    environments = [None for _ in slot_seeds]
    contexts, initials, pins = [], [], []
    actor_views, reference_views, steps = [], [], [[] for _ in slot_seeds]
    official = [None for _ in slot_seeds]
    generators = [policy.new_generator(seed) for seed in slot_seeds]
    pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="fgpo-environment")
    primary_error = None

    try:
        creation_results = _parallel_requests(pool, [
            (slot, partial(env_factory, task_id, initial_seed, ids[slot]),
             {"group_id": group_id, "slot": slot, "task_id": task_id, "initial_seed": initial_seed})
            for slot in range(count)], ledger, "rollout.create")
        for slot, (request, _) in creation_results.items():
            if request.error is None:
                environments[slot] = request.value
        _raise_request_error(creation_results)
        reset_results = _parallel_requests(pool, [
            (slot, partial(_reset_environment, environments[slot], task_id, initial_seed),
             {"group_id": group_id, "slot": slot, "task_id": task_id, "initial_seed": initial_seed})
            for slot in range(count)], ledger, "rollout.reset")
        _raise_request_error(reset_results)
        for slot in range(count):
            context, schema = reset_results[slot][0].value
            initial = project_public_observation(policy.tokenizer, benchmark, str(context["observation"]),
                action_schema=schema, goal=str(context["instruction"]), initial=True)
            pin = pinned_prompt(policy.tokenizer, str(schema["native_system_prompt"]), initial)
            contexts.append(context)
            initials.append(initial)
            pins.append(pin)
            actor = TokenFiltration(policy.tokenizer, pin, ids[slot], max_new_tokens=output_limit,
                                    history_length=history_length)
            actor_views.append(actor)
            reference_views.append(actor if history_length is None else TokenFiltration(
                policy.tokenizer, pin, ids[slot], max_new_tokens=output_limit, kind="full_reference"))
        if any(pin != pins[0] for pin in pins[1:]):
            raise ValueError("Same-root sampling requires the actual seeded public goal/bootstrap to agree.")
        root = policy.prefill_root(pins[0], role="rollout.shared_root")
        cohorts = [_Cohort(tuple(range(count)), policy.fork_cache(root, copies=count))]
        for turn in range(horizon):
            sampled_cohorts, actions, jobs = deque(), {}, []
            for cohort in cohorts:
                model_cost_start = len(ledger.records)
                sampled_state, responses, probabilities = _sample_turn(policy, cohort, generators, output_limit)
                cohort.state = None
                model_event_ids = tuple(row.event_id for row in ledger.records[model_cost_start:])
                sampled_cohorts.append((cohort, sampled_state, responses, probabilities, model_event_ids))
                for local_index, slot in enumerate(cohort.slots):
                    step_id = f"{ids[slot]}:step-{turn}"
                    response = tuple(responses[local_index])
                    text = policy.tokenizer.decode(response, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
                    action, syntax_valid = parse_native_response(benchmark, text)
                    actions[slot] = (action, syntax_valid)
                    jobs.append((slot, partial(environments[slot].step_native, action),
                        {"group_id": group_id, "slot": slot, "step_id": step_id,
                         "task_id": task_id, "actual_step": True}))
            cohorts.clear()
            # Every active slot returns its actual feedback before the next
            # policy round; model operations remain single-threaded batches.
            step_results = _parallel_requests(pool, sorted(jobs, key=lambda job: job[0]),
                                               ledger, "rollout.step_native", env_steps=1)
            _raise_request_error(step_results)
            next_cohorts, evaluation_jobs = [], []
            while sampled_cohorts:
                cohort, sampled_state, responses, probabilities, model_event_ids = sampled_cohorts.popleft()
                advances = {}
                for local_index, slot in enumerate(cohort.slots):
                    step_id = f"{ids[slot]}:step-{turn}"
                    response = tuple(responses[local_index])
                    action, syntax_valid = actions[slot]
                    request, environment_cost = step_results[slot]
                    result = request.value
                    observation = project_public_observation(policy.tokenizer, benchmark, str(result.observation),
                        action_schema=environments[slot].action_schema, goal=str(contexts[slot]["instruction"]))
                    terminal = bool(result.done or turn + 1 == horizon)
                    advance = actor_views[slot].append(step_id, response, observation, terminal=terminal)
                    canonical = (advance if reference_views[slot] is actor_views[slot] else
                        reference_views[slot].append(step_id, response, observation, terminal=terminal))
                    advances[slot] = advance
                    steps[slot].append(StepRecord(step_id, ids[slot], turn, advance.response_span.segment_id,
                        advance.prompt_length, advance.response_span, advance.observation_span, response,
                        tuple(probabilities[local_index]), observation.pure_ids, observation.representation_ids,
                        observation.pure_text, observation.raw_observation, dict(action), float(result.reward),
                        bool(result.done), syntax_valid, advance.block, canonical.response_span.segment_id,
                        canonical.prompt_length, canonical.response_span, canonical.observation_span,
                        (environment_cost,), dict(result.info), model_event_ids))
                    if terminal:
                        evaluation_jobs.append((slot, environments[slot].finalize_evaluation,
                            {"group_id": group_id, "slot": slot, "task_id": task_id}))
                next_cohorts.extend(_advance_cohort(policy, root, cohort, sampled_state, advances,
                                                   history_length=history_length))
                del cohort, sampled_state
            evaluation_results = _parallel_requests(pool, sorted(evaluation_jobs, key=lambda job: job[0]),
                                                     ledger, "rollout.official_evaluation")
            _raise_request_error(evaluation_results)
            for slot, (request, _) in evaluation_results.items():
                official[slot] = dict(request.value)
            cohorts = next_cohorts
            if not cohorts:
                break
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            close_results = _parallel_requests(pool, [
                (slot, environment.close, {"group_id": group_id, "slot": slot})
                for slot, environment in enumerate(environments) if environment is not None],
                ledger, "rollout.close")
            if primary_error is None:
                _raise_request_error(close_results)
        finally:
            pool.shutdown(wait=True)
    trajectories = tuple(TrajectoryRecord(ids[slot], group_id, slot, str(task_id), benchmark,
        int(initial_seed), slot_seeds[slot], str(contexts[slot]["instruction"]),
        encode_text(policy.tokenizer, str(contexts[slot]["instruction"])), initials[slot].pure_ids,
        pins[slot], tuple(steps[slot]), actor_views[slot].segments(), reference_views[slot].segments(),
        bool(official[slot]["success"]), official[slot],
        tuple(ledger.records) if count == 1 else tuple(row for row in ledger.records
            if row.kind == "environment" and row.details.get("slot") == slot),
        history_length) for slot in range(count))
    group = GroupRecord(str(group_id), str(task_id), benchmark, int(initial_seed), trajectories, tuple(ledger.records))
    if on_complete is not None:
        on_complete(group)
    return group


run_episodes_fgpo = collect_group
