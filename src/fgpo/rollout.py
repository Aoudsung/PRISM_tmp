"""Real same-root F-GPO rollout with independently seeded batched KV slots.

collect_group(policy, env_factory, *, task_id, benchmark, initial_seed,
              slot_seeds, group_id, history_length=None, ledger=None,
              on_complete=None) -> GroupRecord

env_factory(task_id, initial_seed, trajectory_id) returns an exclusive lease on
an isolated official process. Training passes independent slot seeds; evaluation passes one.
The complete-group callback is a durable collection boundary.
Completed responses dispatch environment requests while other slots decode.
Requests drain at a turn boundary; only the caller writes ordered cost rows.
Window-rollover siblings merge back into one batch at that same boundary.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from threading import Lock

from .filtration import TokenFiltration
from .public import HORIZONS, TERMINATORS, encode_text, pinned_prompt, parse_native_response, project_public_observation
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


def _drain_requests(submitted, ledger, role, *, env_steps=0):
    """Join outstanding requests and record their actual work in slot order."""
    completed = [(slot, details, future.result())
                 for slot, details, future in sorted(submitted, key=lambda item: item[0])]
    results = {}
    for slot, details, result in completed:
        receipt = dict(details)
        receipt["outcome"] = "success" if result.error is None else "error"
        if result.error is not None:
            receipt.update(error_type=type(result.error).__name__, error=str(result.error))
        cost = ledger.record("environment", role, seconds=result.seconds,
                             env_steps=env_steps, details=receipt)
        results[slot] = (result, cost)
    return results


def _parallel_requests(pool, jobs, ledger, role, *, env_steps=0):
    submitted = []
    submission_error = None
    try:
        for slot, call, details in jobs:
            submitted.append((slot, details, pool.submit(_environment_request, call)))
    except BaseException as error:
        submission_error = error
    results = _drain_requests(submitted, ledger, role, env_steps=env_steps)
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


def _release_cohort(policy, cohort):
    if cohort is not None:
        if cohort.state is not None:
            policy.release_cache(cohort.state)
            cohort.state = None
        if cohort.suffix is not None:
            policy.release_cache(cohort.suffix)
            cohort.suffix = None


def _sample_turn(policy, cohort, generators, max_tokens, *, on_response=None):
    state = cohort.state
    ids = [[] for _ in cohort.slots]
    logprobs = [[] for _ in cohort.slots]
    active = list(range(len(cohort.slots)))
    parked = []
    workspace = policy.decode_workspace(state, max_tokens)
    try:
        for position in range(max_tokens):
            local_rows = tuple(range(len(active)))
            tokens, scores = policy.draw_tokens(state, local_rows,
                [generators[cohort.slots[row]] for row in active], workspace=workspace)
            continuing = []
            for local, (row, token, score) in enumerate(zip(active, tokens, scores, strict=True)):
                ids[row].append(int(token))
                logprobs[row].append(float(score))
                if token not in TERMINATORS and position + 1 < max_tokens:
                    continuing.append(local)
                elif on_response is not None:
                    on_response(cohort.slots[row], tuple(ids[row]))
            if not continuing:
                parked.append((tuple(active), state))
                state = None
                break
            if len(continuing) != len(active):
                keep = set(continuing)
                finished = tuple(row for row in local_rows if row not in keep)
                ended, state = policy.split_cache(state, (finished, continuing),
                                                  preserve_capacity=(False, True))
                parked.append((tuple(active[row] for row in finished), ended))
                workspace.select_rows(continuing, physical=state.valid_mask.shape[1])
                active = [active[row] for row in continuing]
            # EOS/cap tokens remain outside KV, exactly as on the old path;
            # next-turn delta_ids append each slot's last sampled token once.
            previous = state
            state = policy.append_sampled_cache(state, workspace, range(len(active)), role="rollout.decode")
            if state is not previous:
                policy.release_cache(previous)
        return parked, ids, logprobs
    except BaseException:
        # A successful append has transferred storage out of cohort.state's
        # original wrapper; the current owner lives in this local variable.
        if state is not None:
            policy.release_cache(state)
        for _, owned in parked:
            policy.release_cache(owned)
        cohort.state = None
        raise
    finally:
        workspace.close()


def _execute_response(tokenizer, tokenizer_lock, environment, benchmark, response, goal):
    """CPU-only parse, native transition and public projection for one slot."""
    # Fast-tokenizer encode calls can update shared padding/truncation state.
    # Serialize that short CPU section; native RPCs remain fully concurrent.
    with tokenizer_lock:
        text = tokenizer.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    action, syntax_valid = parse_native_response(benchmark, text)
    result = environment.step_native(action)
    with tokenizer_lock:
        observation = project_public_observation(tokenizer, benchmark, str(result.observation),
                                                action_schema=environment.action_schema, goal=goal)
    return action, syntax_valid, result, observation


def _advance_cohort(policy, root, cohort, parked, advances, *, history_length):
    """Consume old cohort ownership before extending exact surviving prefixes."""
    groups = {}
    for index, slot in enumerate(cohort.slots):
        advance = advances[slot]
        if advance.cache_mode != "terminal":
            groups.setdefault(advance.cache_mode, []).append(index)
    append_pieces = []
    append_state = suffix_state = state = suffix = None
    result = []
    try:
        cohort.state = None
        if "append" in groups:
            append_indices = groups["append"]
            index_to_dest = {idx: pos for pos, idx in enumerate(append_indices)}
            while parked:
                piece_indices, piece_state = parked.pop(0)
                keep_local = [local for local, idx in enumerate(piece_indices) if idx in index_to_dest]
                if not keep_local:
                    policy.release_cache(piece_state)
                    continue
                if len(keep_local) == len(piece_indices):
                    dest = tuple(index_to_dest[idx] for idx in piece_indices)
                    append_pieces.append((piece_state, dest))
                else:
                    selected = policy.select_cache(piece_state, keep_local, consume=True, preserve_capacity=True)
                    dest = tuple(index_to_dest[piece_indices[local]] for local in keep_local)
                    append_pieces.append((selected, dest))
            if len(append_pieces) == 1 and append_pieces[0][1] == tuple(range(len(append_indices))):
                append_state = append_pieces.pop()[0]
            else:
                pieces_states = [st for st, _ in append_pieces]
                pieces_groups = [dest for _, dest in append_pieces]
                append_pieces.clear()
                append_state = policy.merge_cache(pieces_states, row_groups=pieces_groups)
        else:
            while parked:
                _, owned = parked.pop(0)
                policy.release_cache(owned)
        suffix_state, cohort.suffix = cohort.suffix, None
        if "suffix" in groups:
            if suffix_state is None:
                raise RuntimeError("The exact two-turn policy needs its previously retained suffix cache.")
            suffix_state = policy.select_cache(suffix_state, groups["suffix"], consume=True)
        elif suffix_state is not None:
            policy.release_cache(suffix_state)
            suffix_state = None
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
            if history_length == 2:
                # Keep root+B_t independently from the two-turn actor cache.
                # Its hidden states never contain the evicted B_(t-1) prefix.
                suffix = policy.fork_cache(root, copies=len(slots))
                suffix = _append_owned(policy, suffix, [advances[slot].block.token_ids for slot in slots],
                                       role="rollout.history2_suffix")
                suffix = policy.compact_cache(suffix)
            result.append(_Cohort(slots, state, suffix))
            state = suffix = None
        return result
    except BaseException:
        # Some modes may have completed while a later mode fails.  None of
        # these owners has been returned to the caller's next-cohort queue yet.
        for _, owned in parked:
            policy.release_cache(owned)
        parked.clear()
        for owned, _ in append_pieces:
            policy.release_cache(owned)
        append_pieces.clear()
        for owned in (append_state, suffix_state, state, suffix):
            if owned is not None:
                policy.release_cache(owned)
        for completed in result:
            _release_cohort(policy, completed)
        raise


def _merge_cohorts(policy, cohorts, *, history_length):
    if len(cohorts) <= 1:
        return cohorts
    slots = tuple(slot for cohort in cohorts for slot in cohort.slots)
    state = suffix = None
    try:
        with policy.phase("rollout.merge_cohorts"):
            state = policy.merge_cache([cohort.state for cohort in cohorts])
            if history_length == 2:
                suffix = policy.merge_cache([cohort.suffix for cohort in cohorts])
        return [_Cohort(slots, state, suffix)]
    except BaseException:
        for owned in (state, suffix):
            if owned is not None:
                policy.release_cache(owned)
        raise
    finally:
        for cohort in cohorts:
            _release_cohort(policy, cohort)


def collect_group(policy, env_factory, *, task_id, benchmark, initial_seed, slot_seeds,
                  group_id, history_length=None, ledger=None, on_complete=None):
    if benchmark not in HORIZONS or history_length not in (None, 2):
        raise ValueError("Select the declared F-GPO benchmark and full/two-turn filtration.")
    slot_seeds = tuple(int(seed) for seed in slot_seeds)
    if not slot_seeds or len(set(slot_seeds)) != len(slot_seeds):
        raise ValueError("Each rollout slot requires its own sampling seed.")
    ledger = policy.ledger if ledger is None else ledger
    if ledger is not policy.ledger:
        raise ValueError("Actor and model forwards share the same actual cost ledger.")
    ledger.begin_group(group_id)
    count, horizon, output_limit = len(slot_seeds), HORIZONS[benchmark], policy.max_new_tokens
    ids = tuple(f"{group_id}:slot-{slot}" for slot in range(count))
    environments = [None for _ in slot_seeds]
    contexts, initials, pins = [], [], []
    actor_views, reference_views, steps = [], [], [[] for _ in slot_seeds]
    official = [None for _ in slot_seeds]
    generators = [policy.new_generator(seed) for seed in slot_seeds]
    pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="fgpo-environment")
    tokenizer_lock = Lock()
    primary_error = None
    submitted_steps = []
    root, cohorts = None, []
    sampled_cohorts, next_cohorts = deque(), []
    cohort = sampled_parked = None

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
                                    history_length=history_length, max_context_tokens=policy.max_context_tokens)
            actor_views.append(actor)
            # r3 reference laws always condition on the actual actor view.
            reference_views.append(actor)
        if any(pin != pins[0] for pin in pins[1:]):
            raise ValueError("Same-root sampling requires the actual seeded public goal/bootstrap to agree.")
        root = policy.prefill_root(pins[0], role="rollout.shared_root")
        cohorts = [_Cohort(tuple(range(count)), policy.fork_cache(root, copies=count))]
        for turn in range(horizon):
            sampled_cohorts = deque()

            def dispatch(slot, response):
                step_id = f"{ids[slot]}:step-{turn}"
                call = partial(_execute_response, policy.tokenizer, tokenizer_lock, environments[slot], benchmark,
                               response, str(contexts[slot]["instruction"]))
                details = {"group_id": group_id, "slot": slot, "step_id": step_id,
                           "task_id": task_id, "actual_step": True,
                           "timing_scope": "parse_native_step_and_public_projection"}
                submitted_steps.append((slot, details, pool.submit(_environment_request, call)))

            for cohort in cohorts:
                model_cost_start = len(ledger.records)
                with ledger.aggregate_model_calls():
                    sampled_parked, responses, probabilities = _sample_turn(
                        policy, cohort, generators, output_limit, on_response=dispatch)
                cohort.state = None
                model_event_ids = tuple(row.event_id for row in ledger.records[model_cost_start:])
                sampled_cohorts.append((cohort, sampled_parked, responses, probabilities, model_event_ids))
                sampled_parked = None
            cohorts.clear()
            # Every active slot returns its actual feedback before the next
            # policy round; model operations remain single-threaded batches.
            step_results = _drain_requests(submitted_steps, ledger, "rollout.step_native", env_steps=1)
            submitted_steps.clear()
            _raise_request_error(step_results)
            next_cohorts, evaluation_jobs = [], []
            while sampled_cohorts:
                cohort, sampled_parked, responses, probabilities, model_event_ids = sampled_cohorts.popleft()
                advances = {}
                for local_index, slot in enumerate(cohort.slots):
                    step_id = f"{ids[slot]}:step-{turn}"
                    response = tuple(responses[local_index])
                    request, environment_cost = step_results[slot]
                    action, syntax_valid, result, observation = request.value
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
                next_cohorts.extend(_advance_cohort(policy, root, cohort, sampled_parked, advances,
                                                   history_length=history_length))
                cohort = sampled_parked = None
            evaluation_results = _parallel_requests(pool, sorted(evaluation_jobs, key=lambda job: job[0]),
                                                     ledger, "rollout.official_evaluation")
            _raise_request_error(evaluation_results)
            for slot, (request, _) in evaluation_results.items():
                official[slot] = dict(request.value)
            cohorts = _merge_cohorts(policy, next_cohorts, history_length=history_length)
            next_cohorts = []
            if not cohorts:
                break
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            # An exception during another slot's decode must not race a pending
            # step RPC against close/release on the same worker channel.
            if submitted_steps:
                pending = _drain_requests(submitted_steps, ledger, "rollout.step_native", env_steps=1)
                submitted_steps.clear()
                if primary_error is None:
                    _raise_request_error(pending)
            close_results = _parallel_requests(pool, [
                (slot, environment.close, {"group_id": group_id, "slot": slot})
                for slot, environment in enumerate(environments) if environment is not None],
                ledger, "rollout.close")
            if primary_error is None:
                _raise_request_error(close_results)
        finally:
            try:
                pool.shutdown(wait=True)
            finally:
                # Ownership may be in a decode queue, the current advancement,
                # or a completed next-turn queue when an environment/model
                # operation fails.  Repeated release of a wrapper is harmless.
                for pending_cohort, pending_parked, *_ in sampled_cohorts:
                    for _, owned in pending_parked:
                        policy.release_cache(owned)
                    _release_cohort(policy, pending_cohort)
                sampled_cohorts.clear()
                for pending_cohort in cohorts:
                    _release_cohort(policy, pending_cohort)
                for pending_cohort in next_cohorts:
                    _release_cohort(policy, pending_cohort)
                _release_cohort(policy, cohort)
                if sampled_parked is not None:
                    for _, owned in sampled_parked:
                        policy.release_cache(owned)
                if root is not None:
                    policy.release_cache(root)
    built = []
    for slot in range(count):
        actor_segments = actor_views[slot].segments()
        reference_segments = (actor_segments if reference_views[slot] is actor_views[slot]
                              else reference_views[slot].segments())
        built.append(TrajectoryRecord(ids[slot], group_id, slot, str(task_id), benchmark,
            int(initial_seed), slot_seeds[slot], str(contexts[slot]["instruction"]),
            encode_text(policy.tokenizer, str(contexts[slot]["instruction"])), initials[slot].pure_ids,
            pins[slot], tuple(steps[slot]), actor_segments, reference_segments,
            bool(official[slot]["success"]), official[slot],
            tuple(ledger.records) if count == 1 else tuple(row for row in ledger.records
                if row.kind == "environment" and row.details.get("slot") == slot),
            history_length))
    trajectories = tuple(built)
    actor_views.clear()
    reference_views.clear()
    steps.clear()
    contexts.clear()
    initials.clear()
    pins.clear()
    del built
    group = GroupRecord(str(group_id), str(task_id), benchmark, int(initial_seed), trajectories, tuple(ledger.records))
    if on_complete is not None:
        on_complete(group)
    return group


run_episodes_fgpo = collect_group
