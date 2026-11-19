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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from threading import Lock

from .appworld_batch import EvaluationEpisode, RolloutGroupSpec
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


def _record_request(ledger, role, details, result, *, env_steps=0):
    receipt = dict(details)
    receipt["outcome"] = "success" if result.error is None else "error"
    if result.error is not None:
        receipt.update(error_type=type(result.error).__name__, error=str(result.error))
    return ledger.record("environment", role, seconds=result.seconds, env_steps=env_steps, details=receipt)


def _drain_requests(submitted, ledger, role, *, env_steps=0):
    """Join outstanding requests and record their actual work in slot order."""
    completed = [(slot, details, future.result())
                 for slot, details, future in sorted(submitted, key=lambda item: item[0])]
    results = {}
    for slot, details, result in completed:
        cost = _record_request(ledger, role, details, result, env_steps=env_steps)
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


def _score_and_release(environment):
    scored = _environment_request(environment.finalize_evaluation)
    released = _environment_request(environment.close)
    return scored, released


def _finish_appworld_episodes(pool, environments, rows, details, ledger, *, score_role, close_role, on_complete):
    """Publish each completed lease on the caller thread while other scores drain."""
    submitted = {}
    first_error = None
    try:
        for row in rows:
            submitted[pool.submit(_score_and_release, environments[row])] = row
    except BaseException as error:
        first_error = error
    publication_failed = False
    for future in as_completed(submitted):
        row = submitted[future]
        try:
            scored, released = future.result()
            if released.error is None:
                environments[row] = None
            score_cost = _record_request(ledger, score_role, details(row), scored)
            _record_request(ledger, close_role, details(row), released)
        except BaseException as error:
            first_error = first_error or error
            continue
        error = scored.error if scored.error is not None else released.error
        if error is not None:
            first_error = first_error or error
        elif not publication_failed:
            try:
                on_complete(row, dict(scored.value), score_cost)
            except BaseException as error:
                # An interrupted append can leave a partial final line.
                # Stop appending to that journal while other leases release.
                publication_failed = isinstance(error, OSError)
                first_error = first_error or error
    if first_error is not None:
        raise first_error.with_traceback(first_error.__traceback__)


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


def _sample_turn(policy, cohort, generators, max_tokens, *, on_response=None, split_kv=False):
    state = cohort.state
    ids = [[] for _ in cohort.slots]
    logprobs = [[] for _ in cohort.slots]
    active = list(range(len(cohort.slots)))
    parked = []
    if split_kv:
        workspace = policy.decode_workspace(state, max_tokens, split_kv=True)
    else:
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


def _advance_cohort(policy, root, cohort, parked, advances, *, history_length, independent_roots=False):
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
                state = (policy.select_cache(root, slots, preserve_capacity=False) if independent_roots
                         else policy.fork_cache(root, copies=len(slots)))
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


def _advance_paired_cohort(policy, roots, cohort, parked, advances, row_groups, ledger):
    """Return each group's append rows and replay candidates to its own root."""
    slots_by_group = {}
    for slot in cohort.slots:
        slots_by_group.setdefault(row_groups[slot], []).append(slot)
    if len(slots_by_group) == 1:
        group = next(iter(slots_by_group))
        with ledger.context(group_id=group):
            return _advance_cohort(policy, roots[group], cohort, parked, advances, history_length=None)
    pieces = {group: [] for group in slots_by_group}
    local_by_group = {slot: local for slots in slots_by_group.values() for local, slot in enumerate(slots)}
    next_cohorts = []
    try:
        while parked:
            indices, state = parked.pop(0)
            keep = {}
            for local, index in enumerate(indices):
                slot = cohort.slots[index]
                if advances[slot].cache_mode == "append":
                    keep.setdefault(row_groups[slot], []).append((local, local_by_group[slot]))
            if not keep:
                policy.release_cache(state)
                continue
            try:
                parts = policy.split_cache(state, [[local for local, _ in rows] for rows in keep.values()],
                                           drop_unselected=True)
            except BaseException:
                policy.release_cache(state)
                raise
            for (group, rows), owned in zip(keep.items(), parts, strict=True):
                pieces[group].append((tuple(destination for _, destination in rows), owned))
        for group, slots in slots_by_group.items():
            with ledger.context(group_id=group):
                next_cohorts.extend(_advance_cohort(policy, roots[group], _Cohort(tuple(slots), None),
                    pieces.pop(group), advances, history_length=None))
        return next_cohorts
    except BaseException:
        for pending in pieces.values():
            for _, owned in pending:
                policy.release_cache(owned)
        for ready in next_cohorts:
            _release_cohort(policy, ready)
        raise


def _merge_cohorts(policy, cohorts, *, history_length, ordered=False):
    if len(cohorts) <= 1:
        return cohorts
    slots = tuple(slot for cohort in cohorts for slot in cohort.slots)
    row_groups = None
    if ordered and slots != tuple(sorted(slots)):
        slots = tuple(sorted(slots))
        destinations = {slot: row for row, slot in enumerate(slots)}
        row_groups = [tuple(destinations[slot] for slot in cohort.slots) for cohort in cohorts]
    state = suffix = None
    try:
        with policy.phase("rollout.merge_cohorts"):
            state = policy.merge_cache([cohort.state for cohort in cohorts], row_groups=row_groups)
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
                  group_id, history_length=None, ledger=None, on_complete=None, split_kv=False):
    spec = RolloutGroupSpec(group_id, task_id, initial_seed, tuple(slot_seeds))
    ledger = policy.ledger if ledger is None else ledger
    trajectories = _collect_slots(policy, env_factory, (spec,), benchmark=benchmark,
                                  history_length=history_length, ledger=ledger, split_kv=split_kv)
    group = GroupRecord(str(group_id), str(task_id), benchmark, int(initial_seed), trajectories, tuple(ledger.records))
    if on_complete is not None:
        on_complete(group)
    return group


def collect_groups(policy, env_factory, specs, *, benchmark, split_kv=False):
    specs = tuple(specs)
    if benchmark != "appworld" or len(specs) != 2 or any(len(spec.slot_seeds) != 8 for spec in specs):
        raise ValueError("Paired collection requires two eight-slot AppWorld groups")
    group_ids = tuple(spec.group_id for spec in specs)
    if len(set(group_ids)) != 2:
        raise ValueError("Paired collection requires two distinct original group identities")
    ledger = policy.ledger
    with ledger.context(group_id=None, group_ids=group_ids):
        with policy.phase("rollout.collection_batch", details={"group_ids": group_ids,
                "group_count": 2, "trajectory_count": 16}):
            trajectories = _collect_slots(policy, env_factory, specs, benchmark=benchmark,
                                          history_length=None, ledger=ledger, split_kv=split_kv)
        for spec in specs:
            ledger.record("controller", "rollout.group_complete",
                          details={"group_id": spec.group_id, "trajectory_count": len(spec.slot_seeds)})
    shared_events = tuple(row.event_id for row in ledger.records if "group_id" not in row.details)
    return tuple(GroupRecord(spec.group_id, str(spec.task_id), benchmark, int(spec.initial_seed),
        tuple(trajectory for trajectory in trajectories if trajectory.group_id == spec.group_id),
        tuple(row for row in ledger.records if row.details.get("group_id") == spec.group_id), shared_events)
        for spec in specs)


def collect_appworld_evaluation(policy, env_factory, specs, *, on_episode_complete):
    return collect_evaluation(policy, env_factory, specs, benchmark="appworld",
                              on_episode_complete=on_episode_complete)


def collect_evaluation(policy, env_factory, specs, *, benchmark, on_episode_complete):
    return _collect_slots(policy, env_factory, specs, benchmark=benchmark, history_length=None,
                          ledger=policy.ledger, on_episode_complete=on_episode_complete)


def _collect_slots(policy, env_factory, specs, *, benchmark, history_length, ledger, on_episode_complete=None,
                   split_kv=False):
    scope = nullcontext()
    if getattr(policy, "paged_graph_enabled", False):
        row_count = sum(len(spec.slot_seeds) for spec in specs)
        root_count = row_count if on_episode_complete is not None else len(specs)
        scope = policy.paged_rollout(row_count, root_count)
    with scope:
        return _collect_slots_open(policy, env_factory, specs, benchmark=benchmark, history_length=history_length,
                                   ledger=ledger, on_episode_complete=on_episode_complete, split_kv=split_kv)


def _collect_slots_open(policy, env_factory, specs, *, benchmark, history_length, ledger,
                        on_episode_complete=None, split_kv=False):
    evaluation = on_episode_complete is not None
    paired = not evaluation and len(specs) > 1
    if benchmark not in HORIZONS or history_length not in (None, 2):
        raise ValueError("Select the declared F-GPO benchmark and full/two-turn filtration.")
    episodes = [(spec, slot, int(seed)) for spec in specs for slot, seed in enumerate(spec.slot_seeds)]
    slot_seeds = tuple(seed for _, _, seed in episodes)
    if not slot_seeds or len(set(slot_seeds)) != len(slot_seeds):
        raise ValueError("Each rollout slot requires its own sampling seed.")
    if ledger is not policy.ledger:
        raise ValueError("Actor and model forwards share the same actual cost ledger.")
    ledger.begin_group(specs[0].group_id if len(specs) == 1 else None)
    count, horizon, output_limit = len(slot_seeds), HORIZONS[benchmark], policy.max_new_tokens
    ids = tuple(f"{spec.group_id}:slot-{slot}" for spec, slot, _ in episodes)
    row_groups = tuple(spec.group_id for spec, _, _ in episodes)

    def details(row, **fields):
        spec, slot, _ = episodes[row]
        return {"group_id": spec.group_id, "slot": slot, "task_id": spec.task_id, **fields}

    environments = [None for _ in slot_seeds]
    contexts, initials, pins = [], [], []
    actor_views, reference_views, steps = [], [], [[] for _ in slot_seeds]
    official = [None for _ in slot_seeds]
    input_tokens, output_tokens, step_counts = [0] * count, [0] * count, [0] * count
    generators = [policy.new_generator(seed) for seed in slot_seeds]
    pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="fgpo-environment")
    tokenizer_lock = Lock()
    primary_error = None
    submitted_steps = []
    root, roots, cohorts = None, {}, []
    sampled_cohorts, next_cohorts = deque(), []
    cohort = sampled_parked = None

    def publish(slot, metrics, _):
        on_episode_complete(episodes[slot][0], EvaluationEpisode(ids[slot], metrics,
            step_counts[slot], input_tokens[slot], output_tokens[slot]))

    try:
        creation_results = _parallel_requests(pool, [
            (row, partial(env_factory, spec.task_id, spec.initial_seed, ids[row]),
             details(row, initial_seed=spec.initial_seed))
            for row, (spec, _, _) in enumerate(episodes)], ledger, "rollout.create")
        for slot, (request, _) in creation_results.items():
            if request.error is None:
                environments[slot] = request.value
        _raise_request_error(creation_results)
        reset_results = _parallel_requests(pool, [
            (row, partial(_reset_environment, environments[row], spec.task_id, spec.initial_seed),
             details(row, initial_seed=spec.initial_seed))
            for row, (spec, _, _) in enumerate(episodes)], ledger, "rollout.reset")
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
        if evaluation:
            root = policy.prefill_batch(pins, role="rollout.evaluation_roots")
            input_tokens = [len(pin) for pin in pins]
            cohorts = [_Cohort(tuple(range(count)), policy.fork_cache(root, copies=1))]
        else:
            start = 0
            for spec in specs:
                end = start + len(spec.slot_seeds)
                if any(pin != pins[start] for pin in pins[start + 1:end]):
                    raise ValueError("Same-root sampling requires the seeded public goal/bootstrap to agree within each group.")
                with ledger.context(group_id=spec.group_id):
                    roots[spec.group_id] = policy.prefill_root(pins[start], role="rollout.shared_root")
                    cohorts.append(_Cohort(tuple(range(start, end)),
                        policy.fork_cache(roots[spec.group_id], copies=end - start)))
                start = end
            cohorts = _merge_cohorts(policy, cohorts, history_length=history_length)
        for turn in range(horizon):
            sampled_cohorts = deque()

            def dispatch(slot, response):
                step_id = f"{ids[slot]}:step-{turn}"
                call = partial(_execute_response, policy.tokenizer, tokenizer_lock, environments[slot], benchmark,
                               response, str(contexts[slot]["instruction"]))
                receipt = details(slot, step_id=step_id, actual_step=True,
                                  timing_scope="parse_native_step_and_public_projection")
                submitted_steps.append((slot, receipt, pool.submit(_environment_request, call)))

            for cohort in cohorts:
                model_cost_start = len(ledger.records)
                groups = {row_groups[slot] for slot in cohort.slots} if paired else ()
                group_id = next(iter(groups)) if len(groups) == 1 else ledger.active_group_id
                with ledger.context(group_id=group_id), ledger.aggregate_model_calls():
                    sampled_parked, responses, probabilities = _sample_turn(
                        policy, cohort, generators, output_limit, on_response=dispatch, split_kv=split_kv)
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
            next_cohorts, finished = [], []
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
                    advances[slot] = advance
                    if evaluation:
                        # Only continuing sampled tokens enter KV. The final token
                        # enters through the actual append or replay on a live turn.
                        output_tokens[slot] += len(response)
                        input_tokens[slot] += len(response) - 1 + len(advance.delta_ids)
                        step_counts[slot] = turn + 1
                    else:
                        canonical = (advance if reference_views[slot] is actor_views[slot] else
                            reference_views[slot].append(step_id, response, observation, terminal=terminal))
                        steps[slot].append(StepRecord(step_id, ids[slot], turn, advance.response_span.segment_id,
                            advance.prompt_length, advance.response_span, advance.observation_span, response,
                            tuple(probabilities[local_index]), observation.pure_ids, observation.representation_ids,
                            observation.pure_text, observation.raw_observation, dict(action), float(result.reward),
                            bool(result.done), syntax_valid, advance.block, canonical.response_span.segment_id,
                            canonical.prompt_length, canonical.response_span, canonical.observation_span,
                            (environment_cost,), dict(result.info), model_event_ids))
                    if terminal:
                        finished.append(slot)
                if evaluation:
                    _finish_appworld_episodes(pool, environments, sorted(finished), details, ledger,
                        score_role="rollout.official_evaluation", close_role="rollout.close", on_complete=publish)
                    finished.clear()
                if paired:
                    next_cohorts.extend(_advance_paired_cohort(policy, roots, cohort, sampled_parked,
                                                               advances, row_groups, ledger))
                else:
                    original_root = root if evaluation else roots[specs[0].group_id]
                    next_cohorts.extend(_advance_cohort(policy, original_root, cohort, sampled_parked, advances,
                                                       history_length=history_length, independent_roots=evaluation))
                cohort = sampled_parked = None
            if not evaluation:
                evaluation_results = _parallel_requests(pool, [
                    (slot, environments[slot].finalize_evaluation, details(slot)) for slot in sorted(finished)],
                    ledger, "rollout.official_evaluation")
                _raise_request_error(evaluation_results)
                for slot, (request, _) in evaluation_results.items():
                    official[slot] = dict(request.value)
            if paired:
                continuing = {row_groups[slot] for pending in next_cohorts for slot in pending.slots}
                for group_id in roots.keys() - continuing:
                    policy.release_cache(roots.pop(group_id))
            cohorts = _merge_cohorts(policy, next_cohorts, history_length=history_length, ordered=paired)
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
                (slot, environment.close, details(slot))
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
                for owned in roots.values():
                    policy.release_cache(owned)
    if evaluation:
        return
    built = []
    for row, (spec, slot, sampling_seed) in enumerate(episodes):
        actor_segments = actor_views[row].segments()
        reference_segments = (actor_segments if reference_views[row] is actor_views[row]
                              else reference_views[row].segments())
        built.append(TrajectoryRecord(ids[row], spec.group_id, slot, str(spec.task_id), benchmark,
            int(spec.initial_seed), sampling_seed, str(contexts[row]["instruction"]),
            encode_text(policy.tokenizer, str(contexts[row]["instruction"])), initials[row].pure_ids,
            pins[row], tuple(steps[row]), actor_segments, reference_segments,
            bool(official[row]["success"]), official[row],
            tuple(ledger.records) if count == 1 else tuple(cost for cost in ledger.records
                if cost.kind == "environment" and cost.details.get("group_id") == spec.group_id
                and cost.details.get("slot") == slot),
            history_length))
    return tuple(built)


run_episodes_fgpo = collect_group
