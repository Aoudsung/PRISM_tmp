"""One fixed-reference window: packed response laws, causal states and TF-BEFP.

Approved finite-context extension: pin the initial pure observation and retain
whole recent observation blocks after overflow. Global posterior frontier never
resets. Packed ALF external surprise includes the newly disclosed command list;
that conditioning is explicitly different from a history/action-only formula.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


@dataclass
class ReferenceStep:
    token_logprobs: tuple[float,...]
    z_hist: torch.Tensor | None = None
    z_obs: torch.Tensor | None = None
    external_surprise: tuple[float,...] = ()
    goal_gain: tuple[float,...] = ()
    eta: float = 0.
    potential_before: float = 0.
    potential_after: float = 0.
    reward: float = 0.


@dataclass
class ReferenceRound:
    steps: dict[str,ReferenceStep] = field(default_factory=dict)
    windows: dict[str,list[dict]] = field(default_factory=dict)
    conditioning: str = "packed-public-frames-including-new-ALF-command-list"
    computed: dict[str,bool] = field(default_factory=dict)


def sampled_logprobs(policy, hidden, target, *, role, chunk_tokens=32):
    values = []
    for start in range(0,len(target),chunk_tokens):
        end = min(start+chunk_tokens,len(target))
        logits = policy.head(hidden[start:end],role=role)
        ids = torch.tensor(target[start:end],dtype=torch.long,device=logits.device)
        values.extend((-F.cross_entropy(logits,ids,reduction="none")).detach().cpu().tolist())
    return tuple(values)


def _observation_segments(policy,trajectory):
    blocks = [tuple(trajectory.initial_observation_ids)]+[tuple(s.observation_ids) for s in trajectory.steps]
    goal = tuple(trajectory.goal_ids)
    if not blocks[0] or not goal:
        raise ValueError("Official initial observation and public task must have causal reference tokens")
    prompt_limit = policy.max_context_tokens-policy.max_new_tokens
    capacity = min(prompt_limit,policy.max_context_tokens-len(goal))
    low_water = min(int(.75*prompt_limit),policy.max_context_tokens-len(goal))
    if len(blocks[0]) > capacity:
        raise ValueError("Pinned initial observation leaves no approved atomic reference window")
    selected = [0]
    stream = list(blocks[0])
    states = [(0,len(stream),0)]
    chunks = []
    for state in range(1,len(blocks)):
        block = blocks[state]
        if len(stream)+len(block) > capacity:
            if len(blocks[0]) > low_water:
                raise ValueError("Pinned pure observation cannot fit the approved 75-percent overflow window")
            chunks.append((tuple(stream),states,tuple(selected)))
            retained,used = [],len(blocks[0])
            for index in range(state,0,-1):
                if used+len(blocks[index]) > low_water:
                    break
                retained.append(index)
                used += len(blocks[index])
            selected = [0]+list(reversed(retained))
            if state not in selected:
                raise ValueError("One complete pure observation cannot fit its approved bounded reference window")
            stream = [token for i in selected for token in blocks[i]]
            start = len(stream)-len(block)
            states = [(state,len(stream),start)]
        else:
            start = len(stream)
            stream.extend(block)
            selected.append(state)
            states.append((state,len(stream),start))
    chunks.append((tuple(stream),states,tuple(selected)))
    return blocks,goal,chunks


@torch.no_grad()
def pure_observation_reference(policy,trajectory):
    blocks,goal,chunks = _observation_segments(policy,trajectory)
    posteriors,delta,windows = {},{},[]
    for stream,states,selected in chunks:
        obs_hidden,_,cache = policy.body(stream,use_cache=True,teacher_forced=True,role="reference.pure_observation_stream")
        goal_hidden,_,_ = policy.body(goal+stream,role="reference.goal_observation_stream")
        for state,length,start in states:
            if state and blocks[state]:
                n = len(blocks[state])
                unconditional = sampled_logprobs(policy,obs_hidden[0,start-1:start-1+n],blocks[state],
                                                 role="reference.observation_law")
                conditional = sampled_logprobs(policy,goal_hidden[0,len(goal)+start-1:len(goal)+start-1+n],blocks[state],
                                               role="reference.goal_conditioned_observation_law")
                delta[state] = tuple(c-u for c,u in zip(conditional,unconditional))
            elif state:
                delta[state] = ()
            windows.append({"state":state,"retained_observation_indices":tuple(i for i in selected if i<=state),
                "window_ids":stream[:length],"observation_start":start,"observation_length":len(blocks[state]),
                "goal_ids":goal})
        # Decreasing prefixes erase all previous goal-query suffixes before the
        # next query; no future observation is reachable through this cache.
        previous_length,previous_value = None,None
        for state,length,_ in reversed(states):
            if length == previous_length:
                posteriors[state] = previous_value
                continue
            cache.crop(length-1)
            query = (stream[length-1],)+goal[:-1]
            mask = torch.ones((1,length-1+len(query)),dtype=torch.long,device=policy.device)
            hidden,_,cache = policy.body(query,cache=cache,attention_mask=mask,use_cache=True,
                teacher_forced=True,cache_position=torch.arange(length-1,length-1+len(query),device=policy.device),
                role="reference.goal_posterior")
            value = sum(sampled_logprobs(policy,hidden[0],goal,role="reference.goal_posterior_head"))
            posteriors[state] = value
            previous_length,previous_value = length,value
    return posteriors,delta,windows


@torch.no_grad()
def prepare_round(policy,trajectories,*,method="fgpo"):
    """Caller activates the fixed reference exactly once for this whole round."""
    result = ReferenceRound()
    own = method.startswith("fgpo")
    befp = own and method != "fgpo_no_befp"
    result.computed = {"policy_reference_logprobs":True,"z_hist":own,"z_obs":own,
                      "external_surprise":befp,"goal_gain":befp,"goal_posterior":befp}
    for trajectory in trajectories:
        views = {s.segment_id:s for s in (*trajectory.segments,*trajectory.full_reference_segments)}
        needed = {row.response_span.segment_id for row in trajectory.steps}
        if befp:
            needed.update(row.full_reference_observation_span.segment_id for row in trajectory.steps)
        cache = {}
        for segment_id in needed:
            segment = views[segment_id]
            key = segment.token_ids if befp else segment.token_ids[:segment.behavior_closed_length]
            if key not in cache:
                hidden,layer = policy.packed_segment_hidden(key,hidden_layer=-8 if own else None,
                                                            role="reference.packed_segment")
                cache[key] = (hidden,layer)
            hidden,layer = cache[key]
            for row in trajectory.steps:
                response_span = row.response_span
                if response_span.segment_id == segment_id:
                    states = hidden[0,response_span.start-1:response_span.stop-1]
                    lp = sampled_logprobs(policy,states,row.response_ids,role="reference.response_head")
                    hist = F.normalize(layer[0,row.prompt_length-1].float(),dim=0).cpu() if own else None
                    result.steps[row.step_id] = ReferenceStep(lp,z_hist=hist)
            # External surprise is assigned in a separate pass so actor/feedback
            # segment iteration order never determines whether its row exists.
        if own:
            if befp:
                for row in trajectory.steps:
                    span = row.full_reference_observation_span
                    segment = views[span.segment_id]
                    hidden,_ = cache[segment.token_ids]
                    scores = sampled_logprobs(policy,hidden[0,span.start-1:span.stop-1],row.observation_ids,
                                             role="reference.external_surprise")
                    result.steps[row.step_id].external_surprise = tuple(-x for x in scores)
            representations = []
            previous = trajectory.initial_observation_ids
            for row in trajectory.steps:
                representations.append(tuple(previous) or policy.encode("[Empty stdout]"))
                previous = row.observation_repr_ids
            # Batches are representation computation only, with the same target
            # layer, last valid token and normalization for each independent row.
            for start in range(0,len(representations),8):
                values = policy.observation_embeddings(representations[start:start+8])
                for row,value in zip(trajectory.steps[start:start+8],values):
                    result.steps[row.step_id].z_obs = value
            if befp:
                posteriors,gains,windows = pure_observation_reference(policy,trajectory)
                result.windows[trajectory.trajectory_id] = windows
                h0,frontier,potential = -posteriors[0],posteriors[0],0.
                for t,row in enumerate(trajectory.steps):
                    data = result.steps[row.step_id]
                    increment = max(0.,posteriors[t+1]-frontier)
                    frontier = max(frontier,posteriors[t+1])
                    data.goal_gain = gains[t+1]
                    bottleneck = sum(min(max(g,0.),s) for g,s in zip(data.goal_gain,data.external_surprise))
                    data.eta = min(increment,bottleneck) if row.observation_ids else 0.
                    data.potential_before = potential
                    potential += data.eta/(h0+1.)
                    data.potential_after = potential
            for t,row in enumerate(trajectory.steps):
                data = result.steps[row.step_id]
                data.reward = (10.*float(trajectory.official_success) if t==len(trajectory.steps)-1 else 0.) + data.potential_after-data.potential_before
    return result
