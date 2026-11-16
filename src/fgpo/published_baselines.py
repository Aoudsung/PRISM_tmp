# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Portions adapted from verl-agent under the Apache License, Version 2.0.
# See THIRD_PARTY_NOTICES.md and THIRD_PARTY_LICENSE_APACHE_2_0.txt.
# Changes: immutable rollout records, scalar CPU reductions, public-history
# adapter, and unit-edge reverse BFS; no verl/Ray/model dependency is required.
"""Credit and ALFWorld state adapters for three published comparator recipes.

Algorithm definitions follow the source snapshot in baseline_specs, including
source/paper differences recorded there.  Every graph edge and hierarchy entry
comes from an existing rollout row; these estimators never sample extra actions.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import re

import torch

from .baseline_specs import PUBLISHED_RECIPES


def published_alfworld_action(text):
    """Released alfworld_projection semantics, including its format penalty.

    Admissible-action membership is an environment diagnostic.  The upstream
    reward penalty instead checks action/think tags and Chinese characters.
    """
    lowered = text.lower()
    start, end = lowered.find("<action>"), lowered.find("</action>")
    if start == -1 or end == -1:
        return {"kind": "native_text", "text": lowered[-30:]}, False
    action = lowered[start + len("<action>"):end].strip()
    valid = ("<think>" in text and "</think>" in text
             and re.search(r"[\u4e00-\u9fff]", text) is None)
    return {"kind": "native_text", "text": action}, valid


def alfworld_prompt(observation, commands, memory, goal, history_length):
    """Exact released ALFWorld template and SimpleMemory formatting."""
    commands = "\n ".join(f"'{command}'" for command in commands if command != "help")
    recent = memory[-history_length:] if history_length else []
    if memory:
        first = len(memory) - len(recent) + 1
        context = "\n".join(
            f"[Observation {first + index}: '{obs}', Action {first + index}: '{action}']"
            for index, (obs, action) in enumerate(recent))
        prefix = (
            f"\nYou are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {goal}\n"
            f"Prior to this step, you have already taken {len(memory)} step(s). Below are the most recent "
            f"{len(recent)} observations and the corresponding actions you took: {context}\n"
            f"You are now at step {len(memory) + 1} and your current observation is: {observation}\n")
    else:
        prefix = ("\nYou are an expert agent operating in the ALFRED Embodied Environment.\n"
                  f"Your current observation is: {observation}\n")
    return (prefix + f"Your admissible actions of the current situation are: [{commands}].\n\n"
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation. This reasoning process MUST "
            "be enclosed within <think> </think> tags. \n"
            "Once you've finished your reasoning, you should choose an admissible action for current step "
            "and present it within <action> </action> tags.\n")


@dataclass
class GraphALFWorldHistory:
    """Released GraphGPO summary computed solely from public action/feedback.

    Mirrors AlfworldWorker's state updates and get_location_holding.  This is
    baseline-specific actor memory; it never accesses a private simulator state.
    """
    location: str = "middle of a room"
    holding: str = "nothing"
    item_states: dict = field(default_factory=dict)
    item_locations: dict = field(default_factory=dict)

    def update(self, action, observation):
        if "Nothing happens" in observation:
            return
        action = action.lower().strip()
        match = re.match(r"^(heat|cool|clean|slice)\s+([\w\s\d]+?)(?:\s+with\s+[\w\s\d]+)?$", action)
        if match and match.group(1) in observation:
            verb, item = match.group(1), match.group(2).strip()
            states = self.item_states.setdefault(item, {"heated": False, "cooled": False,
                                                       "cleaned": False, "slice": False})
            states[{"heat": "heated", "cool": "cooled", "clean": "cleaned", "slice": "sliced"}[verb]] = True
        move = re.match(r"^go to\s+(.+)$", action)
        if move:
            target = move.group(1).strip()
            if target not in observation:
                raise ValueError("GraphGPO public location update disagrees with ALFWorld feedback")
            self.location = target
        previous = self.holding
        take = re.match(r"^take\s+(.+?)(?:\s+from\s+.+)?$", action)
        if take:
            if self.holding != "nothing":
                raise ValueError("GraphGPO public inventory already holds an item")
            if take.group(1).strip() in observation:
                self.holding = take.group(1).strip()
        drop = re.match(r"^drop\s+(.+)$", action)
        if drop and drop.group(1).strip() in observation:
            self.holding = "nothing"
        put = re.match(r"^(put|place)\s+(.+?)\s+(?:in|on)\s+.+$", action)
        if put and put.group(2).strip() in observation:
            self.holding = "nothing"
        move_item = re.match(r"^move\s+(.+?)(?:\s+to\s+.+)?$", action)
        if move_item:
            item = move_item.group(1).strip()
            if self.holding != item:
                raise ValueError("GraphGPO move action disagrees with public inventory")
            if item in observation:
                self.holding = "nothing"
        if previous != self.holding:
            item = self.holding if previous == "nothing" else previous
            if item not in self.item_locations:
                if self.holding != item:
                    raise ValueError("GraphGPO first item-location record requires a public take action")
                self.item_locations[item] = {"old_location": self.location, "new_location": self.location}
            elif previous == item:
                self.item_locations[item]["new_location"] = self.location
            elif self.holding == item:
                self.item_locations[item]["new_location"] = self.item_locations[item]["old_location"]

    def observation_and_anchor(self, observation, commands):
        held_status = "".join(f"{state} " for item, states in self.item_states.items()
                              if item in self.holding for state, active in states.items() if active)
        if held_status:
            held_status = f"({held_status.strip()})"
        elif self.holding != "nothing":
            held_status = "(unprocessed)"
        moving = []
        for item, positions in self.item_locations.items():
            state = "".join(f"{name} " for name, active in self.item_states.get(item, {}).items() if active)
            state = f"({state.strip()})" if state else "(unprocessed)"
            if (positions["old_location"] != positions["new_location"]
                    or (state != "(unprocessed)" and self.holding != item)):
                moving.append(f"{item}{state} from {positions['old_location']} move to {positions['new_location']};")
        moving_text = "History moving list by you: " + "".join(moving) if moving else ""
        location = f"Location: {self.location}. Items in hand (status): {self.holding}{held_status}."
        actor_observation = f"{observation} {location} {moving_text}"
        return actor_observation, f"{actor_observation} {','.join(sorted(commands))}"


def _normalize(values, mode, epsilon, *, unbiased, episode=False):
    """Float32 normalization with the source's singleton conventions."""
    if mode not in ("mean_std_norm", "mean_norm"):
        raise ValueError(f"Unknown published advantage normalization {mode!r}")
    if values.numel() == 1:
        mean = torch.zeros_like(values[0]) if episode else values[0]
        deviation = torch.ones_like(values[0])
    else:
        mean = values.mean()
        deviation = values.std(unbiased=unbiased)
    centered = values - mean
    return centered / (deviation + epsilon) if mode == "mean_std_norm" else centered


def _state_advantages(rows, returns, config):
    clusters = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[row.environment_info["baseline_anchor"]].append(index)
    result = torch.zeros_like(returns)
    for members in clusters.values():
        indices = torch.tensor(members, dtype=torch.long)
        result[indices] = _normalize(returns[indices], config["mode"], config["epsilon"],
                                     unbiased=config["step_std_unbiased"])
    return result


def _hierarchy_advantages(group, rows, returns, episodes, config):
    """Released HGPO grouping/weighting, including zero-item exclusion."""
    clusters = defaultdict(list)
    offset = 0
    for trajectory in group:
        anchors = [row.environment_info["baseline_anchor"] for row in trajectory.steps]
        for step in range(len(anchors)):
            for length in range(1, min(int(config["history"]) + 1, step + 1) + 1):
                key = length, tuple(anchors[step - length + 1:step + 1])
                clusters[key].append(offset + step)
        offset += len(anchors)
    numerator = torch.zeros_like(returns)
    denominator = torch.zeros_like(returns)
    for (length, _), members in clusters.items():
        if len(members) <= 1:
            continue
        indices = torch.tensor(members, dtype=torch.long)
        advantage = _normalize(returns[indices], config["mode"], config["epsilon"], unbiased=False)
        weight = float(length + 1) ** float(config["length_weight_alpha"])
        valid = advantage != 0
        numerator[indices] += weight * advantage
        denominator[indices] += weight * valid
    if config["base_group"]:
        base = _normalize(episodes, config["mode"], config["epsilon"], unbiased=True, episode=True)
        numerator += base
        denominator += (base != 0)
    return numerator / (denominator + float(config["epsilon"]))


def _graph_returns(rows, successful_last_steps, config):
    """Unit-edge reverse BFS is equivalent to the released reverse Dijkstra.

    Parallel edges retain separate training rows.  Their shared adjacency is
    stored once, and paths are never materialized because only distances enter
    the official estimator.
    """
    incoming, nodes, edges = defaultdict(set), {"success"}, []
    for row in rows:
        source = row.environment_info["baseline_anchor"]
        target = "success" if row.step_id in successful_last_steps else row.environment_info["baseline_next_anchor"]
        incoming[target].add(source)
        nodes.update((source, target))
        edges.append((source, target))
    distances, queue = {"success": 0}, deque(("success",))
    while queue:
        target = queue.popleft()
        for source in incoming[target]:
            if source not in distances:
                distances[source] = distances[target] + 1
                queue.append(source)
    unreachable = max(distances.values()) + 1
    for state in nodes:
        distances.setdefault(state, unreachable)
    gamma, scale = float(config["gamma"]), float(config["reward_scale"])
    progress = bool(config["normalize_distance"])
    return torch.tensor([
        scale * gamma ** (distances[target] - distances[source] + 1 if progress else distances[target])
        for source, target in edges], dtype=torch.float32)


def published_advantages(trajectories, method, *, config=None, credit_row_ids=None):
    """Compute released-source scalar advantages over existing task groups."""
    settings = {**PUBLISHED_RECIPES[method], **({} if config is None else config)}
    if settings["enable_similarity"]:
        raise ValueError("The selected published recipes use exact state matching")
    if settings["episode_statistics"] != "steps":
        raise ValueError("Released source recipe requires episode statistics over step rows")
    groups = defaultdict(list)
    step_groups = {}
    for trajectory in trajectories:
        groups[trajectory.group_id].append(trajectory)
        step_groups.update((row.step_id,trajectory.group_id) for row in trajectory.steps)
    copied_groups = defaultdict(list)
    if method == "gigpo" and credit_row_ids is not None:
        for step_id in credit_row_ids:
            copied_groups[step_groups[step_id]].append(step_id)
    output = {}
    for group in groups.values():
        rows = [row for trajectory in group for row in trajectory.steps]
        if not rows:
            continue
        penalty = torch.tensor([float(settings["syntax_reward_penalty"]) * (not row.syntax_valid)
                                for row in rows], dtype=torch.float32)
        episode_scores, returns, success_steps = [], [], set()
        for trajectory in group:
            terminal = float(settings["reward_scale"]) * float(trajectory.official_success)
            episode_scores.extend([terminal] * len(trajectory.steps))
            running, discounted = 0., []
            for index in reversed(range(len(trajectory.steps))):
                running = (terminal if index == len(trajectory.steps) - 1 else 0.) + float(settings["gamma"]) * running
                discounted.append(running)
            returns.extend(reversed(discounted))
            if trajectory.official_success and trajectory.steps:
                success_steps.add(trajectory.steps[-1].step_id)
        # The released trainer applies the format penalty here, after return
        # computation.  Future invalid-action penalties are not discounted back.
        episodes = torch.tensor(episode_scores, dtype=torch.float32) - penalty
        if method == "graphgpo":
            returns = _graph_returns(rows, success_steps, settings) - penalty
        else:
            returns = torch.tensor(returns, dtype=torch.float32) - penalty
        if method == "gigpo" and credit_row_ids is not None:
            # GiGPO's released trainer copies/pads and balances before its
            # episode/state statistics. Every copied row retains its original
            # discounted return and current-row penalty, including multiplicity.
            positions = {row.step_id:index for index,row in enumerate(rows)}
            indices = [positions[step_id] for step_id in copied_groups[group[0].group_id]]
            selection = torch.tensor(indices,dtype=torch.long)
            rows = [rows[index] for index in indices]
            episodes,returns = episodes[selection],returns[selection]
        if method == "hgpo":
            advantages = _hierarchy_advantages(group, rows, returns, episodes, settings)
        else:
            episode = _normalize(episodes, settings["mode"], settings["epsilon"],
                                 unbiased=settings["episode_std_unbiased"], episode=True)
            step = _state_advantages(rows, returns, settings)
            advantages = float(settings["episode_advantage_w"]) * episode + float(settings["step_advantage_w"]) * step
        output.update((row.step_id, value) for row, value in zip(rows, advantages.tolist(), strict=True))
    return output
