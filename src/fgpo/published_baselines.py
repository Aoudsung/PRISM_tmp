# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Portions adapted from verl-agent, Nala-YN/G2PO, and AgentQ0/ECPO under
# the Apache License, Version 2.0.
# See THIRD_PARTY_NOTICES.md and THIRD_PARTY_LICENSE_APACHE_2_0.txt.
# Changes: immutable rollout records, scalar CPU reductions, public-history
# adapter, and unit-edge reverse BFS; no verl/Ray/model dependency is required.
"""Credit and state adapters for published comparator recipes.

Algorithm definitions follow the source snapshot in baseline_specs, including
source/paper differences recorded there.  Every graph edge and hierarchy entry
comes from an existing rollout row; these estimators never sample extra actions.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
import json
import re

import numpy as np
import torch

from .baseline_specs import PUBLISHED_RECIPES
from .public import encode_text, pinned_prompt, project_public_observation


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


def textcraft_prompt_ids(tokenizer, system_prompt, initial_raw, current_raw, memory, history_length, *, schema, goal):
    """Common public TextCraft input with the released two-pair actor memory."""
    observation = project_public_observation(tokenizer, "textcraft", current_raw,
                                             action_schema=schema, goal=goal, initial=not memory,
                                             published_baseline=True)
    if memory:
        task_context = json.loads(initial_raw)["feedback"]
        recent = memory[-history_length:] if history_length else []
        history = "\n".join(
            f"[Observation {step}: '{json.loads(raw)['feedback']}', Action {step}: '{action}']"
            for step, (raw, action) in enumerate(recent, len(memory) - len(recent) + 1))
        prefix = (f"Task context (constant across the episode):\n{task_context}\n\n"
                  f"Prior to this step, you have already taken {len(memory)} step(s). "
                  f"Below are the most recent {len(recent)} observations and the corresponding actions you took:\n"
                  f"{history}\n\nYou are now at step {len(memory) + 1} and your current observation is:\n")
        observation = replace(observation, user_body_ids=encode_text(tokenizer, prefix) + observation.user_body_ids)
    return pinned_prompt(tokenizer, system_prompt, observation)


class G2POALFWorldAnchors:
    """Track the released ALF state keys independently of the actor prompt."""

    def __init__(self, initial_prompt):
        self.history = deque((initial_prompt,), maxlen=2)

    def next_anchor(self, observation, commands):
        if not observation.startswith("Nothing happens"):
            admissible = "\n ".join(f"'{command}'" for command in commands if command != "help")
            self.history.append(f"{observation}\nAdmissible actions:\n {admissible}\n")
        if len(self.history) == 2 and self.history[0].startswith(("You heat", "You cool", "You clean", "You turn on")):
            return " ".join(self.history)
        return self.history[-1]


def g2po_appworld_prompt(goal, supervisor, memory, history_length):
    """Render the released G2PO user prompt from projected code and raw output."""
    values = {f"supervisor_{key}": supervisor[key]
              for key in ("first_name", "last_name", "email", "phone_number")}
    if not memory:
        return (_G2PO_APPWORLD_INTRO + _G2PO_APPWORLD_INITIAL).format(task_description=goal, **values)
    recent = memory[-history_length:] if history_length else ()
    start = len(memory) - len(recent) + 1
    history = "".join(f"\nCode {index}: \n{code}\n\nResult {index}: \n{observation}\n"
                      for index, (observation, code) in enumerate(recent, start))
    if len(history) > 27000:
        history = "... " + history[-27000:]
    return (_G2PO_APPWORLD_INTRO + _G2PO_APPWORLD_HISTORY).format(
        task_description=goal, step_count=len(memory), history_length=len(recent),
        action_history=history.strip(), current_step=len(memory) + 1, **values)


def g2po_appworld_action(text):
    """Project released code tags and keep format validity separate from execution."""
    start, end = text.find("<code>"), text.find("</code>")
    has_code = start != -1 and end != -1
    code = text[start + len("<code>"):end] if has_code else text[-100:]
    valid = has_code and "<think>" in text and "</think>" in text
    action = {"kind": "code", "code": code}
    if "..." in code:
        action["author_error"] = "Execution failed. You are not allowed to use ellipsis in your code."
    elif "except" in code.lower():
        action["author_error"] = "Execution failed. You are not allowed to use try-except in your code."
    return action, valid


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


def _g2po_node_values(group, config):
    """Form observation nodes and successor values from original rollout rows."""
    totals, counts = defaultdict(float), defaultdict(int)
    anchors_by_trajectory = []
    gamma, scale = float(config["gamma"]), float(config["reward_scale"])
    for trajectory in group:
        anchors = [row.environment_info["baseline_anchor"] for row in trajectory.steps]
        terminal = scale * float(trajectory.official_success)
        anchors_by_trajectory.append((anchors, terminal))
        for depth, anchor in enumerate(anchors):
            totals[anchor] += gamma ** (len(anchors) - depth) * terminal
            counts[anchor] += 1
    node_values = {anchor: total / counts[anchor] for anchor, total in totals.items()}
    current, successor = [], []
    for anchors, terminal in anchors_by_trajectory:
        if anchors:
            values = [node_values[anchor] for anchor in anchors]
            current.extend(values)
            successor.extend(values[1:])
            successor.append(terminal)
    return torch.tensor(current, dtype=torch.float64), torch.tensor(successor, dtype=torch.float64)


def _g2po_advantages(rows, current, successor, config):
    """Normalize successor values by node and raw increments by task group."""
    clusters = defaultdict(list)
    for i, row in enumerate(rows):
        clusters[row.environment_info["baseline_anchor"]].append(i)
    mode, epsilon = config["mode"], config["epsilon"]
    local_values = successor - torch.tensor([
        float(config["syntax_reward_penalty"]) * (not row.syntax_valid) for row in rows], dtype=torch.float64)
    local_credit = torch.zeros(len(rows), dtype=torch.float32)
    for members in clusters.values():
        if len(members) > 1:
            indices = torch.tensor(members, dtype=torch.long)
            local_credit[indices] = _normalize(local_values[indices], mode, epsilon,
                                               unbiased=config["step_std_unbiased"]).float()
    edge_credit = _normalize((successor - current).float(), mode, epsilon, unbiased=config["step_std_unbiased"])
    return local_credit + edge_credit


_ECPO_STOP_WORDS = frozenset(("a", "an", "the", "to", "at", "into", "on", "in"))
_ECPO_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9 ]")


def _ecpo_advantages(rows, returns, config):
    """Released ECA and direct variance reliability on adjusted step rows."""
    clusters = defaultdict(list)
    for i, row in enumerate(rows):
        clusters[row.environment_info["baseline_anchor"]].append(i)
    result = torch.zeros_like(returns)
    returns = returns.numpy().astype(np.float64)
    epsilon, kappa = float(config["epsilon"]), float(config["ecpo_kappa"])
    for members in clusters.values():
        if len(members) == 1:
            continue
        buckets = defaultdict(list)
        for i in members:
            response = rows[i].environment_info["baseline_response_text"]
            tokens = _ECPO_NON_ALPHANUMERIC.sub("", response.lower()).split()
            action = " ".join(token for token in tokens if token not in _ECPO_STOP_WORDS)
            buckets[action].append(i)
        anchor_returns = returns[members]
        anchor_mean = anchor_returns.mean()
        action_returns = [returns[indices] for indices in buckets.values()]
        shrunk = np.array([(len(values) / (len(values) + kappa)) * values.mean()
                           + (kappa / (len(values) + kappa)) * anchor_mean for values in action_returns])
        centered = shrunk - anchor_mean
        spread = centered.std() if len(buckets) > 1 else 1.
        if spread < epsilon:
            continue
        rho = 1.
        if np.mean([rows[i].index for i in members]) >= config["ecpo_d_min"]:
            repeated_variances = [values.var() for values in action_returns if len(values) >= 2]
            # The release assigns unit between-action variance when every action bucket is a singleton.
            between = shrunk.var() if repeated_variances else 1.
            within = np.mean(repeated_variances) if repeated_variances else 0.
            sample_weight = 1. - np.exp(-(len(members) - 1) / config["ecpo_tau"])
            rho = min(float(sample_weight * between / (between + within + epsilon)), 1.)
            if len(buckets) >= 2 and anchor_returns.std() > epsilon:
                rho = max(rho, config["ecpo_rho_min"])
        for indices, advantage in zip(buckets.values(), centered / (spread + epsilon), strict=True):
            result[indices] = float(advantage)
        result[members] *= float(rho)
    return result


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
    if method in ("g2po", "ecpo") and credit_row_ids is None:
        raise ValueError(f"Published {method} credit requires the adjusted rollout row order")
    adjust_credit_rows = settings["source_copy_before_advantage"] and credit_row_ids is not None
    if adjust_credit_rows:
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
            if method != "g2po":
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
        if method == "g2po":
            current, successor = _g2po_node_values(group, settings)
        elif method == "graphgpo":
            returns = _graph_returns(rows, success_steps, settings) - penalty
        else:
            returns = torch.tensor(returns, dtype=torch.float32) - penalty
        if adjust_credit_rows:
            # Copied rows retain their pre-copy value inputs and enter all final statistics.
            positions = {row.step_id:index for index,row in enumerate(rows)}
            indices = [positions[step_id] for step_id in copied_groups[group[0].group_id]]
            selection = torch.tensor(indices,dtype=torch.long)
            rows = [rows[index] for index in indices]
            episodes = episodes[selection]
            if method == "g2po":
                current, successor = current[selection], successor[selection]
            else:
                returns = returns[selection]
        if method == "hgpo":
            advantages = _hierarchy_advantages(group, rows, returns, episodes, settings)
        else:
            episode = _normalize(episodes, settings["mode"], settings["epsilon"],
                                 unbiased=settings["episode_std_unbiased"], episode=True)
            if method == "g2po":
                step = _g2po_advantages(rows, current, successor, settings)
            elif method == "ecpo":
                step = _ecpo_advantages(rows, returns, settings)
            else:
                step = _state_advantages(rows, returns, settings)
            advantages = float(settings["episode_advantage_w"]) * episode + float(settings["step_advantage_w"]) * step
        output.update((row.step_id, value) for row, value in zip(rows, advantages.tolist(), strict=True))
    return output


# Source prompt adapted from Nala-YN/G2PO, agent_system/environments/prompts/appworld.py.
# The released initial and subsequent templates share this worked example.
_G2PO_APPWORLD_INTRO = """
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

The code will be executed in an interactive shell. Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.
-----------------------------
Here is an example:

My name is: supervisor_first_name supervisor_last_name. My personal email is supervisor_email and phone number is supervisor_phone_number.

Your task is: What is the password for my Spotify account?

Code 1:
print(apis.api_docs.show_app_descriptions())

Result 1:\x20
[
  {{
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  }},
  {{
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  }},
  ...
  {{
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  }},
  {{
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  }},
  ...
]

Code 2:
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

Result 2:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]

Code 3:
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Result 3:
{{
  'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {{
    'success': [{{'account_name': 'string', 'password': 'string'}}],
    'failure': {{'message': 'string'}}
  }}
}}

Code 4:
print(apis.supervisor.show_account_passwords())

Result 4:
[
  {{
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  }},
  {{
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  }},
  ...
]

Code 5:
# So the Spotify password is an entry in the `passwords` list with the account_name=spotify.
spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)

Result 5:
dummy_spotify_pass

Code 6:
# When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.
apis.supervisor.complete_task(answer=spotify_password)

Result 6:
Marked the active task complete.
-----------------------------

Key Instructions and Disclaimers:
1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current date or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
15. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
16. Once you believe the task is complete, you MUST call `apis.supervisor.complete_task()` to finalize it. If the task requires an answer, provide it using the answer argument — for example, `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, either omit the argument. The task will not end automatically — it will remain open until you explicitly make this call.

Using these APIs, now begin writing code cells step-by-step to solve the actual task:

My name is: {supervisor_first_name} {supervisor_last_name}. My personal email is {supervisor_email} and phone number is {supervisor_phone_number}.

Your task is: {task_description}

"""

_G2PO_APPWORLD_INITIAL = """Now it's your turn to generate code to solve the task.
You should first reason step-by-step about which APIs to call, what arguments to use, and how to build your code block to complete the task. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you present the solution code body within <code> </code> tags.
"""

_G2PO_APPWORLD_HISTORY = """Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} codes you generated and the corresponding environment return:\x20
{action_history}

Now you are at step {current_step} and it's your turn to generate code for this step.
First, you MUST carefully reflect on the history of interactions and the most recent error messages. Then, reason about what should be done next, which APIs to call, what arguments to use, and how to build your code block to complete the task. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reflexion and reasoning, you present the solution code body within <code> </code> tags.

"""
