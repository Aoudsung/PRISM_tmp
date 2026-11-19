"""Resource widths and stable identities for AppWorld computational batches."""
from dataclasses import dataclass
from operator import index


def _width(config, key, maximum, *, default=None):
    value = index(config.get(key, maximum if default is None else default))
    if not 1 <= value <= maximum:
        raise ValueError(f"{key} must be between 1 and {maximum}")
    return value


def appworld_eval_batch_size(config):
    return _width(config, "appworld_eval_batch_size", 8) if config["benchmark"] == "appworld" else 1


def appworld_hgpo_rollout_groups(config):
    if config["benchmark"] == "appworld" and config["method"] == "hgpo":
        return _width(config, "appworld_hgpo_rollout_groups", 2)
    return 1


def appworld_fgpo_rollout_groups(config):
    if config["benchmark"] == "appworld" and config["method"] == "fgpo":
        width = _width(config, "rollout_computational_groups", 2, default=1)
        if width == 2 and (int(config.get("trajectories", 8)) != 8 or config.get("history_length") is not None):
            raise ValueError("Paired AppWorld collection requires the eight-slot full-history F-GPO recipe")
        return width
    return 1


@dataclass(frozen=True)
class RolloutGroupSpec:
    group_id: str
    task_id: str
    initial_seed: int
    slot_seeds: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationEpisode:
    trajectory_id: str
    official_evaluation: dict
    steps: int
    input_tokens: int
    output_tokens: int
