"""Task selection in the released GraphGPO WebShop driver."""

TRAIN_ROUNDS = 60
TRAIN_GROUPS = 16
VALIDATION_WORKERS = 128
VALIDATION_SESSIONS = 500


def task_schedule(split, seed, goal_count):
    import numpy as np

    if split == "train":
        first, stop, rounds, workers = VALIDATION_SESSIONS, goal_count, TRAIN_ROUNDS, TRAIN_GROUPS
    elif split == "test":
        first, stop, rounds, workers = 0, VALIDATION_SESSIONS, 1, VALIDATION_WORKERS
        seed += 1000
    else:
        raise ValueError(f"Unknown WebShop split: {split!r}")
    if goal_count < stop or stop - first < workers:
        raise ValueError(f"Official WebShop catalog has {goal_count} goals for {split}")
    # The numeric session index selects from the goal order of its assigned worker.
    rng = np.random.RandomState(seed)
    return [
        {"task_id": f"{split}/{seed + slot}/{index}", "split": split,
         "environment_seed": seed + slot, "session_index": int(index)}
        for _ in range(rounds)
        for slot, index in enumerate(rng.choice(range(first, stop), size=workers, replace=False))
    ]
