from search_r1.tau2_adapter.core.message import AssistantMessage, UserMessage
from search_r1.tau2_adapter.loader.registry import get_tasks_loader


def is_valid_solo_task(task) -> bool:
    if task.ticket is None:
        return False
    if task.evaluation_criteria is None:
        return False
    has_supported_reward = bool(task.evaluation_criteria.actions or task.evaluation_criteria.env_assertions)
    if not has_supported_reward:
        return False
    if task.initial_state is None or task.initial_state.message_history is None:
        return True
    for message in task.initial_state.message_history:
        if isinstance(message, UserMessage):
            return False
        if isinstance(message, AssistantMessage) and not message.is_tool_call():
            return False
    return True


def load_tasks(task_set_name: str, task_split_name: str | None = None, solo_only: bool = False) -> list:
    tasks = get_tasks_loader(task_set_name)(task_split_name=task_split_name)
    if solo_only:
        tasks = [task for task in tasks if is_valid_solo_task(task)]
    return tasks


def get_tasks(
    task_set_name: str,
    task_split_name: str | None = None,
    task_ids: list[str] | None = None,
    num_tasks: int | None = None,
    solo_only: bool = False,
) -> list:
    tasks = load_tasks(task_set_name=task_set_name, task_split_name=task_split_name, solo_only=solo_only)
    if task_ids is not None:
        tasks = [task for task in tasks if task.id in task_ids]
        if len(tasks) != len(task_ids):
            missing = set(task_ids) - {task.id for task in tasks}
            raise ValueError(f"Missing task ids: {sorted(missing)}")
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks
