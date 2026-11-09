from search_r1.tau2_adapter.core.environment import Environment
from search_r1.tau2_adapter.core.io_utils import load_file
from search_r1.tau2_adapter.core.tasks import Task
from search_r1.tau2_adapter.domains.mock.data_model import MockDB
from search_r1.tau2_adapter.domains.mock.tools import MockTools
from search_r1.tau2_adapter.domains.mock.user_data_model import MockUserDB
from search_r1.tau2_adapter.domains.mock.user_tools import MockUserTools
from search_r1.tau2_adapter.domains.mock.utils import (
    MOCK_DB_PATH,
    MOCK_POLICY_PATH,
    MOCK_POLICY_SOLO_PATH,
    MOCK_TASK_SET_PATH,
    MOCK_TASK_SPLIT_PATH,
    MOCK_USER_DB_PATH,
)


def get_environment(db: MockDB | None = None, user_db: MockUserDB | None = None, solo_mode: bool = False) -> Environment:
    db = db or MockDB.load(str(MOCK_DB_PATH))
    user_db = user_db or MockUserDB.load(str(MOCK_USER_DB_PATH))
    policy = load_file(MOCK_POLICY_SOLO_PATH if solo_mode else MOCK_POLICY_PATH)
    env = Environment(
        domain_name="mock",
        policy=policy,
        tools=MockTools(db),
        user_tools=MockUserTools(user_db),
    )
    if solo_mode:
        env.set_solo_mode(True)
    return env


def get_tasks(task_split_name: str | None = None) -> list[Task]:
    tasks = [Task.model_validate(task) for task in load_file(MOCK_TASK_SET_PATH)]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(f"Invalid task split name: {task_split_name}. Valid splits are: {sorted(task_splits.keys())}")
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    return load_file(MOCK_TASK_SPLIT_PATH)
