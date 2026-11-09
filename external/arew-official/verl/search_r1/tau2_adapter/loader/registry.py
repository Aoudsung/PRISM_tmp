from search_r1.tau2_adapter.domains.mock.environment import get_environment as mock_get_environment
from search_r1.tau2_adapter.domains.mock.environment import get_tasks as mock_get_tasks
from search_r1.tau2_adapter.domains.telecom.environment import get_environment as telecom_get_environment
from search_r1.tau2_adapter.domains.telecom.environment import get_tasks as telecom_get_tasks

_ENV_REGISTRY = {
    "mock": mock_get_environment,
    "telecom": telecom_get_environment,
}

_TASK_REGISTRY = {
    "mock": mock_get_tasks,
    "telecom": telecom_get_tasks,
}


def get_env_constructor(name: str):
    if name not in _ENV_REGISTRY:
        raise KeyError(f"Domain {name} not found in tau2-lite registry")
    return _ENV_REGISTRY[name]


def get_tasks_loader(name: str):
    if name not in _TASK_REGISTRY:
        raise KeyError(f"Task set {name} not found in tau2-lite registry")
    return _TASK_REGISTRY[name]
