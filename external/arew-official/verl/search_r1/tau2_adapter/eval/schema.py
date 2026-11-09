from pydantic import BaseModel, Field

from search_r1.tau2_adapter.core.tasks import Action, EnvAssertion


class DBCheck(BaseModel):
    db_match: bool
    db_reward: float


class ActionCheck(BaseModel):
    action: Action
    action_match: bool
    action_reward: float
    tool_type: str | None = None


class EnvAssertionCheck(BaseModel):
    env_assertion: EnvAssertion
    met: bool
    reward: float


class RewardInfo(BaseModel):
    reward: float
    db_check: DBCheck | None = None
    action_checks: list[ActionCheck] | None = None
    env_assertions: list[EnvAssertionCheck] | None = None
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    info: dict | None = None
