import textwrap
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from .message import AssistantMessage, ToolMessage, UserMessage

ToolRequestor = Literal["assistant", "user"]


class StructuredUserInstructions(BaseModel):
    domain: str
    reason_for_call: str
    known_info: str | None = None
    unknown_info: str | None = None
    task_instructions: str

    def __str__(self) -> str:
        lines = [f"Domain: {self.domain}", f"Reason for call:\n{textwrap.indent(self.reason_for_call, '  ')}"]
        if self.known_info is not None:
            lines.append(f"Known info:\n{textwrap.indent(self.known_info, '  ')}")
        if self.unknown_info is not None:
            lines.append(f"Unknown info:\n{textwrap.indent(self.unknown_info, '  ')}")
        lines.append(f"Task instructions:\n{textwrap.indent(self.task_instructions, '  ')}")
        return "\n".join(lines)


class UserScenario(BaseModel):
    persona: str | None = None
    instructions: str | StructuredUserInstructions

    def __str__(self) -> str:
        lines = []
        if self.persona:
            lines.append("Persona:")
            lines.append(textwrap.indent(self.persona, "  "))
        lines.append("Instructions:")
        lines.append(textwrap.indent(str(self.instructions), "  "))
        return "\n".join(lines)


class Description(BaseModel):
    purpose: str | None = None
    relevant_policies: str | None = None
    notes: str | None = None


class Action(BaseModel):
    action_id: str
    requestor: ToolRequestor = "assistant"
    name: str
    arguments: dict
    info: str | None = None
    compare_args: list[str] | None = None

    def compare_with_tool_call(self, tool_call) -> bool:
        if self.name != tool_call.name:
            return False
        keys = self.compare_args if self.compare_args is not None else tool_call.arguments.keys()
        return {k: tool_call.arguments.get(k) for k in keys} == {k: self.arguments.get(k) for k in keys}


class EnvFunctionCall(BaseModel):
    env_type: ToolRequestor
    func_name: str
    arguments: dict


class EnvAssertion(EnvFunctionCall):
    assert_value: bool = True
    message: str | None = None


class RewardType(str, Enum):
    DB = "DB"
    ENV_ASSERTION = "ENV_ASSERTION"
    ACTION = "ACTION"
    COMMUNICATE = "COMMUNICATE"


class EvaluationCriteria(BaseModel):
    actions: list[Action] | None = None
    env_assertions: list[EnvAssertion] | None = None
    communicate_info: list[str] | None = None
    nl_assertions: list[str] | None = None
    reward_basis: list[RewardType] = Field(default_factory=lambda: [RewardType.DB, RewardType.COMMUNICATE])


class InitializationData(BaseModel):
    agent_data: dict | None = None
    user_data: dict | None = None


class InitialState(BaseModel):
    initialization_data: InitializationData | None = None
    initialization_actions: list[EnvFunctionCall] | None = None
    message_history: list[AssistantMessage | UserMessage | ToolMessage] | None = None


class Task(BaseModel):
    id: str
    description: Description | None = None
    user_scenario: UserScenario
    ticket: str | None = None
    initial_state: InitialState | None = None
    evaluation_criteria: EvaluationCriteria | None = None
    user_tools: list[str] | None = None
