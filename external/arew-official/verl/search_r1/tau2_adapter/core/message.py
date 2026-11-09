import json
from typing import Literal

from pydantic import BaseModel, Field

from .utils import get_now

ToolRequestor = Literal["assistant", "user"]


class ToolCall(BaseModel):
    id: str = Field(default="")
    name: str
    arguments: dict
    requestor: ToolRequestor = "assistant"

    def __str__(self) -> str:
        return f"{self.name}({json.dumps(self.arguments, ensure_ascii=False)})"


class ParticipantMessageBase(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    turn_idx: int | None = None
    timestamp: str | None = Field(default_factory=get_now)
    raw_data: dict | None = None

    def is_tool_call(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def has_text_content(self) -> bool:
        return self.content is not None and len(self.content.strip()) > 0

    def validate(self) -> None:
        if self.is_tool_call() and self.has_text_content():
            raise ValueError("Message cannot contain both text and tool calls.")
        if not self.is_tool_call() and not self.has_text_content():
            raise ValueError("Message cannot be empty.")


class AssistantMessage(ParticipantMessageBase):
    role: Literal["assistant"] = "assistant"


class UserMessage(ParticipantMessageBase):
    role: Literal["user"] = "user"


class ToolMessage(BaseModel):
    id: str = ""
    content: str
    requestor: ToolRequestor = "assistant"
    role: Literal["tool"] = "tool"
    turn_idx: int | None = None
    timestamp: str | None = Field(default_factory=get_now)
    error: bool = False
