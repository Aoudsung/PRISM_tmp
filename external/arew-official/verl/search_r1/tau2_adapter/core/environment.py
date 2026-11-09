import json
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from .message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from .tasks import EnvAssertion, EnvFunctionCall, InitializationData
from .tool import Tool
from .toolkit import ToolKitBase


class EnvironmentInfo(BaseModel):
    domain_name: str = Field(description="The name of the domain.")
    policy: str = Field(description="The policy of the agent.")


class Environment:
    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: ToolKitBase | None = None,
        user_tools: ToolKitBase | None = None,
        solo_mode: bool = False,
    ):
        self.domain_name = domain_name
        self.policy = policy
        self.tools = tools
        self.user_tools = user_tools
        self.solo_mode = solo_mode
        if self.solo_mode:
            self.validate_solo_mode()
        self.sync_tools()

    def get_domain_name(self) -> str:
        return self.domain_name

    def get_policy(self) -> str:
        return self.policy

    def get_tools(self) -> list[Tool]:
        if self.tools is None:
            raise ValueError("Tools not available")
        return list(self.tools.get_tools().values())

    def get_user_tools(self, include: list[str] | None = None) -> list[Tool]:
        if self.user_tools is None:
            raise ValueError("User tools not available")
        return list(self.user_tools.get_tools(include=include).values())

    def get_tools_description(self, env_type: str) -> str | None:
        toolkit = self.user_tools if env_type == "user" else self.tools
        if toolkit is None:
            return None
        tools = sorted(toolkit.get_tools().values(), key=lambda item: item.name)
        return "\n\n".join(f"{idx + 1}. {tool.to_str()}" for idx, tool in enumerate(tools))

    def _has_tool(self, tool_name: str) -> bool:
        return (self.tools is not None and self.tools.has_tool(tool_name)) or (
            self.user_tools is not None and self.user_tools.has_tool(tool_name)
        )

    def _is_mutating_tool(self, tool_name: str) -> bool:
        if self.tools is not None and self.tools.has_tool(tool_name):
            return self.tools.tool_mutates_state(tool_name)
        if self.user_tools is not None and self.user_tools.has_tool(tool_name):
            return self.user_tools.tool_mutates_state(tool_name)
        return True

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        if self.tools is None:
            raise ValueError("Tools not available")
        return self.tools.use_tool(tool_name=tool_name, **kwargs)

    def use_user_tool(self, tool_name: str, **kwargs) -> Any:
        if self.user_tools is None:
            raise ValueError("User tools not available")
        return self.user_tools.use_tool(tool_name=tool_name, **kwargs)

    def make_tool_call(self, tool_name: str, requestor: str = "assistant", **kwargs) -> Any:
        if requestor == "user":
            if self.solo_mode:
                raise ValueError("User tool calls are not allowed in solo mode")
            return self.use_user_tool(tool_name=tool_name, **kwargs)
        if requestor == "assistant":
            if self.solo_mode and self.user_tools is not None and self.user_tools.has_tool(tool_name):
                return self.use_user_tool(tool_name=tool_name, **kwargs)
            return self.use_tool(tool_name=tool_name, **kwargs)
        raise ValueError(f"Invalid requestor: {requestor}")

    def sync_tools(self):
        pass

    def run_env_function_call(self, env_function_call: EnvFunctionCall) -> Any:
        toolkit = self.user_tools if env_function_call.env_type == "user" else self.tools
        if toolkit is None:
            raise ValueError(f"{env_function_call.env_type} toolkit unavailable")
        func = getattr(toolkit, env_function_call.func_name)
        result = func(**env_function_call.arguments)
        self.sync_tools()
        return result

    def run_env_assertion(self, assertion: EnvAssertion, raise_assertion_error: bool = True) -> bool:
        result = self.run_env_function_call(assertion)
        if not isinstance(result, bool):
            raise ValueError(f"Assertion {assertion.func_name} returned non-bool: {type(result)}")
        passed = result == assertion.assert_value
        if raise_assertion_error:
            assert passed, assertion.message or f"Assertion failed: {assertion}"
        return passed

    def run_env_function_calls(self, env_function_calls: list[EnvFunctionCall]) -> None:
        for call in env_function_calls:
            if isinstance(call, EnvAssertion):
                self.run_env_assertion(call, raise_assertion_error=True)
            else:
                self.run_env_function_call(call)

    def get_db_hash(self) -> str | None:
        return None if self.tools is None else self.tools.get_db_hash()

    def get_user_db_hash(self) -> str | None:
        return None if self.user_tools is None else self.user_tools.get_db_hash()

    def set_state(
        self,
        initialization_data: InitializationData | None,
        initialization_actions: list[EnvFunctionCall] | None,
        message_history: list[AssistantMessage | UserMessage | ToolMessage],
    ):
        if self.solo_mode:
            assert all(not isinstance(message, UserMessage) for message in message_history), (
                "User messages are not allowed in solo mode"
            )

        if initialization_data is not None:
            if initialization_data.agent_data is not None and self.tools is not None:
                self.tools.update_db(initialization_data.agent_data)
                if (
                    self.user_tools is not None
                    and self.user_tools.db is not None
                    and type(self.user_tools.db) is type(self.tools.db)
                ):
                    self.user_tools.db = self.tools.db
            if initialization_data.user_data is not None and self.user_tools is not None:
                self.user_tools.update_db(initialization_data.user_data)
                if (
                    self.tools is not None
                    and self.tools.db is not None
                    and type(self.tools.db) is type(self.user_tools.db)
                ):
                    self.tools.db = self.user_tools.db

        if initialization_actions is not None:
            for action in initialization_actions:
                self.run_env_function_call(action)

        history = list(message_history)
        idx = 0
        while idx < len(history):
            message = history[idx]
            if isinstance(message, ToolMessage):
                raise ValueError("Tool message not expected without a preceding tool call.")
            if isinstance(message, (AssistantMessage, UserMessage)) and message.is_tool_call():
                for tool_call in message.tool_calls:
                    idx += 1
                    if idx >= len(history) or not isinstance(history[idx], ToolMessage):
                        raise ValueError("Tool message expected after tool call.")
                    expected = history[idx]
                    if self._has_tool(tool_call.name) and self._is_mutating_tool(tool_call.name):
                        response = self.get_response(tool_call)
                        if self._normalize_content(response.content) != self._normalize_content(expected.content):
                            raise ValueError(
                                "Tool replay mismatch.\n"
                                f"Tool call: {tool_call}\nReturned: {response.content}\nExpected: {expected.content}"
                            )
            idx += 1

        self.sync_tools()

    @staticmethod
    def _normalize_content(content: Any) -> Any:
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
        return content

    @classmethod
    def to_json_str(cls, resp: Any) -> str:
        def _process(obj: Any) -> Any:
            if isinstance(obj, BaseModel):
                return obj.model_dump()
            if isinstance(obj, (list, tuple)):
                return [_process(item) for item in obj]
            if isinstance(obj, dict):
                return {key: _process(value) for key, value in obj.items()}
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            return obj

        if isinstance(resp, str):
            return resp
        return json.dumps(_process(resp), default=str)

    def validate_solo_mode(self) -> None:
        assistant_names = set(self.tools.get_tools().keys()) if self.tools is not None else set()
        user_names = set(self.user_tools.get_tools().keys()) if self.user_tools is not None else set()
        overlap = assistant_names & user_names
        if overlap:
            raise ValueError(f"Tool names overlap: {sorted(overlap)}")

    def set_solo_mode(self, solo_mode: bool):
        self.solo_mode = solo_mode
        if solo_mode:
            self.validate_solo_mode()

    def get_response(self, message: ToolCall) -> ToolMessage:
        error = False
        try:
            response = self.make_tool_call(message.name, requestor=message.requestor, **message.arguments)
            self.sync_tools()
        except Exception as exc:
            response = f"Error: {exc}"
            error = True
        return ToolMessage(
            id=message.id,
            content=self.to_json_str(response),
            requestor=message.requestor,
            error=error,
        )
