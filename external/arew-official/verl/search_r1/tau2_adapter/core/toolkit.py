from enum import Enum
from typing import Any, Callable

from .db import DB
from .pydantic_utils import update_pydantic_model_with_dict
from .tool import Tool, as_tool
from .utils import get_dict_hash

TOOL_ATTR = "__tool__"
TOOL_TYPE_ATTR = "__tool_type__"
MUTATES_STATE_ATTR = "__mutates_state__"


class ToolType(str, Enum):
    READ = "read"
    WRITE = "write"
    THINK = "think"
    GENERIC = "generic"


def is_tool(tool_type: ToolType = ToolType.READ, mutates_state: bool | None = None):
    if mutates_state is None:
        mutates_state = tool_type == ToolType.WRITE

    def decorator(func: Callable[..., Any]):
        setattr(func, TOOL_ATTR, True)
        setattr(func, TOOL_TYPE_ATTR, tool_type)
        setattr(func, MUTATES_STATE_ATTR, mutates_state)
        return func

    return decorator


class ToolKitType(type):
    def __init__(cls, name, bases, attrs):
        func_tools = {}
        for key, method in attrs.items():
            if hasattr(method, TOOL_ATTR):
                func_tools[key] = method

        @property
        def _func_tools(self):
            inherited = {}
            for base in bases:
                inherited.update(getattr(base, "_declared_tools", {}))
            inherited.update(func_tools)
            return inherited

        cls._declared_tools = func_tools
        cls._func_tools = _func_tools
        super().__init__(name, bases, attrs)


class ToolKitBase(metaclass=ToolKitType):
    def __init__(self, db: DB | None = None):
        self.db = db

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        return {name: getattr(self, name) for name in self._func_tools.keys()}

    def get_tools(self, include: list[str] | None = None) -> dict[str, Tool]:
        tools = {name: as_tool(tool) for name, tool in self.tools.items()}
        if include is None:
            return tools
        unknown = set(include) - set(tools.keys())
        if unknown:
            raise ValueError(f"Tool(s) not found: {sorted(unknown)}")
        return {name: tools[name] for name in include}

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return self.tools[tool_name](**kwargs)

    def tool_type(self, tool_name: str) -> ToolType:
        return getattr(self.tools[tool_name], TOOL_TYPE_ATTR, ToolType.GENERIC)

    def tool_mutates_state(self, tool_name: str) -> bool:
        return getattr(self.tools[tool_name], MUTATES_STATE_ATTR, True)

    def update_db(self, update_data: dict[str, Any] | None = None):
        if self.db is None:
            raise ValueError("Database has not been initialized.")
        self.db = update_pydantic_model_with_dict(self.db, update_data or {})

    def get_db_hash(self) -> str | None:
        if self.db is None:
            return None
        return get_dict_hash(self.db.model_dump())
