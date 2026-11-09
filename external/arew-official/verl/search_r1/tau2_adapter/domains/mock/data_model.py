from typing import Literal

from pydantic import BaseModel, Field

from search_r1.tau2_adapter.core.db import DB

TaskStatus = Literal["pending", "completed"]


class Task(BaseModel):
    task_id: str = Field(description="Unique identifier for the task")
    title: str = Field(description="Title of the task")
    description: str | None = Field(default=None, description="Description of the task")
    status: TaskStatus = Field(description="Status of the task")


class User(BaseModel):
    user_id: str = Field(description="Unique identifier for the user")
    name: str = Field(description="User's name")
    tasks: list[str] = Field(description="List of task IDs assigned to the user")


class MockDB(DB):
    tasks: dict[str, Task]
    users: dict[str, User]
