from typing import Literal

from pydantic import Field

from search_r1.tau2_adapter.core.db import DB
from search_r1.tau2_adapter.core.pydantic_utils import BaseModelNoExtra

NotificationStatus = Literal["unread", "read"]


class Notification(BaseModelNoExtra):
    notification_id: str = Field(description="Unique identifier for the notification")
    message: str = Field(description="The notification message")
    status: NotificationStatus = Field(default="unread", description="Notification status")
    task_id: str | None = Field(default=None, description="Associated task ID")


class MockUserDB(DB):
    notifications: dict[str, Notification] = Field(default_factory=dict)
