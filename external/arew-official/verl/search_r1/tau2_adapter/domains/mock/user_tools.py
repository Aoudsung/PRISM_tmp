from search_r1.tau2_adapter.core.toolkit import ToolKitBase, ToolType, is_tool
from search_r1.tau2_adapter.domains.mock.user_data_model import MockUserDB, Notification


class MockUserTools(ToolKitBase):
    db: MockUserDB

    def __init__(self, db: MockUserDB):
        super().__init__(db)

    @is_tool(ToolType.READ)
    def check_notifications(self) -> list[Notification]:
        """Check all notifications in the user's inbox."""
        return list(self.db.notifications.values())

    @is_tool(ToolType.WRITE)
    def dismiss_notification(self, notification_id: str) -> str:
        """Dismiss a notification."""
        if notification_id not in self.db.notifications:
            raise ValueError(f"Notification {notification_id} not found")
        self.db.notifications[notification_id].status = "read"
        return f"Notification {notification_id} dismissed"

    def add_notification(self, notification_id: str, message: str, task_id: str | None = None) -> Notification:
        notification = Notification(notification_id=notification_id, message=message, task_id=task_id)
        self.db.notifications[notification_id] = notification
        return notification

    def assert_notification_status(self, notification_id: str, expected_status: str) -> bool:
        if notification_id not in self.db.notifications:
            raise ValueError(f"Notification {notification_id} not found")
        return self.db.notifications[notification_id].status == expected_status
