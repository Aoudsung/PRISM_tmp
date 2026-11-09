from search_r1.tau2_adapter.core.toolkit import ToolKitBase, ToolType, is_tool
from search_r1.tau2_adapter.domains.mock.data_model import MockDB, Task, TaskStatus, User


class MockTools(ToolKitBase):
    db: MockDB

    def __init__(self, db: MockDB):
        super().__init__(db)

    @is_tool(ToolType.WRITE)
    def create_task(self, user_id: str, title: str, description: str | None = None) -> Task:
        """Create a new task for a user."""
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        task_id = f"task_{len(self.db.tasks) + 1}"
        task = Task(task_id=task_id, title=title, description=description, status="pending")
        self.db.tasks[task_id] = task
        self.db.users[user_id].tasks.append(task_id)
        return task

    @is_tool(ToolType.READ)
    def get_users(self) -> list[User]:
        """Get all users in the database."""
        return list(self.db.users.values())

    @is_tool(ToolType.WRITE)
    def update_task_status(self, task_id: str, status: TaskStatus) -> Task:
        """Update the status of a task."""
        if task_id not in self.db.tasks:
            raise ValueError(f"Task {task_id} not found")
        self.db.tasks[task_id].status = status
        return self.db.tasks[task_id]

    def assert_number_of_tasks(self, user_id: str, expected_number: int) -> bool:
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        return len(self.db.users[user_id].tasks) == expected_number

    def assert_task_status(self, task_id: str, expected_status: TaskStatus) -> bool:
        if task_id not in self.db.tasks:
            raise ValueError(f"Task {task_id} not found")
        return self.db.tasks[task_id].status == expected_status

    @is_tool(ToolType.GENERIC, mutates_state=False)
    def transfer_to_human_agents(self, summary: str) -> str:
        """Transfer the user to a human agent when the task cannot be solved with tools."""
        return "Transfer successful"
