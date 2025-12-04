"""
Data models for PMO task management.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    """Task priority enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    """Represents a single project task."""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = Field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class Project(BaseModel):
    """Represents a project with multiple tasks."""
    id: str
    name: str
    description: str
    owner: str
    tasks: List[Task] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING

    class Config:
        use_enum_values = True


# In-memory storage (replace with database in production)
TASKS_DB: dict = {}
PROJECTS_DB: dict = {}