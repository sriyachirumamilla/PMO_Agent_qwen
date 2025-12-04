
from langchain.tools import tool
from src.models import Task, Project, TaskStatus, TaskPriority, TASKS_DB, PROJECTS_DB
from datetime import datetime
import uuid


@tool
def create_task(
    title: str,
    description: str,
    project_id: str = "default",
    priority: str = "medium"
) -> str:
    """Create a new task in a project.
    
    Args:
        title: Task title (short, clear)
        description: Detailed task description
        project_id: Project ID (defaults to 'default')
        priority: Task priority (low, medium, high, critical)
    
    Returns:
        Confirmation with task ID
    """
    task_id = f"TASK-{str(uuid.uuid4())[:8]}"
    
    try:
        priority_enum = TaskPriority(priority.lower())
    except ValueError:
        priority_enum = TaskPriority.MEDIUM
    
    task = Task(
        id=task_id,
        title=title,
        description=description,
        priority=priority_enum,
    )
    
    TASKS_DB[task_id] = task
    
    return f"✓ Task created: {task_id}\n  Title: {title}\n  Priority: {priority}"


@tool
def assign_task(task_id: str, assignee: str) -> str:
    """Assign a task to a team member.
    
    Args:
        task_id: ID of the task
        assignee: Name of team member to assign
    
    Returns:
        Confirmation message
    """
    if task_id not in TASKS_DB:
        return f"✗ Task {task_id} not found. Available: {list(TASKS_DB.keys())}"
    
    TASKS_DB[task_id].assigned_to = assignee
    return f"✓ Task {task_id} assigned to {assignee}"


@tool
def update_task_status(task_id: str, new_status: str) -> str:
    """Update task status.
    
    Args:
        task_id: ID of the task
        new_status: New status (pending, in_progress, completed, blocked)
    
    Returns:
        Confirmation message
    """
    if task_id not in TASKS_DB:
        return f"✗ Task {task_id} not found"
    
    try:
        status_enum = TaskStatus(new_status.lower())
    except ValueError:
        return f"✗ Invalid status. Use: pending, in_progress, completed, blocked"
    
    old_status = TASKS_DB[task_id].status
    TASKS_DB[task_id].status = status_enum
    
    return f"✓ Task {task_id} status updated: {old_status} → {new_status}"


@tool
def list_tasks(status_filter: str = None, assignee_filter: str = None) -> str:
    """List all tasks with optional filters.
    
    Args:
        status_filter: Filter by status (optional)
        assignee_filter: Filter by assignee (optional)
    
    Returns:
        Formatted task list
    """
    if not TASKS_DB:
        return "📋 No tasks found. Create one with 'create_task'."
    
    tasks = list(TASKS_DB.values())
    
    # Apply filters
    if status_filter:
        tasks = [t for t in tasks if t.status.value == status_filter.lower()]
    if assignee_filter:
        tasks = [t for t in tasks if t.assigned_to == assignee_filter]
    
    if not tasks:
        return "📋 No tasks match the filters."
    
    result = "📋 **TASK LIST**\n"
    result += "=" * 70 + "\n"
    
    for i, task in enumerate(tasks, 1):
        status_icon = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "blocked": "🚫"
        }.get(task.status.value, "?")
        
        priority_icon = {
            "low": "🔵",
            "medium": "🟡",
            "high": "🔴",
            "critical": "🔴🔴"
        }.get(task.priority.value, "?")
        
        assigned = task.assigned_to or "Unassigned"
        
        result += f"\n{i}. {status_icon} [{task.id}] {task.title}\n"
        result += f"   └─ Assigned: {assigned} | Priority: {priority_icon} {task.priority.value}\n"
        result += f"      {task.description[:60]}...\n" if len(task.description) > 60 else f"      {task.description}\n"
    
    result += "\n" + "=" * 70
    return result


@tool
def get_task_details(task_id: str) -> str:
    """Get detailed information about a specific task.
    
    Args:
        task_id: ID of the task
    
    Returns:
        Detailed task information
    """
    if task_id not in TASKS_DB:
        return f"✗ Task {task_id} not found"
    
    task = TASKS_DB[task_id]
    
    result = f"""
📌 **TASK DETAILS**
{'=' * 50}
ID:          {task.id}
Title:       {task.title}
Description: {task.description}
Status:      {task.status.value.upper()}
Assigned To: {task.assigned_to or 'Unassigned'}
Priority:    {task.priority.value.upper()}
Created:     {task.created_at.strftime('%Y-%m-%d %H:%M')}
Due Date:    {task.due_date.strftime('%Y-%m-%d') if task.due_date else 'No due date'}
Tags:        {', '.join(task.tags) if task.tags else 'None'}
{'=' * 50}
"""
    return result


@tool
def get_statistics() -> str:
    """Get project statistics.
    
    Returns:
        Statistics summary
    """
    if not TASKS_DB:
        return "📊 No tasks yet. Create one to see statistics."
    
    tasks = list(TASKS_DB.values())
    total = len(tasks)
    
    by_status = {}
    for task in tasks:
        status = task.status.value
        by_status[status] = by_status.get(status, 0) + 1
    
    by_assignee = {}
    for task in tasks:
        assignee = task.assigned_to or "Unassigned"
        by_assignee[assignee] = by_assignee.get(assignee, 0) + 1
    
    result = f"""
📊 **PROJECT STATISTICS**
{'=' * 50}
Total Tasks:      {total}

By Status:
"""
    for status, count in by_status.items():
        result += f"  • {status}: {count}\n"
    
    result += "\nBy Assignee:\n"
    for assignee, count in by_assignee.items():
        result += f"  • {assignee}: {count}\n"
    
    result += f"{'=' * 50}"
    return result


# Export all tools
tools = [
    create_task,
    assign_task,
    update_task_status,
    list_tasks,
    get_task_details,
    get_statistics,
]