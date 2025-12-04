"""
Unit tests for PMO Agent and Tools.

Run tests with:
    pytest tests/test_agent.py -v
    
Or with coverage:
    pytest tests/test_agent.py -v --cov=src
"""

import pytest
from src.models import Task, TaskStatus, TaskPriority, TASKS_DB, PROJECTS_DB
from src.tools import (
    create_task,
    assign_task,
    update_task_status,
    list_tasks,
    get_task_details,
    get_statistics,
)
from datetime import datetime


# ============================================================================
# FIXTURES (Setup/Teardown)
# ============================================================================

@pytest.fixture(autouse=True)
def clear_databases():
    """Clear databases before and after each test."""
    TASKS_DB.clear()
    PROJECTS_DB.clear()
    yield
    TASKS_DB.clear()
    PROJECTS_DB.clear()


# ============================================================================
# TESTS FOR TASK CREATION
# ============================================================================

class TestTaskCreation:
    """Test task creation functionality."""
    
    def test_create_task_basic(self):
        """Test creating a basic task."""
        result = create_task(
            title="Implement authentication",
            description="Add user login system"
        )
        
        assert "✓ Task created" in result
        assert len(TASKS_DB) == 1
        
        task = list(TASKS_DB.values())[0]
        assert task.title == "Implement authentication"
        assert task.status == TaskStatus.PENDING
    
    def test_create_task_with_priority(self):
        """Test creating task with priority."""
        result = create_task(
            title="Fix critical bug",
            description="Database connection issue",
            priority="critical"
        )
        
        assert "✓ Task created" in result
        task = list(TASKS_DB.values())[0]
        assert task.priority == TaskPriority.CRITICAL
    
    def test_create_multiple_tasks(self):
        """Test creating multiple tasks."""
        for i in range(3):
            create_task(
                title=f"Task {i+1}",
                description=f"Description {i+1}"
            )
        
        assert len(TASKS_DB) == 3
    
    def test_task_has_unique_id(self):
        """Test that each task gets unique ID."""
        create_task(title="Task 1", description="Desc 1")
        create_task(title="Task 2", description="Desc 2")
        
        task_ids = list(TASKS_DB.keys())
        assert len(set(task_ids)) == 2
        assert all("TASK-" in tid for tid in task_ids)


# ============================================================================
# TESTS FOR TASK ASSIGNMENT
# ============================================================================

class TestTaskAssignment:
    """Test task assignment functionality."""
    
    def test_assign_task_to_person(self):
        """Test assigning task to a person."""
        create_task(title="Write documentation", description="API docs")
        task_id = list(TASKS_DB.keys())[0]
        
        result = assign_task(task_id, "Alice")
        
        assert "✓" in result
        assert TASKS_DB[task_id].assigned_to == "Alice"
    
    def test_assign_nonexistent_task(self):
        """Test assigning non-existent task (error handling)."""
        result = assign_task("TASK-FAKE", "Bob")
        
        assert "✗" in result or "not found" in result.lower()
    
    def test_reassign_task(self):
        """Test reassigning task to different person."""
        create_task(title="Code review", description="Review PR")
        task_id = list(TASKS_DB.keys())[0]
        
        assign_task(task_id, "Alice")
        assert TASKS_DB[task_id].assigned_to == "Alice"
        
        assign_task(task_id, "Bob")
        assert TASKS_DB[task_id].assigned_to == "Bob"


# ============================================================================
# TESTS FOR TASK STATUS
# ============================================================================

class TestTaskStatus:
    """Test task status updates."""
    
    def test_update_task_status(self):
        """Test updating task status."""
        create_task(title="Deploy app", description="To production")
        task_id = list(TASKS_DB.keys())[0]
        
        result = update_task_status(task_id, "in_progress")
        
        assert "✓" in result
        assert TASKS_DB[task_id].status == TaskStatus.IN_PROGRESS
    
    def test_update_to_completed(self):
        """Test marking task as completed."""
        create_task(title="Write tests", description="Unit tests")
        task_id = list(TASKS_DB.keys())[0]
        
        update_task_status(task_id, "completed")
        
        assert TASKS_DB[task_id].status == TaskStatus.COMPLETED
    
    def test_invalid_status(self):
        """Test invalid status update."""
        create_task(title="Task", description="Desc")
        task_id = list(TASKS_DB.keys())[0]
        
        result = update_task_status(task_id, "invalid_status")
        
        assert "✗" in result or "Invalid" in result


# ============================================================================
# TESTS FOR TASK LISTING
# ============================================================================

class TestTaskListing:
    """Test task listing functionality."""
    
    def test_list_tasks_empty(self):
        """Test listing when no tasks exist."""
        result = list_tasks()
        
        assert "No tasks" in result or "empty" in result.lower()
    
    def test_list_all_tasks(self):
        """Test listing all tasks."""
        for i in range(3):
            create_task(title=f"Task {i+1}", description=f"Desc {i+1}")
        
        result = list_tasks()
        
        assert "TASK-" in result
        assert len(TASKS_DB) == 3
    
    def test_list_tasks_by_status(self):
        """Test filtering tasks by status."""
        create_task(title="Task 1", description="D1")
        task_id = list(TASKS_DB.keys())[0]
        
        update_task_status(task_id, "completed")
        create_task(title="Task 2", description="D2")
        
        result = list_tasks(status_filter="completed")
        
        assert "Task 1" in result
    
    def test_list_tasks_by_assignee(self):
        """Test filtering tasks by assignee."""
        create_task(title="Task 1", description="D1")
        task_id_1 = list(TASKS_DB.keys())[0]
        
        create_task(title="Task 2", description="D2")
        task_id_2 = list(TASKS_DB.keys())[1]
        
        assign_task(task_id_1, "Alice")
        assign_task(task_id_2, "Bob")
        
        result = list_tasks(assignee_filter="Alice")
        
        assert "Task 1" in result


# ============================================================================
# TESTS FOR TASK DETAILS
# ============================================================================

class TestTaskDetails:
    """Test task details retrieval."""
    
    def test_get_task_details(self):
        """Test getting details of a task."""
        create_task(
            title="Deploy service",
            description="Deploy to AWS",
            priority="high"
        )
        task_id = list(TASKS_DB.keys())[0]
        
        result = get_task_details(task_id)
        
        assert "Deploy service" in result
        assert "AWS" in result
        assert "HIGH" in result
    
    def test_get_details_nonexistent(self):
        """Test getting details of non-existent task."""
        result = get_task_details("TASK-FAKE")
        
        assert "✗" in result or "not found" in result.lower()


# ============================================================================
# TESTS FOR STATISTICS
# ============================================================================

class TestStatistics:
    """Test statistics functionality."""
    
    def test_statistics_empty(self):
        """Test statistics when no tasks."""
        result = get_statistics()
        
        assert "No tasks" in result
    
    def test_statistics_with_tasks(self):
        """Test statistics with tasks."""
        create_task(title="Task 1", description="D1")
        task_id = list(TASKS_DB.keys())[0]
        update_task_status(task_id, "in_progress")
        
        create_task(title="Task 2", description="D2")
        
        result = get_statistics()
        
        assert "Total Tasks" in result
        assert "in_progress" in result or "1" in result


# ============================================================================
# TESTS FOR DATA MODELS
# ============================================================================

class TestDataModels:
    """Test Pydantic data models."""
    
    def test_task_model_creation(self):
        """Test creating a Task model instance."""
        task = Task(
            id="TASK-001",
            title="Test task",
            description="Test description"
        )
        
        assert task.id == "TASK-001"
        assert task.title == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM
    
    def test_task_model_validation(self):
        """Test Task model validation."""
        task = Task(
            id="TASK-002",
            title="Another task",
            description="Description",
            priority="high"
        )
        
        assert task.priority == TaskPriority.HIGH
    
    def test_task_with_assignee(self):
        """Test Task model with assignee."""
        task = Task(
            id="TASK-003",
            title="Task with assignee",
            description="Assigned task",
            assigned_to="Alice"
        )
        
        assert task.assigned_to == "Alice"
    
    def test_task_with_tags(self):
        """Test Task model with tags."""
        task = Task(
            id="TASK-004",
            title="Tagged task",
            description="Task with tags",
            tags=["backend", "urgent"]
        )
        
        assert len(task.tags) == 2
        assert "backend" in task.tags
    
    def test_project_model_creation(self):
        """Test creating a Project model instance."""
        project = project(
            id="PROJECT-001",
            name="Test Project",
            description="A test project",
            owner="Project Manager"
        )
        
        assert project.id == "PROJECT-001"
        assert project.name == "Test Project"
        assert len(project.tasks) == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple operations."""
    
    def test_full_task_workflow(self):
        """Test complete task workflow."""
        # Create task
        result = create_task(
            title="Complete workflow test",
            description="Full integration test",
            priority="high"
        )
        assert "✓ Task created" in result
        task_id = list(TASKS_DB.keys())[0]
        
        # Assign task
        result = assign_task(task_id, "Alice")
        assert "✓" in result
        
        # Update status
        result = update_task_status(task_id, "in_progress")
        assert "✓" in result
        
        # Get details
        result = get_task_details(task_id)
        assert "Complete workflow test" in result
        assert "Alice" in result
        
        # List tasks
        result = list_tasks()
        assert "Complete workflow test" in result
    
    def test_multiple_tasks_workflow(self):
        """Test workflow with multiple tasks."""
        # Create 3 tasks
        for i in range(3):
            create_task(
                title=f"Task {i+1}",
                description=f"Description {i+1}",
                priority="high" if i == 0 else "medium"
            )
        
        task_ids = list(TASKS_DB.keys())
        
        # Assign different people
        assign_task(task_ids[0], "Alice")
        assign_task(task_ids[1], "Bob")
        assign_task(task_ids[2], "Charlie")
        
        # Update different statuses
        update_task_status(task_ids[0], "in_progress")
        update_task_status(task_ids[1], "completed")
        
        # Get statistics
        result = get_statistics()
        assert "Total Tasks" in result
        assert "3" in result


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance with larger datasets."""
    
    def test_create_many_tasks(self):
        """Test creating many tasks."""
        task_count = 50
        
        for i in range(task_count):
            create_task(
                title=f"Task {i+1}",
                description=f"Description for task {i+1}"
            )
        
        assert len(TASKS_DB) == task_count
    
    def test_list_large_dataset(self):
        """Test listing with large dataset."""
        for i in range(30):
            create_task(
                title=f"Task {i+1}",
                description=f"Desc {i+1}",
                priority="high" if i % 5 == 0 else "medium"
            )
        
        result = list_tasks()
        assert "TASK-" in result


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_special_characters_in_task(self):
        """Test task with special characters."""
        special_title = "Task: #1 @Important! (Urgent) [Priority]"
        result = create_task(
            title=special_title,
            description="Testing special chars: !@#$%^&*()"
        )
        
        assert "✓ Task created" in result
        task_id = list(TASKS_DB.keys())[0]
        assert TASKS_DB[task_id].title == special_title
    
    def test_unicode_characters(self):
        """Test task with Unicode characters."""
        unicode_title = "任务 1 - Test Task 你好"
        result = create_task(
            title=unicode_title,
            description="Unicode test: 中文, 日本語"
        )
        
        assert "✓ Task created" in result
        task_id = list(TASKS_DB.keys())[0]
        assert TASKS_DB[task_id].title == unicode_title
    
    def test_all_statuses_transition(self):
        """Test transitioning through all status states."""
        create_task(title="Status test", description="All states")
        task_id = list(TASKS_DB.keys())[0]
        
        statuses = ["pending", "in_progress", "completed", "blocked", "pending"]
        
        for status in statuses:
            result = update_task_status(task_id, status)
            assert "✓" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])