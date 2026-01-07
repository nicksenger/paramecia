Manage a structured todo list for tracking tasks and progress.

**Actions:**
- `read`: View current todo list
- `write`: Update the todo list

**Todo Item Structure:**
- `id`: Unique identifier
- `content`: Task description
- `status`: `pending`, `in_progress`, `completed`, or `cancelled`

**When to Use:**
- Planning complex multi-step tasks
- Tracking progress on work
- Giving users visibility into your focus
- Ensuring no requirements are missed

**Best Practices:**
1. Create todos early when starting complex tasks
2. Mark `in_progress` when you start a task
3. Mark `completed` immediately when done (don't batch)
4. Keep only one task `in_progress` at a time
5. Add new todos as scope expands

**Example - Create todo list:**

<tool_call>
{"name": "todo", "arguments": {"action": "write", "todos": [{"id": "1", "content": "Research existing implementation", "status": "in_progress"}, {"id": "2", "content": "Implement core functionality", "status": "pending"}, {"id": "3", "content": "Add tests", "status": "pending"}, {"id": "4", "content": "Update documentation", "status": "pending"}]}}
</tool_call>

**Example - Read current todos:**

<tool_call>
{"name": "todo", "arguments": {"action": "read"}}
</tool_call>
