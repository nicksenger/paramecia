Read and return the content of a specified file.

**Parameters:**
- `path` (required): The absolute path to the file. Relative paths are not supported.
- `offset`: Optional line number to start reading from (0-based).
- `limit`: Optional maximum number of lines to read.

If the file is large, content will be truncated. Use `offset` and `limit` to paginate through large files.

**Strategy for large files:**
1. Call with a `limit` (e.g., 500 lines) to get the start
2. If truncated, call again with `offset` to read the next chunk
3. Example: `offset=500, limit=500` reads lines 500-999

**Example:**

<tool_call>
{"name": "read_file", "arguments": {"path": "/path/to/your/project/src/main.rs"}}
</tool_call>

**Reading a specific range:**

<tool_call>
{"name": "read_file", "arguments": {"path": "/path/to/your/project/src/main.rs", "offset": 100, "limit": 50}}
</tool_call>
