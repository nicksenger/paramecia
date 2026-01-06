Create or overwrite a file with the specified content.

**Parameters:**
- `path` (required): Absolute path to the file. Must start with '/'.
- `content` (required): The content to write to the file.
- `overwrite`: Set to `true` to overwrite existing files. Default is `false`.

**Safety Rules:**
- By default, fails if the file already exists (prevents accidental data loss)
- Set `overwrite: true` to replace an existing file
- Parent directories are created automatically if they don't exist

**Best Practices:**
- ALWAYS use `read_file` first before overwriting to understand current contents
- PREFER `search_replace` for editing existing files rather than full rewrites
- NEVER write new files unless explicitly required

**Example - Create new file:**

<tool_call>
{"name": "write_file", "arguments": {"path": "/path/to/your/project/output.txt", "content": "Hello, world!\nThis is the file content."}}
</tool_call>

**Example - Overwrite existing file:**

<tool_call>
{"name": "write_file", "arguments": {"path": "/path/to/your/project/config.json", "content": "{\"key\": \"value\"}", "overwrite": true}}
</tool_call>
