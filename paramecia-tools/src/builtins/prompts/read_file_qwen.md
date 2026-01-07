Read and return the content of a specified file. If the file is large, the content will be truncated.

The tool's response will clearly indicate if truncation has occurred and will provide details on how to read more of the file using the 'offset' and 'limit' parameters.

Handles text files and can read specific line ranges.

**Parameters:**
- `path` (string, required): The absolute path to the file to read. Relative paths are not supported.
- `offset` (integer): The 0-based line number to start reading from. Use for paginating through large files.
- `limit` (integer): Maximum number of lines to read. Use with 'offset' to paginate through large files.

**Strategy for large files:**

1. Call `read_file` with a `limit` (e.g., 500 lines) to get the start of the file.
2. If the output indicates truncation, you know the file is large.
3. To read the next chunk, call `read_file` again with an `offset`. For example, `offset=500, limit=500`.

This is more efficient than using `bash` with `cat` or `head`.

**Example tool call:**
```
<tool_call>
<function=read_file>
<parameter=path>
/path/to/your/project/src/main.rs
</parameter>
</function>
</tool_call>
```

**Example with pagination:**
```
<tool_call>
<function=read_file>
<parameter=path>
/path/to/your/project/src/main.rs
</parameter>
<parameter=offset>
100
</parameter>
<parameter=limit>
50
</parameter>
</function>
</tool_call>
```
