Replace text within a file using SEARCH/REPLACE blocks.

**Parameters:**
- `file_path` (required): Absolute path to the file.
- `content` (required): The SEARCH/REPLACE blocks defining changes.

**Format:**

```
<<<<<<< SEARCH
[exact text to find]
=======
[replacement text]
>>>>>>> REPLACE
```

**Critical Rules:**
1. `file_path` MUST be absolute (starts with `/`)
2. SEARCH block must match file content EXACTLY (whitespace, indentation, everything)
3. Include 3-5 lines of context before/after the change for unique matching
4. NEVER escape the content - use exact literal text

**Example:**

<tool_call>
{"name": "search_replace", "arguments": {"file_path": "/path/to/your/project/src/main.rs", "content": "<<<<<<< SEARCH\nfn old_function() {\n    println!(\"old\");\n}\n=======\nfn new_function() {\n    println!(\"new\");\n}\n

**Multiple replacements:** Set `replace_all` to true to replace every occurrence.
