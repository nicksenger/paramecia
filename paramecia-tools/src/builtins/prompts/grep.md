Recursively search files for a regex pattern. Fast and respects .gitignore.

**Parameters:**
- `pattern` (required): Regex pattern to search for.
- `path`: Directory or file to search. Defaults to current directory.
- `max_matches`: Optional limit on number of matches returned.

**Example - Find function definitions:**

<tool_call>
{"name": "grep", "arguments": {"pattern": "fn main", "path": "src/"}}
</tool_call>

**Example - Find TODO comments:**

<tool_call>
{"name": "grep", "arguments": {"pattern": "TODO|FIXME", "path": "."}}
</tool_call>

**Example - Find imports:**

<tool_call>
{"name": "grep", "arguments": {"pattern": "^use std::", "path": "src/"}}
</tool_call>

**Regex Tips:**
- `^` matches start of line
- `$` matches end of line
- `.*` matches any characters
- `\s+` matches whitespace
- Escape special characters with `\`
