Use the `bash` tool to execute shell commands.

**Execution Mode:**
- `is_background: false` (default): Wait for command to complete, capture full output
- `is_background: true`: Start process in background, return immediately

**Use background execution for:**
- Development servers: `npm run dev`, `python manage.py runserver`
- Build watchers: `npm run watch`, `webpack --watch`
- Database servers: `mongod`, `redis-server`
- Any command that runs indefinitely

**Use foreground execution for:**
- One-time commands: `ls`, `cat`, `grep`, `pwd`
- Build commands: `npm run build`, `cargo build`
- Installation: `npm install`, `pip install`
- Git operations: `git commit`, `git push`
- Tests: `npm test`, `cargo test`, `pytest`

**IMPORTANT - Use dedicated tools instead of bash:**

| Instead of... | Use... |
|---------------|--------|
| `cat file.txt` | `read_file(path="file.txt")` |
| `head -n 20 file` | `read_file(path="file", limit=20)` |
| `grep -r "pattern" .` | `grep(pattern="pattern", path=".")` |
| `sed -i 's/old/new/' file` | `search_replace` tool |
| `echo "text" > file` | `write_file(path="file", content="text")` |

**Appropriate bash uses:**
- System info: `pwd`, `whoami`, `date`, `uname -a`
- Directory listings: `ls -la`, `tree`
- Git operations: `git status`, `git log --oneline -10`, `git diff`
- Process info: `ps aux`, `top -n 1`
- Package info: `pip list`, `npm list`
- File metadata: `stat file`, `file filename`, `wc -l file`

**Example:**

<tool_call>
{"name": "bash", "arguments": {"command": "git status"}}
</tool_call>

**WARNING:** Command substitution using `$()`, backticks, `<()`, or `>()` may be restricted for security.
