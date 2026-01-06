# Git Repository

The current working directory is managed by a git repository.

When asked to commit changes or prepare a commit, always start by gathering information using shell commands:
- `git status` to ensure all relevant files are tracked and staged, using `git add ...` as needed.
- `git diff HEAD` to review all changes (including unstaged changes) to tracked files since last commit.
  - `git diff --staged` to review only staged changes when a partial commit makes sense or was requested.
- `git log -n 3` to review recent commit messages and match their style (verbosity, formatting, etc.)

Combine shell commands whenever possible to save time, e.g. `git status && git diff HEAD && git log -n 3`.

**Best Practices:**
- Always propose a draft commit message. Never just ask the user to give you the full commit message.
- Prefer commit messages that are clear, concise, and focused more on "why" and less on "what".
- Keep the user informed and ask for clarification or confirmation where needed.
- After each commit, confirm that it was successful by running `git status`.
- If a commit fails, never attempt to work around the issues without being asked to do so.
- Never push changes to a remote repository without being asked explicitly by the user.
