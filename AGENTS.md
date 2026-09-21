# Repository Instructions

## Python and Notebooks: Suggestions Only

Never modify, overwrite, delete, rename, or move existing `.py` or `.ipynb`
files anywhere in this repository. This includes notebook cells, outputs,
and metadata. The user makes all changes to these files manually.

This restriction applies to every tool and indirect operation, including
patches, shell commands, scripts, formatters, notebook execution, and Git
operations. Do not run an operation that could write to these files.

Read and analyze code as needed. When asked to fix a bug, implement a feature,
refactor, or update code, provide suggestions only in the conversation:
identify the file and relevant location, show the proposed replacement code
or diff, and explain why the change is needed. Do not apply the changes.
These requests do not authorize edits to existing `.py` or `.ipynb` files.

## Memory

Do not read, write, or update files in memory directories. Do not use the
auto-memory system.

## Code Review Rules

- Before reporting an issue, read the exact file and verify its current line
  number directly.
- Discard any finding that cannot be confirmed in the current file. Do not
  rely on subagent summaries without verifying the source yourself.
- Do not carry findings forward from previous reviews without re-verifying
  them against the current working copy.
