---
name: code-review
description: Review the full repository with an overall grade, category scorecard, and up to ten highest-severity issues in codex/code-review.html. Run only when explicitly requested.
---

# Code Review

Review the current codebase, not just changes since a previous review.
Follow the repository AGENTS.md. Never modify existing Python files or
notebooks, including notebook outputs and metadata. Suggest fixes without
applying them; writing the requested HTML report is the deliverable.

## Scope and evidence

- Inspect every project `.py` and `.ipynb` file, including files excluded in
  `pyproject.toml` or ignored by search defaults. Exclude `sdevpy/thirdparty`.
  Do not treat installed environments or tool caches as project source.
- Read notebooks as data without executing or saving them.
- Verify each finding directly in the current source. Cite file paths and
  line numbers; for notebooks, identify the cell and the relevant source.
- Do not reuse findings from example reports or earlier reviews without
  independently verifying them. Disclose any unreadable files or incomplete
  coverage instead of claiming a complete review.

## Report

Give an overall grade and explain the scoring scale. Provide a score, rating,
and brief notes for Architecture, Correctness, Error Handling, Testing,
Code Quality, Maintainability, Type Safety, and Documentation.

Rank the ten most serious verified issues and bad practices by severity.
If more than ten critical or high-severity issues exist, report only the ten
most severe. If fewer than ten substantiated issues exist, do not invent extras.
For each, give the severity, exact location, evidence, practical impact, and
an explicit proposed code change showing what to replace, where, and how.
Proposed code is illustrative and must never be applied to source files.

Write a self-contained HTML report to `codex/code-review.html`, relative to
the repository root. Create the output directory if needed. Use
[assets/example.html](assets/example.html) for layout and styling only;
replace all example findings, scores, dates, and claims with current evidence.
Use Codex branding. Escape code snippets correctly for HTML display.
Check the report for broken markup, missing sections, and stale example text.
Return a link to the report and a short summary in the conversation.
