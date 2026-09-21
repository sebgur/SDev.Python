---
name: recommendations
description: Analyse the repository and rank improvements toward commercial FinTech library quality in codex/recommendations.html. Run only when explicitly requested.
---

# Recommendations

Analyse the current codebase and recommend improvements that would make the
library suitable for commercial Financial Technology use.
Follow the repository AGENTS.md. Never modify existing Python files or
notebooks, including notebook outputs and metadata. Suggest changes without
implementing them; writing the requested HTML report is the deliverable.

## Analysis

- Exclude `sdevpy/thirdparty`; the user does not intend to change that code.
- Inspect the relevant source, notebook source, tests, packaging, and
  documentation. Read notebooks as data without executing or saving them.
- Consider code design, refactoring, file organisation, third-party packages,
  and other practices where they would materially improve the library.
- Ground recommendations in verified current repository evidence. Cite
  relevant file locations and distinguish observed weaknesses from proposed
  improvements. Do not reuse the example report's claims as evidence.
- Rank recommendations from highest to lowest materiality for a commercial
  FinTech library. Explain the benefit, suggested approach, and relevant
  tradeoffs or dependencies. Include concrete illustrative changes where
  helpful, without applying them.
- Verify current external package or product claims against primary sources
  when needed and link those sources. Do not claim that recommendations
  establish regulatory compliance or commercial readiness by themselves.

## Report

Write a self-contained HTML report to `codex/recommendations.html`, relative
to the repository root. Create the output directory if needed. Use
[assets/example.html](assets/example.html) as a broad reference for layout
and styling only. Replace all example recommendations, dates, rankings,
and claims with the current analysis, using Codex branding.
State the scope and any material coverage limitations. Escape code snippets
correctly for HTML display. Check for broken markup, missing content, and
stale example text. Return a link and a short summary in the conversation.
