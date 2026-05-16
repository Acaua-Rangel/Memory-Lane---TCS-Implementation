---
description: "Use when writing or editing markdown documentation files (.md). Enforces academic writing style with references, connective transitions, topic introductions, and proper citations. Applies to all markdown files in docs/."
applyTo: "docs/**/*.md"
---

# Markdown Writing Standards

## Academic Rigor

### References Are Mandatory
- Every factual claim MUST have a citation: `[[N]](#ref-N)` linking to a `## References` section.
- Statistics, benchmarks, model performance numbers, and architectural claims all require sources.
- Prefer primary sources: peer-reviewed papers (arXiv, IEEE, ACL), official documentation (WHO, Google AI).
- Format references as:
  ```markdown
  <a id="ref-1"></a>**[1]** Author, A., Author, B. (Year). "Title." *Venue*. https://doi.org/...
  ```

### Connective Transitions
- Between every two sentences, ensure a logical connector exists for reading fluidity.
- Use transition words: "Furthermore,", "In contrast,", "Consequently,", "Building on this,", "To address this,", "More specifically,".
- Avoid abrupt topic jumps. Each paragraph should flow from the previous one.

### Topic Introductions
- New sections MUST begin with an introductory sentence that frames the topic.
- Pattern: "This section introduces...", "We present here...", "Building on the previous section,...".
- Example:
  ```markdown
  ## Face Recognition Pipeline

  This section introduces the lightweight face recognition module, which operates
  independently from the LLM to maintain low latency on mobile devices.
  ```

## Structure

### Headings
- Use `##` for main sections, `###` for subsections, `####` for sub-subsections.
- Headings must be descriptive and concise.

### Tables
- Use tables for structured comparisons (components, hyperparameters, benchmarks).
- Always include a header row.

### Code Blocks
- Use fenced code blocks with language identifiers: ` ```python `, ` ```bash `.
- Code examples must be correct and runnable.

## Consistency with Codebase

### Sync Rule
- If a code change alters architecture, training, tools, or routing:
  1. Update `docs/architecture.md` with structural changes.
  2. Update `docs/training_plan.md` with hyperparameter or phase changes.
  3. Update `docs/PROJECT.md` with feature or tool changes.
  4. Update `docs/KAGGLE_WRITEUP.md` with any user-facing description changes.
  5. Update `.github/copilot-instructions.md` if file organization or constraints change.
  6. Update `.instructions/*.instructions.md` if domain-specific patterns change.

### Numbers Must Match
- If `src/config.py` says `lr=2e-4`, then `docs/training_plan.md` must say `2e-4`.
- If `src/tools/sqlite_tools.py` has 9 tools, then `docs/PROJECT.md` must list 9 tools.
- If `src/model.py` uses `kernel_size=7`, then `docs/architecture.md` must say `kernel=7`.
