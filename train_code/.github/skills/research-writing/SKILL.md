---
name: research-writing
description: "Write academic-quality markdown with proper citations, connective transitions, and topic introductions. Use when creating or editing docs/KAGGLE_WRITEUP.md, docs/PROJECT.md, paper sections, or any document that requires referenced claims. Covers citation format, source verification, and writing style."
---

# Research Writing Skill

## When to Use

- Writing or editing `docs/KAGGLE_WRITEUP.md`
- Writing or editing `docs/PROJECT.md`
- Writing any markdown that contains factual claims, benchmarks, or statistics
- Adding new technical sections that reference external work

## Citation Format

### Inline Citations
Use anchor-linked numbered references: `[[N]](#ref-N)`

Example: "MobileFaceNet achieves 99.55% accuracy on LFW [[6]](#ref-6)."

### Reference Section
Every document with citations MUST end with a `## References` section:

```markdown
## References

<a id="ref-1"></a>**[1]** Last, F., Last, F. (Year). "Paper Title." *Venue/Journal*. https://arxiv.org/abs/XXXX.XXXXX

<a id="ref-2"></a>**[2]** Organization. "Document Title." Publisher, Year. Available at: https://...
```

### Source Priority
1. **Peer-reviewed papers** (arXiv, IEEE, ACL, NeurIPS, ICML)
2. **Official documentation** (WHO, Google AI, PyTorch docs)
3. **GitHub repositories** (for open-source tools)
4. **Never**: Wikipedia, blog posts, or unverified sources

## Writing Style

### Connective Transitions (MANDATORY)
Between every two sentences, ensure logical flow. Use:
- **Additive**: "Furthermore,", "Moreover,", "In addition,"
- **Causal**: "Consequently,", "As a result,", "Therefore,"
- **Contrastive**: "However,", "In contrast,", "Nevertheless,"
- **Sequential**: "First,", "Subsequently,", "Building on this,"
- **Elaborative**: "More specifically,", "In particular,", "To illustrate,"

### Topic Introductions (MANDATORY)
New sections/subsections MUST open with a framing sentence:
- "This section introduces the [topic], which [relevance to project]."
- "Building on the [previous topic], we now present [new topic]."
- "We introduce here the [component], designed to address [problem]."

### Tone
- Technical deep-dive: precise, data-driven, process-oriented.
- Honest about limitations: "training is in progress", "not yet tested with real patients".
- First-person plural ("we") for team decisions.

## Procedure

### Step 1 — Identify Claims
Read the text and highlight every factual statement that needs a source:
- Statistics ("55 million people", "99.55% accuracy")
- Architectural claims ("LoRA reduces parameters by 10,000×")
- Performance benchmarks ("sub-millisecond on mobile GPUs")
- Tool/method descriptions (citing the original paper)

### Step 2 — Find Sources
For each claim:
1. Check if a reference already exists in the document's `## References`.
2. If not, search for the primary source (arXiv, official docs).
3. Verify the claim matches the source (exact numbers, correct attribution).

### Step 3 — Insert Citations
Add `[[N]](#ref-N)` inline and append to `## References`.

### Step 4 — Apply Writing Style
Review each paragraph for:
- Missing connectives between sentences
- Missing topic introduction at section start
- Abrupt transitions between ideas

### Step 5 — Cross-Check with Code
Verify all technical details match the codebase:
- Hyperparameters → `src/config.py`
- Architecture → `src/model.py`
- Tool list → `src/tools/sqlite_tools.py`
- Training phases → `src/config.py` PHASE_PRESETS

## Known References for This Project

| Ref | Paper | Citation Key |
|---|---|---|
| LoRA | Hu et al. (2021), arXiv:2106.09685 | Low-Rank Adaptation |
| Knowledge Distillation | Hinton et al. (2015), arXiv:1503.02531 | Self-distillation basis |
| BlazeFace | Bazarevsky et al. (2019), arXiv:1907.05047 | MediaPipe face detection |
| MobileFaceNet | Chen et al. (2018), arXiv:1804.07573 | Face embeddings |
| ArcFace | Deng et al. (2019), arXiv:1801.07698 | Face recognition baseline |
| CLIP | Radford et al. (2021), arXiv:2103.00020 | Vision-language baseline |
| sqlite-vec | asg017/sqlite-vec, GitHub | Vector search |
| ADI | Alzheimer's Disease International, 2024 | Dementia statistics |
| WHO | WHO Dementia Fact Sheet, March 2025 | Dementia epidemiology |
