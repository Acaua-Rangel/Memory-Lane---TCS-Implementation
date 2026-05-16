# Training Plan

## Overview

Three-phase training pipeline targeting 2× NVIDIA T4 GPUs (Kaggle).

## Hardware Constraints

- **GPUs:** 2× NVIDIA T4 (16GB VRAM each, 32GB total)
- **T4 specs:** FP16 throughput ~65 TFLOPS, INT8 ~130 TOPS
- **System RAM:** 30GB (Kaggle)
- **Disk:** ~19.5GB available ⚠️ tight — checkpoint rotation required
- **Session limit:** 12 hours (Kaggle)

### Disk Budget Strategy

| Item | Size | Notes |
|---|---|---|
| Base model (int8 cached) | ~4GB | HF cache, downloaded once |
| Dataset cache (wikitext) | ~2GB | Streaming avoids full download |
| Synthetic data (JSONL) | ~50MB | Generated locally |
| Checkpoints (rolling 3) | ~450MB | max_checkpoints_to_keep=3 |
| Phase final weights | ~50MB | TCS + LoRA only |
| **Total peak** | **~7GB** | Leaves ~12GB headroom |

**Key rule:** After each phase finishes, intermediate checkpoints are **deleted**. Only `final/` directory with merged LoRA + TCS weights is kept.

## Phase 1: TCS Pre-training

**Duration:** ~30 minutes

**Objective:** Train the Token Compression Sub-network to produce meaningful compressed representations while Gemma 4 is completely frozen.

**Setup:**
```bash
poetry run train --phase tcs-pretrain \
    --model google/gemma-4-e2b-it \
    --compression-ratio 4 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --lr 1e-3 \
    --warmup-steps 100 \
    --max-steps 2000 \
    --mixed-precision bf16
```

**Loss:** Self-distillation
- Forward pass WITHOUT TCS → teacher logits (detached)
- Forward pass WITH TCS → student logits
- L = KL(student_soft ‖ teacher_soft) * T²

**Data preprocessing cache:**
- After loading and filtering raw text, token chunks are cached to `data/text_cache/`.
- Future runs with the same dataset/tokenizer/sequence settings reuse cached chunks and skip full re-tokenization.
- Cache key includes dataset name/config, split, max sequence length, sample cap, and tokenizer identity.

**Key decisions:**
- High LR (1e-3) — TCS is randomly initialized, needs to learn fast
- Short training — only ~3M params to learn
- General text data (no domain-specific data needed yet)

## Phase 2: TCS + LoRA Joint Training

**Duration:** 2-4 hours

**Objective:** Train LoRA adapters so attention layers work correctly with compressed tokens. The Gemma 4 base weights remain frozen.

**Setup:**
```bash
torchrun --standalone --nproc_per_node=2 -m src.train_ddp \
    --phase tcs-lora \
    --model google/gemma-4-e2b-it \
    --tcs-checkpoint outputs/phase1/tcs_final.pt \
    --compression-ratio 4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --lr 2e-4 \
    --warmup-steps 200 \
    --max-steps 10000 \
    --mixed-precision bf16
```

**Loss:** Hybrid self-distillation
- α · T² · KL(compressed ‖ uncompressed) + (1-α) · CE(compressed, targets)
- α = 0.5, T = 2.0

**Key decisions:**
- Load TCS weights from Phase 1 (warm start)
- Lower LR (2e-4) — LoRA adapters are small perturbations
- DDP across 2× T4 for throughput
- Gradient accumulation = 4 → effective batch 32
- `train_ddp` is wrapped with Torch Elastic `@record`, so rank-level root-cause tracebacks are surfaced directly in launcher logs.
- In addition, DDP is configured with `find_unused_parameters=True` because optional TCS components (for example, decompression blocks) may be present in the module while not participating in the current phase loss graph.
- Building on these diagnostics, `train_ddp` also applies an early-step OOM fallback: if CUDA OOM is detected in the initial step window, the run is retried with a smaller micro-batch and adjusted gradient accumulation to preserve the effective batch size.

## Phase 3: Tool Calling Fine-tune

**Duration:** 1-2 hours

**Objective:** Fine-tune the compressed model to correctly invoke SQLite tools in Alzheimer companion scenarios.

**Setup:**
```bash
torchrun --standalone --nproc_per_node=2 -m src.train_ddp \
    --phase tool-calling \
    --model google/gemma-4-e2b-it \
    --tcs-checkpoint outputs/phase2/tcs_lora_final.pt \
    --lora-checkpoint outputs/phase2/lora_final/ \
    --compression-ratio 4 \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --lr 5e-5 \
    --warmup-steps 100 \
    --max-steps 5000 \
    --mixed-precision bf16
```

**Single-GPU 4-bit alternative:**
```bash
python -m src.train_unsloth \
    --phase tool-calling \
    --model google/gemma-4-e2b-it \
    --compression-ratio 4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --tcs-checkpoint outputs/tcs_lora/final/tcs_weights.pt \
    --lora-checkpoint outputs/tcs_lora/final/lora_adapters \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --mixed-precision bf16 \
    --output-dir outputs \
    --load-in-4bit
```

**Loss:** Standard CE on tool calling formatted outputs, with masked assistant labels aligned to compressed positions before the reduction

**Data:** Synthetic conversations generated via `poetry run generate-data`

**Key decisions:**
- Lowest LR (5e-5) — preserving Phase 2 representations
- Smaller batch size — tool calling sequences are longer
- Focus on structured output accuracy
- To use both Kaggle T4s, launch `src.train_ddp` with `torchrun`; the `src.train_unsloth` entrypoint remains single-GPU and is intended for the 4-bit alternative above.
- Because the compressed TCS path emits fewer logits than the original token sequence, the loss aligns unmasked target positions to the compressed length by deterministic index sampling before cross-entropy.
- In addition, the Unsloth wrapper resolves hidden size from either `config.hidden_size` or `config.text_config.hidden_size`, ensuring compatibility with Gemma 4 multimodal configs.
- In addition, the Unsloth wrapper accepts both TCS checkpoint formats: nested (`tcs`, `pos_emb`, `decompressor`) and flat state-dict prefixes (`tcs.`, `compressed_pos_emb.`, `decompressor.`).
- In addition, when Unsloth returns a multimodal processor object, the training path normalizes it to the nested text tokenizer (`processor.tokenizer`) so dataset tokenization can use `encode()` safely.
- In addition, the Unsloth training path passes `model_cfg.max_seq_len` explicitly into dataset construction, so text chunking is tied to sequence length and never to micro-batch size.
- In addition, when `mixed_precision=bf16` is requested on GPUs without bfloat16 support (for example, T4), the Unsloth loop now falls back to fp16 autocast automatically instead of failing at runtime.
- In addition, Gemini teacher cache generation now falls back to local model generation not only for quota/import failures, but also for runtime API-path errors (for example, prompt decode or response tokenization failures).
- In addition, the Unsloth compressed forward path now bypasses Gemma 4 multimodal per-layer input expansion by running decoder layers directly on compressed embeddings, preventing catastrophic OOM allocation on `inputs_embeds` routes.
- In addition, text-backbone discovery for this bypass now walks nested Unsloth/PEFT wrapper layers recursively, reducing breakage across wrapper-structure changes between library versions.
- In addition, for distillation phases (`tcs-pretrain`, `tcs-lora`), the Unsloth loader now sets `UNSLOTH_RETURN_LOGITS=1` automatically before model initialization so teacher logits remain available in newer Unsloth releases.

## Checkpointing Strategy

```
outputs/
├── phase1/
│   ├── tcs_step_0500.pt
│   ├── tcs_step_1000.pt
│   └── tcs_final.pt
├── phase2/
│   ├── tcs_lora_step_2000.pt
│   ├── tcs_lora_step_5000.pt
│   ├── tcs_lora_final.pt
│   └── lora_final/          # HuggingFace PEFT format
├── phase3/
│   ├── tcs_lora_tc_step_1000.pt
│   ├── tcs_lora_tc_final.pt
│   └── lora_final/          # Final merged LoRA
└── final/
    ├── model/                # Merged model for HF upload
    └── tcs_weights.pt        # TCS weights separate
```

## Validation Metrics

| Phase | Primary Metric | Target |
|---|---|---|
| Phase 1 | KL divergence (compressed vs full) | < 0.5 |
| Phase 2 | Perplexity on validation set | < 15.0 |
| Phase 3 | Tool call accuracy (exact match) | > 90% |
| Phase 3 | Tool argument accuracy | > 85% |

## Dataset Requirements

| Phase | Dataset | Size |
|---|---|---|
| Phase 1 | General text (wikitext-103 or OpenWebText subset) | ~100M tokens |
| Phase 2 | General + conversational mix | ~200M tokens |
| Phase 3 | Synthetic tool calling conversations | ~10K examples |
