---
applyTo: 'src/train.py,src/train_ddp.py,scripts/**'
---

# Training Skill

## Training Phases

### Phase 1: tcs-pretrain
- Only TCS parameters are trainable (~3M params).
- All Gemma 4 params frozen (including LoRA — not added yet).
- Loss: Self-distillation KL divergence.
- LR: 1e-3 (high — TCS is randomly initialized).
- Duration: ~2000 steps, ~30 min on 2× T4.

### Phase 2: tcs-lora
- TCS + LoRA adapters trainable (~23M params).
- Gemma 4 base frozen.
- Loss: α · KL(compressed ‖ uncompressed) + (1-α) · CE.
- LR: 2e-4 with cosine decay.
- Load TCS checkpoint from Phase 1.
- Duration: ~10000 steps, ~2-4h on 2× T4.

### Phase 3: tool-calling
- Same trainable params as Phase 2, lower LR.
- Loss: Standard CE on tool calling formatted text.
- LR: 5e-5 with cosine decay.
- Load TCS + LoRA from Phase 2.
- Duration: ~5000 steps, ~1-2h on 2× T4.

## DDP Configuration (Kaggle 2× T4)
- Use `torch.distributed` with `nccl` backend.
- Or use HuggingFace `accelerate` for simpler DDP.
- Batch size per GPU = 4, gradient accumulation = 4, effective batch = 32.
- Gradient clipping: max_norm = 1.0.

## Checkpointing
- Save every 500 steps (Phase 1) or 1000 steps (Phase 2/3).
- Save only trainable params (TCS + LoRA) — not the full model.
- Use safetensors format.
- Save optimizer + scheduler state for resume.
- **Rotation:** Keep only last 3 checkpoints (`max_checkpoints_to_keep=3`).
- **Phase cleanup:** After saving final weights, delete all intermediate checkpoints and `best/` dir.
- **Disk constraint:** Kaggle has ~19.5GB total. Monitor with `du -sh` between phases.

## Mixed Precision
- Use `torch.amp.autocast('cuda', dtype=torch.bfloat16)`.
- Use `torch.amp.GradScaler('cuda')` only if using float16 (not needed for bf16).
- T4 supports FP16 natively. BF16 is emulated but works.

## Validation
- Validate every 500 steps.
- Metrics: perplexity (Phase 1/2), tool call accuracy (Phase 3).
- Use a small held-out set for fast validation.

## CLI Pattern
All training is launched via CLI commands:
```bash
poetry run train --phase <phase> [options]
poetry run train-ddp --phase <phase> --nproc_per_node 2 [options]
```

Use argparse for CLI. Each phase has sensible defaults that can be overridden.
