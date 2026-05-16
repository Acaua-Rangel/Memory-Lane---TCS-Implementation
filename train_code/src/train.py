"""
Single-GPU training script for the Gemma 4 Memory Companion.

Supports three training phases:
  1. tcs-pretrain  — Train TCS only (self-distillation)
  2. tcs-lora      — Train TCS + LoRA jointly
  3. tool-calling  — Fine-tune on tool calling data
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.config import (
    ModelConfig,
    CompressionConfig,
    LoraConfig,
    TrainingConfig,
    apply_phase_preset,
)
from src.data import build_dataloader, TextDataset, ToolCallingDataset
from src.losses import build_loss
from src.model import Gemma4WithTCS
from src.utils.checkpoint import (
    CheckpointState,
    cleanup_phase_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini API Teacher
# ---------------------------------------------------------------------------

class GeminiTeacher:
    """
    Uses the Gemini API to generate teacher completions for training data.

    The Gemini API doesn't expose full logits for teacher-forcing, so we use
    sequence-level distillation (Kim & Rush, 2016): the teacher generates
    completions which are cached and used as auxiliary CE targets.

    The student learns from two signals:
    - KL loss against the local model's uncompressed path (self-distillation)
    - CE loss against the API teacher's token predictions (sequence distillation)
    """

    def __init__(
        self,
        model_name: str = "gemma-4-31b-it",
        api_key: str | None = None,
        rpm_limit: int = 14,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Gemini teacher. "
                "Install with: poetry add google-genai"
            ) from exc

        from src.data import _resolve_gemini_api_key

        resolved_key = api_key or _resolve_gemini_api_key()
        if not resolved_key:
            raise ValueError(
                "No API key found for Gemini teacher. Set GEMINI_API_KEY "
                "or pass gemini_api_key in TrainingConfig."
            )

        self._client = genai.Client(api_key=resolved_key)
        self._types = types
        self._model_name = model_name
        self._rpm_limit = rpm_limit

        import threading
        self._rate_lock = threading.Lock()
        self._request_times: list[float] = []

        logger.info("GeminiTeacher initialized: model=%s, rpm=%d", model_name, rpm_limit)

    def _wait_for_rate_limit(self) -> None:
        """Block until a request slot is available within the RPM window."""
        import time
        with self._rate_lock:
            now = time.monotonic()
            window = 60.0
            self._request_times = [
                t for t in self._request_times if now - t < window
            ]
            if len(self._request_times) >= self._rpm_limit:
                sleep_time = window - (now - self._request_times[0]) + 0.1
                if sleep_time > 0:
                    self._rate_lock.release()
                    try:
                        time.sleep(sleep_time)
                    finally:
                        self._rate_lock.acquire()
                    now = time.monotonic()
                    self._request_times = [
                        t for t in self._request_times if now - t < window
                    ]
            self._request_times.append(time.monotonic())

    def generate_completion(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a teacher completion for a prompt."""
        self._wait_for_rate_limit()
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=0.3,  # Low temperature for consistent teacher signal
                top_p=0.95,
                max_output_tokens=max_tokens,
            ),
        )
        return (response.text or "").strip()

    def cleanup(self) -> None:
        """Close the API client."""
        if hasattr(self, "_client"):
            self._client.close()


def precompute_teacher_cache(
    tokenizer,
    cfg: TrainingConfig,
    local_teacher_model: Any | None = None,
) -> Path | None:
    """
    Pre-compute teacher completions via Gemini API and cache to disk.

    For each training text, sends it to the API teacher model and caches
    the teacher's continuation tokens. These are used during training as
    auxiliary CE targets (sequence-level distillation).

    Returns the cache path, or None if API is unavailable.
    """
    from src.data import _is_quota_error

    def _build_teacher_dataset(max_seq_len: int = 512):
        if cfg.phase in ("tcs-pretrain", "tcs-lora"):
            return TextDataset(
                tokenizer=tokenizer,
                dataset_name=cfg.dataset_name,
                dataset_config=cfg.dataset_config,
                split="train",
                max_seq_len=max_seq_len,
            )
        return ToolCallingDataset(
            tokenizer=tokenizer,
            data_path="data/tool_calling_train.jsonl",
            max_seq_len=max_seq_len,
        )

    def _save_teacher_cache(teacher_labels_list: list[torch.Tensor], cache_path: Path) -> Path | None:
        if not teacher_labels_list:
            logger.warning("No teacher completions generated, skipping cache")
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(teacher_labels_list, cache_path)
        logger.info(
            "Teacher cache saved: %d examples → %s",
            len(teacher_labels_list), cache_path,
        )
        return cache_path

    def _resolve_local_generation_model(candidate_model: Any) -> Any | None:
        if candidate_model is None:
            return None
        if hasattr(candidate_model, "generate"):
            return candidate_model
        if hasattr(candidate_model, "base_model") and hasattr(candidate_model.base_model, "generate"):
            return candidate_model.base_model

        logger.warning("Local teacher fallback unavailable: model does not expose generate()")
        return None

    def _infer_model_device(model_obj: Any) -> torch.device:
        try:
            return next(model_obj.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _precompute_local_teacher_cache(
        initial_labels: list[torch.Tensor] | None = None,
    ) -> Path | None:
        generation_model = _resolve_local_generation_model(local_teacher_model)
        if generation_model is None:
            return None

        cache_dir = Path(cfg.teacher_cache_dir)
        local_cache_path = cache_dir / f"teacher_{cfg.phase}.pt"

        teacher_labels_list: list[torch.Tensor] = []
        if initial_labels:
            teacher_labels_list.extend([labels.detach().cpu().to(torch.long) for labels in initial_labels])

        dataset = _build_teacher_dataset(max_seq_len=512)
        num_examples = len(dataset)
        max_cache = min(num_examples, cfg.max_steps * cfg.effective_batch_size, 2048)
        if max_cache <= len(teacher_labels_list):
            return _save_teacher_cache(teacher_labels_list, local_cache_path)

        model_device = _infer_model_device(generation_model)
        was_training = bool(getattr(generation_model, "training", False))
        if hasattr(generation_model, "eval"):
            generation_model.eval()

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        logger.info(
            "Building local teacher cache fallback for phase=%s (%d -> %d examples)",
            cfg.phase,
            len(teacher_labels_list),
            max_cache,
        )

        try:
            for idx in range(len(teacher_labels_list), max_cache):
                item = dataset[idx]
                input_ids = item["input_ids"]
                attention_mask = item.get("attention_mask")

                generate_kwargs: dict[str, Any] = {
                    "input_ids": input_ids.unsqueeze(0).to(model_device),
                    "max_new_tokens": 128,
                    "do_sample": False,
                }
                if attention_mask is not None:
                    generate_kwargs["attention_mask"] = attention_mask.unsqueeze(0).to(model_device)
                if pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = pad_token_id
                if eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = eos_token_id

                generated = generation_model.generate(**generate_kwargs)
                generated_ids = generated.squeeze(0).detach().to("cpu", dtype=torch.long)

                prompt_len = input_ids.size(0)
                teacher_tokens = generated_ids[prompt_len:]
                if teacher_tokens.numel() == 0:
                    teacher_tokens = input_ids.detach().cpu().to(torch.long)

                teacher_labels_list.append(teacher_tokens)

                if (idx + 1) % 50 == 0:
                    logger.info("  Local teacher cache: %d/%d examples", idx + 1, max_cache)
        except Exception as exc:
            logger.warning("Local teacher fallback failed (%s)", exc)
        finally:
            if was_training and hasattr(generation_model, "train"):
                generation_model.train()

        return _save_teacher_cache(teacher_labels_list, local_cache_path)

    cache_dir = Path(cfg.teacher_cache_dir)
    cache_path = cache_dir / f"teacher_{cfg.phase}.pt"

    if cache_path.exists():
        logger.info("Teacher cache already exists: %s", cache_path)
        return cache_path

    try:
        teacher = GeminiTeacher(
            model_name=cfg.gemini_teacher_model,
            api_key=cfg.gemini_api_key,
            rpm_limit=cfg.gemini_rpm_limit,
        )
    except (ImportError, ValueError) as exc:
        logger.warning("Gemini teacher unavailable (%s), using local teacher fallback", exc)
        return _precompute_local_teacher_cache()

    # Smoke test
    try:
        teacher.generate_completion("Say hello.", max_tokens=10)
    except Exception as exc:
        logger.warning("Gemini teacher API failed (%s), using local teacher fallback", exc)
        teacher.cleanup()
        return _precompute_local_teacher_cache()

    logger.info("Pre-computing Gemini teacher cache for phase=%s...", cfg.phase)

    dataset = _build_teacher_dataset(max_seq_len=512)

    # Generate teacher completions for each example
    teacher_labels_list: list[torch.Tensor] = []
    num_examples = len(dataset)
    max_cache = min(num_examples, cfg.max_steps * cfg.effective_batch_size)

    for idx in range(max_cache):
        item = dataset[idx]
        input_ids = item["input_ids"]

        try:
            # Decode input tokens back to text (for API prompt)
            prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            teacher_response = teacher.generate_completion(prompt_text, max_tokens=512)

            # Tokenize teacher response
            teacher_tokens = tokenizer(
                teacher_response,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            teacher_labels_list.append(teacher_tokens)

            if (idx + 1) % 50 == 0:
                logger.info("  Teacher cache: %d/%d examples", idx + 1, max_cache)
        except Exception as exc:
            if _is_quota_error(exc):
                logger.warning(
                    "Gemini API quota exhausted after %d/%d examples; switching to local teacher fallback",
                    idx, max_cache,
                )
            else:
                logger.warning(
                    "Gemini API failed after %d/%d examples (%s); switching to local teacher fallback",
                    idx, max_cache, exc,
                )

            teacher.cleanup()
            fallback_path = _precompute_local_teacher_cache(initial_labels=teacher_labels_list)
            if fallback_path is not None:
                return fallback_path

            if not teacher_labels_list:
                return None

            logger.warning("Local fallback unavailable; saving partial Gemini teacher cache")
            break

    teacher.cleanup()
    return _save_teacher_cache(teacher_labels_list, cache_path)


def load_teacher_cache(cache_path: Path) -> list[torch.Tensor]:
    """Load pre-computed teacher labels from cache."""
    return torch.load(cache_path, weights_only=True)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(model: Gemma4WithTCS, cfg: TrainingConfig) -> None:
    """Main training loop (single GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Move trainable parts to device (base model may already be on device via device_map)
    if model.tcs is not None:
        model.tcs.to(device)
    if hasattr(model, "compressed_pos_emb"):
        model.compressed_pos_emb.to(device)
    if hasattr(model, "decompressor"):
        model.decompressor.to(device)

    # Build DataLoader
    train_loader = _build_train_loader(model.tokenizer, cfg)
    val_loader = _build_val_loader(model.tokenizer, cfg)

    # Optimizer (only trainable params)
    params = model.get_trainable_params(cfg.phase)
    optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger.info("Optimizer: AdamW, lr=%.2e, %d param groups", cfg.lr, len(params))

    # Scheduler
    scheduler = _build_scheduler(optimizer, cfg)

    # Loss
    loss_fn = build_loss(
        phase=cfg.phase,
        alpha=cfg.distill_alpha,
        temperature=cfg.distill_temperature,
    )

    # Gemini API teacher (optional: pre-compute or load cache)
    teacher_cache: list[torch.Tensor] | None = None
    if cfg.use_gemini_teacher and cfg.phase in ("tcs-pretrain", "tcs-lora"):
        cache_path = Path(cfg.teacher_cache_dir) / f"teacher_{cfg.phase}.pt"
        if cache_path.exists():
            teacher_cache = load_teacher_cache(cache_path)
            logger.info("Loaded teacher cache: %d examples from %s", len(teacher_cache), cache_path)
        else:
            result_path = precompute_teacher_cache(
                model.tokenizer,
                cfg,
                local_teacher_model=model.base_model,
            )
            if result_path is not None:
                teacher_cache = load_teacher_cache(result_path)
                logger.info("Loaded teacher cache: %d examples", len(teacher_cache))
            else:
                logger.info("Gemini teacher unavailable, using self-distillation only")

    teacher_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Resume from checkpoint
    start_step = 0
    best_metric = float("inf")
    if cfg.resume_from:
        start_step, best_metric = _resume(model, optimizer, scheduler, cfg.resume_from)
    elif cfg.tcs_checkpoint:
        _load_tcs_weights(model, cfg.tcs_checkpoint)

    # Mixed precision
    use_amp = cfg.mixed_precision != "no"
    amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision == "fp16"))

    # Training
    model.train()
    data_iter = _infinite_iter(train_loader)
    optimizer.zero_grad()

    phase_dir = cfg.phase_output_dir
    phase_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(range(start_step, cfg.max_steps), desc=f"Phase: {cfg.phase}")
    running_loss = 0.0
    teacher_idx = 0  # Track position in teacher cache

    for step in progress:
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss_dict = _forward_step(model, batch, loss_fn, cfg)

            # Add auxiliary Gemini teacher CE loss
            if teacher_cache is not None and cfg.phase in ("tcs-pretrain", "tcs-lora"):
                teacher_loss = _compute_teacher_loss(
                    model, batch, teacher_cache, teacher_idx,
                    teacher_ce, device,
                )
                if teacher_loss is not None:
                    loss_dict["loss"] = loss_dict["loss"] + cfg.teacher_loss_weight * teacher_loss
                    loss_dict["teacher_loss"] = teacher_loss.detach()
                teacher_idx = (teacher_idx + batch["input_ids"].size(0)) % len(teacher_cache)

        loss = loss_dict["loss"] / cfg.gradient_accumulation_steps

        # Backward
        scaler.scale(loss).backward()

        # Accumulate gradients
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += loss_dict["loss"].item()

        # Logging
        if (step + 1) % cfg.logging_steps == 0:
            avg_loss = running_loss / cfg.logging_steps
            lr = scheduler.get_last_lr()[0]
            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
            running_loss = 0.0

        # Validation
        if (step + 1) % cfg.eval_steps == 0:
            val_metric = _validate(model, val_loader, loss_fn, cfg, device, amp_dtype, use_amp)
            logger.info("Step %d — val metric: %.4f", step + 1, val_metric)

            if val_metric < best_metric:
                best_metric = val_metric
                model.save_trainable(phase_dir / "best")
                logger.info("New best metric: %.4f", best_metric)

            model.train()

        # Checkpointing
        if (step + 1) % cfg.save_steps == 0:
            _save_step(model, optimizer, scheduler, step + 1, best_metric, cfg, phase_dir)

    # Final save
    _save_step(model, optimizer, scheduler, cfg.max_steps, best_metric, cfg, phase_dir)
    model.save_trainable(phase_dir / "final")

    # Cleanup: delete intermediate checkpoints to free disk
    if cfg.cleanup_checkpoints_on_finish:
        cleanup_phase_checkpoints(phase_dir)

    logger.info("Training complete. Best metric: %.4f", best_metric)


def _forward_step(
    model: Gemma4WithTCS,
    batch: dict[str, torch.Tensor],
    loss_fn: torch.nn.Module,
    cfg: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Execute one forward step based on the training phase."""
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch.get("labels")

    if cfg.phase in ("tcs-pretrain", "tcs-lora"):
        # Self-distillation: get both compressed and uncompressed outputs
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mode="both",
        )
        return loss_fn(
            student_logits=outputs["student_logits"],
            teacher_logits=outputs["teacher_logits"],
            targets=labels,
        )
    else:
        # Tool calling: standard compressed forward + CE loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode="compressed",
        )
        return loss_fn(logits=outputs["logits"], targets=labels)


def _compute_teacher_loss(
    model: Gemma4WithTCS,
    batch: dict[str, torch.Tensor],
    teacher_cache: list[torch.Tensor],
    cache_start_idx: int,
    ce_fn: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Compute auxiliary CE loss against pre-computed Gemini teacher labels.

    Uses sequence-level distillation: the student's compressed-path logits
    are compared against the teacher's token predictions via CE loss.
    """
    batch_size = batch["input_ids"].size(0)
    max_len = batch["input_ids"].size(1)
    cache_len = len(teacher_cache)

    if cache_len == 0:
        return None

    # Gather teacher labels for this batch
    teacher_ids_list = []
    for i in range(batch_size):
        idx = (cache_start_idx + i) % cache_len
        t_ids = teacher_cache[idx]
        # Pad or truncate to match sequence length
        if len(t_ids) < max_len:
            padded = torch.full((max_len,), -100, dtype=torch.long)
            padded[:len(t_ids)] = t_ids
            teacher_ids_list.append(padded)
        else:
            teacher_ids_list.append(t_ids[:max_len])

    teacher_labels = torch.stack(teacher_ids_list).to(device)

    # Get student logits from compressed path
    with torch.no_grad():
        student_out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            mode="compressed",
        )

    student_logits = student_out["logits"]

    # Align sequence lengths if TCS compressed them
    if student_logits.size(1) != max_len:
        indices = torch.linspace(
            0, max_len - 1, student_logits.size(1),
            device=device,
        ).long()
        teacher_labels = teacher_labels[:, indices]

    V = student_logits.size(-1)
    return ce_fn(student_logits.reshape(-1, V), teacher_labels.reshape(-1))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    model: Gemma4WithTCS,
    val_loader,
    loss_fn: torch.nn.Module,
    cfg: TrainingConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    """Run validation and return the average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss_dict = _forward_step(model, batch, loss_fn, cfg)

        total_loss += loss_dict["loss"].item()
        num_batches += 1

        if num_batches >= 50:  # Cap validation length
            break

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_train_loader(tokenizer, cfg: TrainingConfig):
    """Build training DataLoader."""
    if cfg.phase in ("tcs-pretrain", "tcs-lora"):
        dataset = TextDataset(
            tokenizer=tokenizer,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            split="train",
            max_seq_len=512,
        )
    else:
        dataset = ToolCallingDataset(
            tokenizer=tokenizer,
            data_path="data/tool_calling_train.jsonl",
            max_seq_len=512,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


def _build_val_loader(tokenizer, cfg: TrainingConfig):
    """Build validation DataLoader."""
    if cfg.phase in ("tcs-pretrain", "tcs-lora"):
        dataset = TextDataset(
            tokenizer=tokenizer,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            split="validation",
            max_seq_len=512,
            max_samples=cfg.max_eval_samples,
        )
    else:
        dataset = ToolCallingDataset(
            tokenizer=tokenizer,
            data_path="data/tool_calling_val.jsonl",
            max_seq_len=512,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


def _build_scheduler(optimizer, cfg: TrainingConfig):
    """Build learning rate scheduler with warmup."""
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.warmup_steps)
    main_steps = max(cfg.max_steps - cfg.warmup_steps, 1)

    if cfg.scheduler == "cosine":
        main_sched = CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=cfg.lr * 0.01)
    else:
        main_sched = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=main_steps)

    return SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[cfg.warmup_steps])


def _infinite_iter(loader):
    """Infinite iterator over a DataLoader."""
    while True:
        yield from loader


def _resume(model, optimizer, scheduler, checkpoint_path: str) -> tuple[int, float]:
    """Resume training from a checkpoint."""
    state = load_checkpoint(checkpoint_path)
    model.load_state_dict(state.model_state, strict=False)
    optimizer.load_state_dict(state.optimizer_state)
    if scheduler and state.scheduler_state:
        scheduler.load_state_dict(state.scheduler_state)
    logger.info("Resumed from step %d (best=%.4f)", state.step, state.best_metric)
    return state.step, state.best_metric


def _load_tcs_weights(model: Gemma4WithTCS, path: str) -> None:
    """Load pre-trained TCS weights (from Phase 1 → Phase 2)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded TCS weights from %s (missing=%d)", path, len(missing))


def _save_step(model, optimizer, scheduler, step, best_metric, cfg, phase_dir):
    """Save a training checkpoint with rotation (keep last N)."""
    trainable_state = {k: v for k, v in model.state_dict().items()
                       if any(k.startswith(p) for p in ("tcs", "compressed_pos_emb", "decompressor"))}

    # Also include LoRA params
    for k, v in model.state_dict().items():
        if "lora" in k.lower():
            trainable_state[k] = v

    save_checkpoint(
        CheckpointState(
            step=step,
            model_state=trainable_state,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler else None,
            best_metric=best_metric,
            config={"phase": cfg.phase, "lr": cfg.lr, "max_steps": cfg.max_steps},
        ),
        save_dir=phase_dir / "checkpoints",
        max_to_keep=cfg.max_checkpoints_to_keep,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gemma 4 Memory Companion")

    # Phase
    parser.add_argument("--phase", choices=["tcs-pretrain", "tcs-lora", "tool-calling"],
                        default="tcs-pretrain", help="Training phase")

    # Model
    parser.add_argument("--model", default="google/gemma-4-e2b-it", help="Base model name")
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")

    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")

    # Distillation
    parser.add_argument("--distill-alpha", type=float, default=None)
    parser.add_argument("--distill-temperature", type=float, default=2.0)

    # Checkpoints
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--tcs-checkpoint", default=None)
    parser.add_argument("--lora-checkpoint", default=None)

    # Gemini API teacher
    parser.add_argument("--use-gemini-teacher", action="store_true",
                        help="Use Gemini API as auxiliary teacher for distillation")
    parser.add_argument("--gemini-api-key", default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--gemini-teacher-model", default="gemma-4-31b-it")
    parser.add_argument("--gemini-rpm", type=int, default=14,
                        help="Gemini API requests per minute limit")
    parser.add_argument("--teacher-loss-weight", type=float, default=0.3,
                        help="Weight (beta) for auxiliary teacher CE loss")

    # Data
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Build configs
    comp_cfg = CompressionConfig(ratio=args.compression_ratio)
    lora_cfg = LoraConfig(
        enabled=(args.phase != "tcs-pretrain"),  # No LoRA in Phase 1
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    model_cfg = ModelConfig(
        base_model_name=args.model,
        compression=comp_cfg,
        lora=lora_cfg,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    train_cfg = TrainingConfig(
        phase=args.phase,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        max_grad_norm=args.max_grad_norm,
        distill_temperature=args.distill_temperature,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        tcs_checkpoint=args.tcs_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        use_gemini_teacher=args.use_gemini_teacher,
        gemini_teacher_model=args.gemini_teacher_model,
        gemini_api_key=args.gemini_api_key,
        gemini_rpm_limit=args.gemini_rpm,
        teacher_loss_weight=args.teacher_loss_weight,
    )

    # Apply phase presets for unset values
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.warmup_steps is not None:
        train_cfg.warmup_steps = args.warmup_steps
    if args.save_steps is not None:
        train_cfg.save_steps = args.save_steps
    if args.eval_steps is not None:
        train_cfg.eval_steps = args.eval_steps
    if args.distill_alpha is not None:
        train_cfg.distill_alpha = args.distill_alpha

    train_cfg = apply_phase_preset(train_cfg)

    logger.info("Phase: %s", train_cfg.phase)
    logger.info("Config: lr=%.2e, steps=%d, batch=%d×%d",
                train_cfg.lr, train_cfg.max_steps,
                train_cfg.batch_size, train_cfg.gradient_accumulation_steps)

    # Build model
    model = Gemma4WithTCS(model_cfg)

    # Train
    train(model, train_cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
