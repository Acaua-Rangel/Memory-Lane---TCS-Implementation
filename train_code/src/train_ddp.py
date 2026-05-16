"""
Multi-GPU DDP training script using HuggingFace Accelerate.

Designed for 2× T4 on Kaggle. Wraps the single-GPU training logic
with accelerate for distributed data parallel.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import logging
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

from src.config import (
    ModelConfig,
    CompressionConfig,
    LoraConfig,
    TrainingConfig,
    apply_phase_preset,
)
from src.data import TextDataset, ToolCallingDataset
from src.losses import build_loss
from src.model import Gemma4WithTCS
from src.utils.checkpoint import (
    CheckpointState,
    cleanup_phase_checkpoints,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


class RecoverableOOMError(RuntimeError):
    """Signal that training can be retried with a smaller batch size."""

    def __init__(self, step: int, message: str | None = None) -> None:
        self.step = step
        if message is None:
            message = f"Recoverable CUDA OOM at step {step}."
        super().__init__(message)


def _log_failure_context() -> None:
    """Log rank and CUDA memory context when a worker fails."""
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    logger.exception("DDP worker failed on local_rank=%d", local_rank)

    if not torch.cuda.is_available():
        return

    try:
        device_idx = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
        logger.error(
            "CUDA memory at failure (device=%d): allocated=%.2f GB reserved=%.2f GB",
            device_idx, allocated, reserved,
        )
    except Exception:
        logger.exception("Failed to collect CUDA memory diagnostics")


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True when an exception corresponds to a CUDA OOM condition."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True

    message = str(exc).lower()
    if "cuda" not in message:
        return False
    if "out of memory" not in message:
        return False

    return True


def _did_any_rank_oom(accelerator: Accelerator, local_oom: bool) -> bool:
    """Synchronize a local OOM flag and report if any rank encountered OOM."""
    oom_flag = torch.tensor(float(local_oom), device=accelerator.device)
    global_flag = accelerator.reduce(oom_flag, reduction="max")
    return bool(global_flag.item() >= 0.999)


def _clear_cuda_memory(optimizer: torch.optim.Optimizer | None = None) -> None:
    """Best-effort memory cleanup after OOM before retrying."""
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    gc.collect()
    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()


def _should_retry_oom_batch_fallback(cfg: TrainingConfig, step: int) -> bool:
    """Allow retries only in the configured early-step window."""
    if not cfg.ddp_auto_batch_retry_on_oom:
        return False
    if step > cfg.ddp_oom_retry_max_step:
        return False

    return True


def _next_oom_retry_config(cfg: TrainingConfig, target_effective_batch: int) -> TrainingConfig | None:
    """Compute the next retry config by halving batch size and scaling accumulation."""
    if cfg.batch_size <= cfg.ddp_oom_retry_min_batch_size:
        return None

    next_batch = max(cfg.batch_size // 2, cfg.ddp_oom_retry_min_batch_size)
    if next_batch == cfg.batch_size:
        return None

    next_grad_accum = max(math.ceil(target_effective_batch / next_batch), 1)
    return replace(
        cfg,
        batch_size=next_batch,
        gradient_accumulation_steps=next_grad_accum,
    )


def _train_with_oom_batch_retry(model_cfg: ModelConfig, cfg: TrainingConfig) -> None:
    """Run DDP training and retry with smaller micro-batch when early OOM occurs."""
    current_cfg = cfg
    target_effective_batch = cfg.effective_batch_size
    retry_count = 0

    while True:
        model = Gemma4WithTCS(model_cfg)
        try:
            train_ddp(model, current_cfg)
            return
        except RecoverableOOMError as exc:
            if not current_cfg.ddp_auto_batch_retry_on_oom:
                raise
            if retry_count >= current_cfg.ddp_oom_retry_max_attempts:
                raise

            next_cfg = _next_oom_retry_config(current_cfg, target_effective_batch)
            if next_cfg is None:
                raise

            retry_count += 1
            logger.warning(
                "Recoverable OOM at step %d. Retry %d/%d with batch_size=%d "
                "gradient_accumulation_steps=%d (effective batch=%d).",
                exc.step,
                retry_count,
                current_cfg.ddp_oom_retry_max_attempts,
                next_cfg.batch_size,
                next_cfg.gradient_accumulation_steps,
                next_cfg.effective_batch_size,
            )
            del model
            _clear_cuda_memory()
            current_cfg = next_cfg


def _build_accelerator(cfg: TrainingConfig) -> Accelerator:
    """Build Accelerator with DDP settings compatible with optional trainable blocks."""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    mixed_precision = cfg.mixed_precision
    if mixed_precision == "no":
        mixed_precision = None

    return Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=None,
        kwargs_handlers=[ddp_kwargs],
    )


def train_ddp(model: Gemma4WithTCS, cfg: TrainingConfig) -> None:
    """DDP training loop using HuggingFace Accelerate."""
    accelerator = _build_accelerator(cfg)
    set_seed(42)

    is_main = accelerator.is_main_process
    device = accelerator.device

    if is_main:
        logger.info("DDP Training — %d processes", accelerator.num_processes)
        logger.info("Phase: %s, LR: %.2e, Steps: %d", cfg.phase, cfg.lr, cfg.max_steps)

    # DataLoaders
    train_loader = _build_loader(model.tokenizer, cfg, split="train")
    val_loader = _build_loader(model.tokenizer, cfg, split="validation")

    # Optimizer (only trainable params)
    params = model.get_trainable_params(cfg.phase)
    optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Scheduler
    scheduler = _build_scheduler(optimizer, cfg)

    # Loss
    loss_fn = build_loss(
        phase=cfg.phase,
        alpha=cfg.distill_alpha,
        temperature=cfg.distill_temperature,
    )

    # Gemini API teacher (optional)
    teacher_cache: list[torch.Tensor] | None = None
    if cfg.use_gemini_teacher and cfg.phase in ("tcs-pretrain", "tcs-lora") and is_main:
        from src.train import precompute_teacher_cache, load_teacher_cache
        cache_path = Path(cfg.teacher_cache_dir) / f"teacher_{cfg.phase}.pt"
        if cache_path.exists():
            teacher_cache = load_teacher_cache(cache_path)
            logger.info("Loaded teacher cache: %d examples", len(teacher_cache))
        else:
            result_path = precompute_teacher_cache(model.tokenizer, cfg)
            if result_path is not None:
                teacher_cache = load_teacher_cache(result_path)
                logger.info("Loaded teacher cache: %d examples", len(teacher_cache))
            else:
                logger.info("Gemini teacher unavailable, using self-distillation only")

    teacher_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Load TCS weights from previous phase if provided
    if cfg.tcs_checkpoint:
        state = torch.load(cfg.tcs_checkpoint, map_location="cpu", weights_only=True)
        missing, _ = model.load_state_dict(state, strict=False)
        if is_main:
            logger.info("Loaded TCS from %s (missing=%d)", cfg.tcs_checkpoint, len(missing))

    # Prepare with accelerate
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Training loop
    phase_dir = cfg.phase_output_dir
    if is_main:
        phase_dir.mkdir(parents=True, exist_ok=True)

    data_iter = _infinite_iter(train_loader)
    best_metric = float("inf")
    running_loss = 0.0
    teacher_idx = 0

    progress = tqdm(range(cfg.max_steps), desc=f"Phase: {cfg.phase}", disable=not is_main)

    for step in progress:
        model.train()
        batch = next(data_iter)
        step_id = step + 1

        with accelerator.accumulate(model):
            loss_dict = None
            step_error: RuntimeError | None = None

            try:
                loss_dict = _forward_step(model, batch, loss_fn, cfg, accelerator)

                # Auxiliary Gemini teacher CE loss
                if teacher_cache is not None and cfg.phase in ("tcs-pretrain", "tcs-lora"):
                    from src.train import _compute_teacher_loss
                    unwrapped = accelerator.unwrap_model(model)
                    t_loss = _compute_teacher_loss(
                        unwrapped, batch, teacher_cache, teacher_idx,
                        teacher_ce, accelerator.device,
                    )
                    if t_loss is not None:
                        loss_dict["loss"] = loss_dict["loss"] + cfg.teacher_loss_weight * t_loss
                    teacher_idx = (teacher_idx + batch["input_ids"].size(0)) % len(teacher_cache)
            except RuntimeError as exc:
                if not _is_cuda_oom_error(exc):
                    raise
                step_error = exc
            except torch.OutOfMemoryError as exc:
                step_error = exc

            if _did_any_rank_oom(accelerator, step_error is not None):
                _clear_cuda_memory(optimizer)
                if _should_retry_oom_batch_fallback(cfg, step_id):
                    raise RecoverableOOMError(step_id) from step_error
                if step_error is not None:
                    raise step_error
                raise RuntimeError("Another rank reported CUDA OOM and retry is disabled for this step.")

            if loss_dict is None:
                raise RuntimeError("Forward step did not produce a loss dictionary.")

            loss = loss_dict["loss"]
            try:
                accelerator.backward(loss)
            except RuntimeError as exc:
                if not _is_cuda_oom_error(exc):
                    raise
                _clear_cuda_memory(optimizer)
                if _should_retry_oom_batch_fallback(cfg, step_id):
                    raise RecoverableOOMError(step_id) from exc
                raise
            except torch.OutOfMemoryError as exc:
                _clear_cuda_memory(optimizer)
                if _should_retry_oom_batch_fallback(cfg, step_id):
                    raise RecoverableOOMError(step_id) from exc
                raise

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params, cfg.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss_dict["loss"].item()

        # Logging
        if is_main and (step + 1) % cfg.logging_steps == 0:
            avg_loss = running_loss / cfg.logging_steps
            lr = scheduler.get_last_lr()[0]
            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
            running_loss = 0.0

        # Validation
        if (step + 1) % cfg.eval_steps == 0:
            val_metric = _validate_ddp(model, val_loader, loss_fn, cfg, accelerator)

            if is_main:
                logger.info("Step %d — val metric: %.4f", step + 1, val_metric)
                if val_metric < best_metric:
                    best_metric = val_metric
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_trainable(phase_dir / "best")
                    logger.info("New best: %.4f", best_metric)

        # Checkpointing
        if is_main and (step + 1) % cfg.save_steps == 0:
            unwrapped = accelerator.unwrap_model(model)
            _save_step_ddp(unwrapped, optimizer, scheduler, step + 1, best_metric, cfg, phase_dir)

    # Final save
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        _save_step_ddp(unwrapped, optimizer, scheduler, cfg.max_steps, best_metric, cfg, phase_dir)
        unwrapped.save_trainable(phase_dir / "final")

        # Cleanup: delete intermediate checkpoints to free disk for next phase
        if cfg.cleanup_checkpoints_on_finish:
            cleanup_phase_checkpoints(phase_dir)

        logger.info("DDP Training complete. Best metric: %.4f", best_metric)

    accelerator.end_training()


def _forward_step(model, batch, loss_fn, cfg, accelerator):
    """Forward step adapted for accelerate."""
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = batch.get("labels")

    if cfg.phase in ("tcs-pretrain", "tcs-lora"):
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
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mode="compressed",
        )
        return loss_fn(logits=outputs["logits"], targets=labels)


@torch.no_grad()
def _validate_ddp(model, val_loader, loss_fn, cfg, accelerator):
    """Distributed validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        loss_dict = _forward_step(model, batch, loss_fn, cfg, accelerator)
        total_loss += loss_dict["loss"].item()
        num_batches += 1
        if num_batches >= 50:
            break

    # Gather across processes
    avg = torch.tensor(total_loss / max(num_batches, 1), device=accelerator.device)
    avg = accelerator.reduce(avg, reduction="mean")
    return avg.item()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_loader(tokenizer, cfg, split):
    if cfg.phase in ("tcs-pretrain", "tcs-lora"):
        dataset = TextDataset(
            tokenizer=tokenizer,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            split=split,
            max_seq_len=512,
            max_samples=cfg.max_eval_samples if split == "validation" else None,
        )
    else:
        data_file = f"data/tool_calling_{split}.jsonl"
        if split == "validation":
            data_file = "data/tool_calling_val.jsonl"
        dataset = ToolCallingDataset(
            tokenizer=tokenizer,
            data_path=data_file,
            max_seq_len=512,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


def _build_scheduler(optimizer, cfg):
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=cfg.warmup_steps)
    main_steps = max(cfg.max_steps - cfg.warmup_steps, 1)
    main_sched = CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=cfg.lr * 0.01)
    return SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[cfg.warmup_steps])


def _infinite_iter(loader):
    while True:
        yield from loader


def _save_step_ddp(model, optimizer, scheduler, step, best_metric, cfg, phase_dir):
    trainable_state = {}
    for k, v in model.state_dict().items():
        if any(k.startswith(p) for p in ("tcs", "compressed_pos_emb", "decompressor")):
            trainable_state[k] = v
        elif "lora" in k.lower():
            trainable_state[k] = v

    save_checkpoint(
        CheckpointState(
            step=step,
            model_state=trainable_state,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler else None,
            best_metric=best_metric,
            config={"phase": cfg.phase, "lr": cfg.lr},
        ),
        save_dir=phase_dir / "checkpoints",
        max_to_keep=cfg.max_checkpoints_to_keep,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DDP Train Gemma 4 Memory Companion")

    parser.add_argument("--phase", choices=["tcs-pretrain", "tcs-lora", "tool-calling"],
                        default="tcs-pretrain")
    parser.add_argument("--model", default="google/gemma-4-e2b-it")
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--disable-oom-batch-retry",
        action="store_true",
        help="Disable auto-retry with reduced micro-batch when early CUDA OOM occurs.",
    )
    parser.add_argument(
        "--oom-retry-max-attempts",
        type=int,
        default=3,
        help="Maximum number of OOM-triggered retry attempts.",
    )
    parser.add_argument(
        "--oom-retry-max-step",
        type=int,
        default=1,
        help="Only retry if OOM happens up to this 1-based step.",
    )
    parser.add_argument(
        "--oom-retry-min-batch-size",
        type=int,
        default=1,
        help="Lowest micro-batch size allowed during OOM fallback.",
    )

    parser.add_argument("--distill-alpha", type=float, default=None)
    parser.add_argument("--distill-temperature", type=float, default=2.0)

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
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

    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")

    return parser.parse_args()


@record
def main():
    faulthandler.enable()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    comp_cfg = CompressionConfig(ratio=args.compression_ratio)
    lora_cfg = LoraConfig(
        enabled=(args.phase != "tcs-pretrain"),
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
        ddp_auto_batch_retry_on_oom=not args.disable_oom_batch_retry,
        ddp_oom_retry_max_attempts=args.oom_retry_max_attempts,
        ddp_oom_retry_max_step=args.oom_retry_max_step,
        ddp_oom_retry_min_batch_size=args.oom_retry_min_batch_size,
        mixed_precision=args.mixed_precision,
        distill_temperature=args.distill_temperature,
        output_dir=args.output_dir,
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

    try:
        _train_with_oom_batch_retry(model_cfg, train_cfg)
    except Exception:
        _log_failure_context()
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
