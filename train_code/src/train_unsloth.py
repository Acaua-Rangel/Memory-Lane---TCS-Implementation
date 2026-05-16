"""
Unsloth-accelerated training for Gemma 4 Memory Companion.

Replaces HuggingFace PEFT with Unsloth for 2× faster LoRA training and
60% less memory usage. This is critical for Kaggle's constrained environment
(2× T4, 30GB RAM, 19.5GB disk).

Usage:
    poetry run train-unsloth --phase tcs-lora
    poetry run train-unsloth --phase tool-calling

Note: Phase 1 (tcs-pretrain) does NOT use Unsloth since it only trains
the TCS layer (no LoRA). Use the standard train.py for Phase 1.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.config import (
    ModelConfig,
    CompressionConfig,
    LoraConfig,
    TrainingConfig,
    apply_phase_preset,
)
from src.data import build_dataloader
from src.losses import build_loss
from src.utils.checkpoint import (
    CheckpointState,
    cleanup_phase_checkpoints,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unsloth Model Loading
# ---------------------------------------------------------------------------

def _resolve_text_tokenizer(tokenizer_or_processor: Any) -> Any:
    """Return a text tokenizer exposing encode/decode for dataset builders."""
    if hasattr(tokenizer_or_processor, "encode"):
        return tokenizer_or_processor

    if hasattr(tokenizer_or_processor, "tokenizer"):
        nested_tokenizer = tokenizer_or_processor.tokenizer
        if hasattr(nested_tokenizer, "encode"):
            logger.info(
                "Unsloth returned a processor; using nested text tokenizer for dataset tokenization"
            )
            return nested_tokenizer

    raise TypeError(
        "Unsloth tokenizer object must expose encode(), or provide tokenizer.encode() "
        "when returning a processor."
    )


def _resolve_amp_settings(mixed_precision: str, device: torch.device) -> tuple[bool, torch.dtype]:
    """Resolve AMP enablement and dtype with GPU capability fallbacks."""
    if mixed_precision == "no":
        return False, torch.float16

    if device.type != "cuda":
        logger.warning("AMP requested but CUDA is unavailable; disabling AMP")
        return False, torch.float16

    if mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        logger.warning(
            "Current CUDA device does not support bfloat16; falling back to float16 autocast"
        )
        return True, torch.float16

    if mixed_precision == "bf16":
        return True, torch.bfloat16

    return True, torch.float16


def _validate_unsloth_launch() -> None:
    """Warn about single-GPU usage and reject distributed launches."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = os.environ.get("LOCAL_RANK")

    if world_size > 1 or local_rank is not None:
        raise RuntimeError(
            "src.train_unsloth does not support distributed launches. "
            "Use torchrun --standalone --nproc_per_node=<num_gpus> -m src.train_ddp "
            "to use multiple GPUs."
        )

    if not torch.cuda.is_available():
        return

    device_count = torch.cuda.device_count()
    if device_count < 2:
        return

    logger.warning(
        "Detected %d CUDA devices, but src.train_unsloth runs in single-process mode "
        "and will use only one GPU. Use torchrun --standalone --nproc_per_node=%d "
        "-m src.train_ddp to train on all visible GPUs.",
        device_count,
        device_count,
    )

def _load_model_with_unsloth(
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
):
    """
    Load Gemma 4 with Unsloth's FastLanguageModel + apply LoRA.

    Unsloth patches the model for:
    - 2× faster training via fused kernels
    - 60% less VRAM via smarter gradient checkpointing
    - Native 4-bit / 8-bit quantization support
    """
    if training_cfg.phase in ("tcs-pretrain", "tcs-lora"):
        if os.environ.get("UNSLOTH_RETURN_LOGITS") != "1":
            os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
            logger.info(
                "Set UNSLOTH_RETURN_LOGITS=1 for distillation phase to ensure logits are available"
            )

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Unsloth is required for this training script. "
            "Install it: pip install unsloth"
        )

    # Determine quantization
    load_in_4bit = model_cfg.load_in_4bit
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(model_cfg.torch_dtype, torch.bfloat16)

    logger.info(
        "Loading model with Unsloth: %s (4-bit=%s, dtype=%s)",
        model_cfg.base_model_name,
        load_in_4bit,
        model_cfg.torch_dtype,
    )

    # Unsloth handles model loading + LoRA application in one call
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg.base_model_name,
        max_seq_length=model_cfg.max_seq_len,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = _resolve_text_tokenizer(tokenizer)

    # Apply LoRA via Unsloth (uses optimized kernels)
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_cfg.lora.rank,
        lora_alpha=model_cfg.lora.alpha,
        lora_dropout=model_cfg.lora.dropout,
        target_modules=model_cfg.lora.target_modules,
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized GC
        random_state=42,
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# TCS Wrapper for Unsloth Model
# ---------------------------------------------------------------------------

class UnslothWithTCS(torch.nn.Module):
    """
    Wraps an Unsloth-loaded model with our TCS compression layer.

    Unsloth handles the LoRA + base model optimization. We add TCS on top
    for token compression. This gives us the best of both worlds:
    - Unsloth's 2× training speed for LoRA
    - TCS's 4-16× inference speed from token compression
    """

    def __init__(self, unsloth_model, tokenizer, model_cfg: ModelConfig):
        super().__init__()
        self.base_model = unsloth_model
        self.tokenizer = tokenizer
        self.cfg = model_cfg

        # Resolve text backbone and LM head for multimodal-safe compressed forward.
        self._lm = self._resolve_text_backbone()
        self._lm_head = self._find_lm_head()
        self._rotary_emb: torch.nn.Module | None = None
        self._decoder_layers, self._decoder_norm = self._find_decoder_components()

        # Get hidden size
        self.d_model = self._resolve_hidden_size(unsloth_model.config)
        self._embed_scale: float = self.d_model ** 0.5 if self._decoder_layers is not None else 1.0

        if self._decoder_layers is not None:
            logger.info(
                "Unsloth compressed forward will use direct decoder bypass (%d layers)",
                len(self._decoder_layers),
            )
        else:
            logger.warning(
                "Could not resolve decoder layers for direct bypass; compressed inputs_embeds path may be memory intensive"
            )

        # Import and create TCS
        from src.model import CompressionLayer, DecompressionLayer, RMSNorm

        if model_cfg.compression.enabled:
            self.tcs = CompressionLayer(self.d_model, model_cfg.compression)
            self.decompressor = DecompressionLayer(self.d_model, model_cfg.compression)
            comp_len = math.ceil(model_cfg.max_seq_len / model_cfg.compression.ratio) + 1
            self.compressed_pos_emb = torch.nn.Embedding(comp_len, self.d_model)
        else:
            self.tcs = None

    @staticmethod
    def _resolve_hidden_size(config: Any) -> int:
        """Resolve hidden size from text or multimodal model configs."""
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return config.text_config.hidden_size

        raise AttributeError(
            f"Cannot determine hidden_size from {type(config).__name__}"
        )

    def _walk_wrapped_objects(self, root: Any):
        """Yield nested wrapper objects reachable via common wrapper attributes."""
        if not isinstance(root, torch.nn.Module):
            return

        stack = [root]
        seen: set[int] = set()

        while stack:
            obj = stack.pop()
            if obj is None:
                continue

            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)

            yield obj

            for attr in ("model", "base_model", "language_model"):
                child = getattr(obj, attr, None)
                if child is None:
                    continue
                if not isinstance(child, torch.nn.Module):
                    continue
                stack.append(child)

    def _resolve_text_backbone(self):
        """Resolve the text language model from nested PEFT/Unsloth wrappers."""
        for candidate in self._walk_wrapped_objects(self.base_model):
            nested_model = getattr(candidate, "model", None)
            nested_lm = getattr(nested_model, "language_model", None)
            if isinstance(nested_lm, torch.nn.Module):
                return nested_lm

            direct_lm = getattr(candidate, "language_model", None)
            if isinstance(direct_lm, torch.nn.Module):
                return direct_lm

        logger.warning("Could not resolve nested text backbone; falling back to base model")
        return self.base_model

    def _find_lm_head(self):
        """Find LM head used to project hidden states to vocabulary logits."""
        for candidate in self._walk_wrapped_objects(self.base_model):
            lm_head = getattr(candidate, "lm_head", None)
            if isinstance(lm_head, torch.nn.Module):
                return lm_head

        lm_head = getattr(self._lm, "lm_head", None)
        if isinstance(lm_head, torch.nn.Module):
            return lm_head

        return None

    def _find_decoder_components(self):
        """Find decoder layers and final norm for direct layer-wise forward."""
        for obj in self._walk_wrapped_objects(self._lm):

            layers = getattr(obj, "layers", None)
            norm = getattr(obj, "norm", None)
            if isinstance(layers, (torch.nn.ModuleList, list, tuple)) and isinstance(norm, torch.nn.Module):
                self._rotary_emb = getattr(obj, "rotary_emb", None)
                return layers, norm

        return None, None

    def _hidden_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        if self._lm_head is not None:
            return self._lm_head(hidden_states)

        embed_weight = self._lm.get_input_embeddings().weight
        return torch.nn.functional.linear(hidden_states, embed_weight)

    def _compute_logits(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "logits"):
            return outputs.logits
        return self._hidden_to_logits(outputs.last_hidden_state)

    def _forward_decoder_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Bypass multimodal per-layer input expansion for inputs_embeds paths."""
        B, M = hidden_states.shape[:2]
        device = hidden_states.device
        dtype = hidden_states.dtype

        hidden_states = hidden_states * self._embed_scale
        position_ids = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
        pos_emb_cache: dict[str | None, tuple[torch.Tensor, torch.Tensor]] = {}

        causal_mask = torch.zeros(1, 1, M, M, device=device, dtype=dtype)
        upper_mask = torch.triu(
            torch.ones(M, M, device=device, dtype=torch.bool),
            diagonal=1,
        )
        causal_mask = causal_mask.masked_fill(upper_mask, torch.finfo(dtype).min)

        per_layer_input = torch.ones(1, device=device, dtype=dtype)
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for layer in self._decoder_layers:
            layer_type = getattr(layer, "layer_type", None)
            if layer_type is None:
                self_attn = getattr(layer, "self_attn", None)
                layer_type = getattr(self_attn, "layer_type", None) if self_attn else None

            position_embeddings = None
            if self._rotary_emb is not None:
                if layer_type not in pos_emb_cache:
                    pos_emb_cache[layer_type] = self._rotary_emb(
                        hidden_states,
                        position_ids,
                        layer_type=layer_type,
                    )
                position_embeddings = pos_emb_cache[layer_type]

            layer_out = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
                use_cache=False,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_states = self._decoder_norm(hidden_states)
        return {"logits": self._hidden_to_logits(hidden_states), "loss": None}

    def _forward_from_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self._decoder_layers is not None:
            return self._forward_decoder_layers(embeds, attention_mask)

        outputs = self._lm(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "logits": self._compute_logits(outputs),
            "loss": getattr(outputs, "loss", None),
        }

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._lm.get_input_embeddings()(input_ids)

    def forward_compressed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward with TCS compression."""
        embeds = self._get_embeddings(input_ids)
        compressed = self.tcs(embeds)
        M = compressed.size(1)
        pos_ids = torch.arange(M, device=compressed.device).unsqueeze(0)
        compressed = compressed + self.compressed_pos_emb(pos_ids)

        comp_mask = None
        if attention_mask is not None:
            B = attention_mask.size(0)
            comp_mask = torch.ones(B, M, device=attention_mask.device, dtype=attention_mask.dtype)

        result = self._forward_from_embeds(
            embeds=compressed,
            attention_mask=comp_mask,
            labels=None,
        )
        result["compressed_embeds"] = compressed
        return result

    def forward_uncompressed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Standard forward without TCS (teacher path)."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"logits": outputs.logits, "loss": getattr(outputs, "loss", None)}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mode: str = "compressed",
    ) -> dict[str, torch.Tensor]:
        if mode == "uncompressed" or self.tcs is None:
            return self.forward_uncompressed(input_ids, attention_mask, labels)
        if mode == "compressed":
            return self.forward_compressed(input_ids, attention_mask)

        # mode == "both" for self-distillation
        with torch.no_grad():
            teacher = self.forward_uncompressed(input_ids, attention_mask, labels)
        student = self.forward_compressed(input_ids, attention_mask)

        return {
            "student_logits": student["logits"],
            "teacher_logits": teacher["logits"].detach(),
            "compressed_embeds": student.get("compressed_embeds"),
        }

    def save_trainable(self, save_dir: str | Path) -> None:
        """Save TCS weights + Unsloth LoRA adapters."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save TCS
        if self.tcs is not None:
            tcs_state = {
                "tcs": self.tcs.state_dict(),
                "pos_emb": self.compressed_pos_emb.state_dict(),
                "decompressor": self.decompressor.state_dict(),
            }
            torch.save(tcs_state, save_dir / "tcs_weights.pt")

        # Save LoRA via Unsloth
        self.base_model.save_pretrained(save_dir / "lora_adapters")
        logger.info("Saved trainable weights to %s", save_dir)

    def get_trainable_params(self, phase: str) -> list[dict]:
        """Get trainable parameters for optimizer."""
        tcs_params = []
        lora_params = []

        if self.tcs is not None:
            tcs_params = [
                {"params": self.tcs.parameters(), "lr_scale": 1.0},
                {"params": self.compressed_pos_emb.parameters(), "lr_scale": 1.0},
                {"params": self.decompressor.parameters(), "lr_scale": 1.0},
            ]

        if phase != "tcs-pretrain":
            lora_params = [
                {
                    "params": [
                        p for p in self.base_model.parameters() if p.requires_grad
                    ],
                    "lr_scale": 0.1,
                }
            ]

        return tcs_params + lora_params


def _load_tcs_checkpoint_into_unsloth_wrapper(
    model: UnslothWithTCS,
    checkpoint_state: dict[str, Any],
) -> None:
    """Load TCS weights from either nested or flat checkpoint formats."""
    if not isinstance(checkpoint_state, dict):
        raise TypeError("TCS checkpoint must be a dictionary")

    has_nested_format = all(
        key in checkpoint_state
        for key in ("tcs", "pos_emb", "decompressor")
    )
    if has_nested_format:
        model.tcs.load_state_dict(checkpoint_state["tcs"])
        model.compressed_pos_emb.load_state_dict(checkpoint_state["pos_emb"])
        model.decompressor.load_state_dict(checkpoint_state["decompressor"])
        return

    tcs_state = {
        key.removeprefix("tcs."): value
        for key, value in checkpoint_state.items()
        if key.startswith("tcs.")
    }
    pos_emb_state = {
        key.removeprefix("compressed_pos_emb."): value
        for key, value in checkpoint_state.items()
        if key.startswith("compressed_pos_emb.")
    }
    decompressor_state = {
        key.removeprefix("decompressor."): value
        for key, value in checkpoint_state.items()
        if key.startswith("decompressor.")
    }

    if not tcs_state or not pos_emb_state or not decompressor_state:
        raise KeyError(
            "Unsupported TCS checkpoint format. Expected nested keys "
            "{'tcs', 'pos_emb', 'decompressor'} or flat prefixes "
            "'tcs.', 'compressed_pos_emb.', and 'decompressor.'."
        )

    model.tcs.load_state_dict(tcs_state)
    model.compressed_pos_emb.load_state_dict(pos_emb_state)
    model.decompressor.load_state_dict(decompressor_state)


# ---------------------------------------------------------------------------
# Training Loop (Unsloth-optimized)
# ---------------------------------------------------------------------------

def train_unsloth(model_cfg: ModelConfig, train_cfg: TrainingConfig) -> None:
    """
    Training loop using Unsloth for LoRA optimization.

    Unsloth benefits:
    - 2× faster backward pass via fused CUDA kernels
    - 60% less VRAM via optimized gradient checkpointing
    - Native 4-bit training support
    """
    if train_cfg.phase == "tcs-pretrain":
        logger.warning(
            "Phase tcs-pretrain does not use LoRA. "
            "Use standard train.py for Phase 1 — Unsloth adds no benefit."
        )

    _validate_unsloth_launch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model with Unsloth
    unsloth_model, tokenizer = _load_model_with_unsloth(model_cfg, train_cfg)
    model = UnslothWithTCS(unsloth_model, tokenizer, model_cfg)

    # Load TCS checkpoint from previous phase
    if train_cfg.tcs_checkpoint:
        tcs_state = torch.load(train_cfg.tcs_checkpoint, map_location=device, weights_only=True)
        _load_tcs_checkpoint_into_unsloth_wrapper(model, tcs_state)
        logger.info("Loaded TCS from %s", train_cfg.tcs_checkpoint)

    # Move TCS to device
    if model.tcs is not None:
        model.tcs.to(device)
        model.compressed_pos_emb.to(device)
        model.decompressor.to(device)

    # Build DataLoader
    from src.data import build_dataloader
    train_loader = build_dataloader(
        tokenizer,
        train_cfg,
        max_seq_len=model_cfg.max_seq_len,
        split="train",
    )

    # Optimizer
    params = model.get_trainable_params(train_cfg.phase)
    optimizer = torch.optim.AdamW(
        [{"params": pg["params"], "lr": train_cfg.lr * pg.get("lr_scale", 1.0)}
         for pg in params],
        weight_decay=train_cfg.weight_decay,
    )

    # Loss
    loss_fn = build_loss(
        phase=train_cfg.phase,
        alpha=train_cfg.distill_alpha,
        temperature=train_cfg.distill_temperature,
    )

    # Gemini API teacher (optional)
    teacher_cache: list[torch.Tensor] | None = None
    if train_cfg.use_gemini_teacher and train_cfg.phase in ("tcs-pretrain", "tcs-lora"):
        from src.train import precompute_teacher_cache, load_teacher_cache
        cache_path = Path(train_cfg.teacher_cache_dir) / f"teacher_{train_cfg.phase}.pt"
        if cache_path.exists():
            teacher_cache = load_teacher_cache(cache_path)
            logger.info("Loaded teacher cache: %d examples", len(teacher_cache))
        else:
            result_path = precompute_teacher_cache(
                model.tokenizer,
                train_cfg,
                local_teacher_model=model.base_model,
            )
            if result_path is not None:
                teacher_cache = load_teacher_cache(result_path)
                logger.info("Loaded teacher cache: %d examples", len(teacher_cache))
            else:
                logger.info("Gemini teacher unavailable, using self-distillation only")

    teacher_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.max_steps, eta_min=train_cfg.lr * 0.01
    )

    # Mixed precision
    use_amp, amp_dtype = _resolve_amp_settings(train_cfg.mixed_precision, device)

    # Training loop
    model.train()
    phase_dir = train_cfg.phase_output_dir
    phase_dir.mkdir(parents=True, exist_ok=True)

    def _infinite_iter(loader):
        while True:
            yield from loader

    data_iter = _infinite_iter(train_loader)
    optimizer.zero_grad()

    progress = tqdm(range(train_cfg.max_steps), desc=f"[Unsloth] {train_cfg.phase}")
    running_loss = 0.0
    teacher_idx = 0

    for step in progress:
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            if train_cfg.phase in ("tcs-pretrain", "tcs-lora"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                    mode="both",
                )
                loss_dict = loss_fn(
                    student_logits=outputs["student_logits"],
                    teacher_logits=outputs["teacher_logits"],
                    targets=batch.get("labels"),
                )

                # Auxiliary Gemini teacher CE loss
                if teacher_cache is not None:
                    from src.train import _compute_teacher_loss
                    t_loss = _compute_teacher_loss(
                        model, batch, teacher_cache, teacher_idx,
                        teacher_ce, device,
                    )
                    if t_loss is not None:
                        loss_dict["loss"] = loss_dict["loss"] + train_cfg.teacher_loss_weight * t_loss
                        loss_dict["teacher_loss"] = t_loss.detach()
                    teacher_idx = (teacher_idx + batch["input_ids"].size(0)) % len(teacher_cache)
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    mode="compressed",
                )
                loss_dict = loss_fn(
                    logits=outputs["logits"],
                    targets=batch.get("labels"),
                )

        loss = loss_dict["loss"] / train_cfg.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for pg in params for p in pg["params"]], train_cfg.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += loss_dict["loss"].item()

        if (step + 1) % train_cfg.logging_steps == 0:
            avg = running_loss / train_cfg.logging_steps
            progress.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0.0

        if (step + 1) % train_cfg.save_steps == 0:
            trainable_state = {
                k: v for k, v in model.state_dict().items()
                if any(t in k.lower() for t in ("tcs", "compressed_pos_emb", "decompressor", "lora"))
            }
            save_checkpoint(
                CheckpointState(
                    step=step + 1,
                    model_state=trainable_state,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None,
                    best_metric=running_loss,
                    config={"phase": train_cfg.phase, "lr": train_cfg.lr, "max_steps": train_cfg.max_steps},
                ),
                save_dir=phase_dir / "checkpoints",
                max_to_keep=train_cfg.max_checkpoints_to_keep,
            )

    # Final save
    model.save_trainable(phase_dir / "final")

    if train_cfg.cleanup_checkpoints_on_finish:
        cleanup_phase_checkpoints(phase_dir)

    logger.info("[Unsloth] Training complete for phase: %s", train_cfg.phase)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Unsloth-accelerated training")
    parser.add_argument("--phase", required=True, choices=["tcs-pretrain", "tcs-lora", "tool-calling"])
    parser.add_argument("--model", default="google/gemma-4-e2b-it")
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
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

    args = parser.parse_args()

    # Build configs
    model_cfg = ModelConfig(
        base_model_name=args.model,
        compression=CompressionConfig(ratio=args.compression_ratio),
        lora=LoraConfig(rank=args.lora_rank, alpha=args.lora_alpha),
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    train_cfg = TrainingConfig(
        phase=args.phase,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        tcs_checkpoint=args.tcs_checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        use_gemini_teacher=args.use_gemini_teacher,
        gemini_teacher_model=args.gemini_teacher_model,
        gemini_api_key=args.gemini_api_key,
        gemini_rpm_limit=args.gemini_rpm,
        teacher_loss_weight=args.teacher_loss_weight,
    )
    apply_phase_preset(train_cfg)

    if args.max_steps:
        train_cfg.max_steps = args.max_steps
    if args.lr:
        train_cfg.lr = args.lr

    train_unsloth(model_cfg, train_cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
