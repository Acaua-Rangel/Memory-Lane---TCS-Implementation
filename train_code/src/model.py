"""
Model: Gemma 4 with Token Compression Sub-network (TCS) + LoRA.

The TCS compresses N input tokens into M = ceil(N/r) tokens via strided Conv1D,
reducing self-attention cost from O(N²) to O(M²). LoRA adapters teach the frozen
transformer blocks to work with compressed representations.
"""

from __future__ import annotations

import math
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig as PeftLoraConfig, get_peft_model, PeftModel

from src.config import ModelConfig, CompressionConfig, LoraConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Token Compression Sub-network
# ---------------------------------------------------------------------------

class CompressionLayer(nn.Module):
    """
    Token Compression via strided 1D Convolution.

    Projects a sequence of N tokens (B, N, D) into M = ceil(N / ratio) tokens.
    Fully differentiable, enabling end-to-end training with the frozen LLM.
    """

    def __init__(self, d_model: int, comp_cfg: CompressionConfig):
        super().__init__()
        self.ratio = comp_cfg.ratio
        kernel_size = max(comp_cfg.kernel_size, comp_cfg.ratio)
        padding = (kernel_size - 1) // 2

        self.compress = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                stride=comp_cfg.ratio,
                padding=padding,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1,
                bias=False,
            ),
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) → (B, M, D) where M ≈ N / ratio."""
        out = self.compress(x.transpose(1, 2))  # (B, D, N) → (B, D, M)
        return self.norm(out.transpose(1, 2))    # (B, M, D)


class DecompressionLayer(nn.Module):
    """
    Upsamples compressed tokens back to original sequence length.
    Used for consistency loss against full-resolution hidden states.
    """

    def __init__(self, d_model: int, comp_cfg: CompressionConfig):
        super().__init__()
        self.ratio = comp_cfg.ratio
        kernel_size = max(comp_cfg.kernel_size, comp_cfg.ratio)
        padding = (kernel_size - 1) // 2
        output_padding = comp_cfg.ratio - 1

        self.decompress = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                stride=comp_cfg.ratio,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        out = self.decompress(x.transpose(1, 2)).transpose(1, 2)
        if out.size(1) > target_len:
            out = out[:, :target_len, :]
        elif out.size(1) < target_len:
            pad = torch.zeros(
                out.size(0), target_len - out.size(1), out.size(2),
                device=out.device, dtype=out.dtype,
            )
            out = torch.cat([out, pad], dim=1)
        return self.norm(out)


# ---------------------------------------------------------------------------
# Multi-Scale Compression
# ---------------------------------------------------------------------------

class MultiScaleCompressionLayer(nn.Module):
    """
    Multiple compression/decompression pairs for simultaneous multi-ratio training.
    During inference a single ratio is selected.
    """

    def __init__(self, d_model: int, max_seq_len: int, comp_cfg: CompressionConfig):
        super().__init__()
        self.ratios = sorted(comp_cfg.ratios)
        self.d_model = d_model

        self.compressors = nn.ModuleDict()
        self.decompressors = nn.ModuleDict()
        self.pos_embs = nn.ModuleDict()

        for ratio in self.ratios:
            per_cfg = CompressionConfig(
                enabled=True,
                ratio=ratio,
                kernel_size=max(comp_cfg.kernel_size, ratio),
            )
            key = str(ratio)
            self.compressors[key] = CompressionLayer(d_model, per_cfg)
            self.decompressors[key] = DecompressionLayer(d_model, per_cfg)
            comp_len = math.ceil(max_seq_len / ratio) + 1
            self.pos_embs[key] = nn.Embedding(comp_len, d_model)

    def compress(self, x: torch.Tensor, ratio: int) -> torch.Tensor:
        key = str(ratio)
        return self.compressors[key](x)

    def add_pos(self, x_comp: torch.Tensor, ratio: int) -> torch.Tensor:
        key = str(ratio)
        M = x_comp.size(1)
        pos = torch.arange(M, device=x_comp.device).unsqueeze(0)
        return x_comp + self.pos_embs[key](pos)

    def decompress(self, h: torch.Tensor, target_len: int, ratio: int) -> torch.Tensor:
        key = str(ratio)
        return self.decompressors[key](h, target_len)


# ---------------------------------------------------------------------------
# Gemma 4 with TCS + LoRA
# ---------------------------------------------------------------------------

def _load_base_model(cfg: ModelConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the frozen Gemma base model with optional quantization."""
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(cfg.torch_dtype, torch.bfloat16)

    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )
    elif cfg.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # With quantization: device_map determines GPU placement.
    #   - DDP: each rank loads onto its own GPU via LOCAL_RANK env var
    #   - Single GPU: "auto" lets accelerate / bitsandbytes pick
    # Without quantization: load to CPU so DDP / Accelerate can place each
    # replica on the correct GPU via accelerator.prepare().
    if quantization_config:
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            device_map = {"": int(local_rank)}
        else:
            device_map = "auto"
    else:
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _apply_lora(model: AutoModelForCausalLM, lora_cfg: LoraConfig) -> PeftModel:
    """Wrap the base model with LoRA adapters."""
    peft_config = PeftLoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


class Gemma4WithTCS(nn.Module):
    """
    Gemma 4 E2B wrapped with Token Compression Sub-network and LoRA.

    The base model is frozen. Only TCS and LoRA adapters are trainable.
    Supports three forward modes:
    1. compressed: TCS → Transformer + LoRA → logits (for training/inference)
    2. uncompressed: Embedding → Transformer (no TCS) → logits (self-distillation teacher)
    3. both: Returns both for self-distillation loss computation
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Load frozen base model
        logger.info("Loading base model: %s", cfg.base_model_name)
        self.base_model, self.tokenizer = _load_base_model(cfg)

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Gemma 4 E2B is multimodal (Gemma4ForConditionalGeneration).
        # For text-only TCS training, extract the text language model
        # to avoid the vision/audio pipeline and its memory-expensive
        # per-layer input expansion.
        #
        # Structure: Gemma4ForConditionalGeneration
        #   .model (Gemma4Model)
        #     .language_model (Gemma4TextModel — base model, no LM head)
        # The language_model returns BaseModelOutputWithPast (hidden states),
        # NOT logits. We need a separate LM head to project → vocab.
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "language_model"):
            logger.info("Multimodal model detected — using text language model directly")
            self._lm = self.base_model.model.language_model
            self._lm_head = self._find_lm_head()
        else:
            self._lm = self.base_model
            self._lm_head = None  # CausalLM models return logits directly

        # Get hidden size from the loaded model
        # Gemma 4 is multimodal — hidden_size lives under text_config
        if hasattr(self.base_model.config, "hidden_size"):
            self.d_model = self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, "text_config"):
            self.d_model = self.base_model.config.text_config.hidden_size
        else:
            raise AttributeError(
                f"Cannot determine hidden_size from {type(self.base_model.config).__name__}"
            )

        # Gemma 4 TextModel has per-layer input processing
        # (get_per_layer_inputs) that allocates huge tensors when
        # called with inputs_embeds but no input_ids.  For the
        # compressed forward path we bypass TextModel.forward() and
        # iterate through decoder layers directly.
        self._rotary_emb: nn.Module | None = None
        self._decoder_layers, self._decoder_norm = self._find_decoder_components()
        self._embed_scale: float = 1.0
        if self._decoder_layers is not None:
            self._embed_scale = self.d_model ** 0.5 if self.d_model else 1.0
            logger.info(
                "Found %d decoder layers for direct forward (embed_scale=%.1f)",
                len(self._decoder_layers), self._embed_scale,
            )

        # Initialize TCS
        if cfg.compression.enabled:
            if cfg.compression.multi_scale:
                self.tcs = MultiScaleCompressionLayer(
                    self.d_model, cfg.max_seq_len, cfg.compression
                )
            else:
                self.tcs = CompressionLayer(self.d_model, cfg.compression)
                self.decompressor = DecompressionLayer(self.d_model, cfg.compression)
                comp_len = math.ceil(cfg.max_seq_len / cfg.compression.ratio) + 1
                self.compressed_pos_emb = nn.Embedding(comp_len, self.d_model)
        else:
            self.tcs = None

        # Apply LoRA (wraps base_model in-place)
        if cfg.lora.enabled:
            logger.info("Applying LoRA (rank=%d, alpha=%d)", cfg.lora.rank, cfg.lora.alpha)
            self.base_model = _apply_lora(self.base_model, cfg.lora)

        # Gradient checkpointing
        if cfg.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        self._log_param_counts()

    def _log_param_counts(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Parameters — total: %.2fM, trainable: %.2fM (%.1f%%)",
            total / 1e6,
            trainable / 1e6,
            100.0 * trainable / total if total > 0 else 0,
        )

    def _find_decoder_components(self) -> tuple[nn.ModuleList | None, nn.Module | None]:
        """Find decoder layers and norm for direct layer-by-layer forward.

        Searches common HuggingFace model structures:
        - _lm.layers + _lm.norm  (Gemma4TextModel)
        - _lm.model.layers + _lm.model.norm  (nested decoder)

        Returns (layers, norm) or (None, None) if not found.
        Also sets self._rotary_emb if found (needed for position embeddings).
        """
        if self._lm is self.base_model:
            return None, None  # Standard CausalLM — no bypass needed

        for obj in [self._lm, getattr(self._lm, "model", None)]:
            if obj is None:
                continue
            if hasattr(obj, "layers") and hasattr(obj, "norm"):
                self._rotary_emb = getattr(obj, "rotary_emb", None)
                return obj.layers, obj.norm
        return None, None

    def _find_lm_head(self) -> nn.Module | None:
        """Find the LM head that projects hidden states → vocab logits."""
        # Check common locations for the LM head
        for obj in [self._lm, self.base_model]:
            head = getattr(obj, "lm_head", None)
            if isinstance(head, nn.Module):
                logger.info("Found LM head at %s.lm_head", type(obj).__name__)
                return head
        # Tied embeddings: logits = hidden @ embed_weight.T
        logger.info("No explicit lm_head found — will use tied embedding weights")
        return None

    def _hidden_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        if self._lm_head is not None:
            return self._lm_head(hidden_states)
        # Tied embeddings: logits = hidden @ embed_weight.T
        embed_weight = self._lm.get_input_embeddings().weight
        return nn.functional.linear(hidden_states, embed_weight)

    def _compute_logits(self, outputs: Any) -> torch.Tensor:
        """Extract logits from model outputs, applying LM head if needed."""
        if hasattr(outputs, "logits"):
            return outputs.logits
        return self._hidden_to_logits(outputs.last_hidden_state)

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from the text language model's embedding layer."""
        return self._lm.get_input_embeddings()(input_ids)

    def _forward_decoder_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward through decoder layers directly.

        Bypasses Gemma4TextModel.forward() and its per-layer input
        processing (get_per_layer_inputs) which allocates huge tensors
        when receiving inputs_embeds without input_ids.
        """
        B, M = hidden_states.shape[:2]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Apply Gemma-style embedding scaling (sqrt(d_model))
        hidden_states = hidden_states * self._embed_scale

        # Position IDs for RoPE
        position_ids = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)

        # Pre-compute rotary embeddings (cos, sin) per layer type.
        # Gemma 4's rotary embedding has different inv_freq buffers for
        # different layer types (e.g., "global", "sliding").  The rotary
        # module's forward() requires layer_type= to select the right
        # frequencies.  We cache by layer_type to avoid redundant work.
        pos_emb_cache: dict[str | None, tuple[torch.Tensor, torch.Tensor]] = {}

        # 4D causal mask: (1, 1, M, M) — broadcastable to (B, heads, M, M)
        # Additive convention: 0 for attend, large negative for mask
        causal_mask = torch.zeros(1, 1, M, M, device=device, dtype=dtype)
        mask_cond = torch.triu(
            torch.ones(M, M, device=device, dtype=torch.bool), diagonal=1,
        )
        causal_mask = causal_mask.masked_fill(mask_cond, torch.finfo(dtype).min)

        # Gemma 4 decoder layers expect per_layer_input — a multiplicative
        # gate from get_per_layer_inputs().  We bypass that (it OOMs on
        # inputs_embeds), so pass a scalar one to make the multiply an
        # identity: hidden_states * 1.0 = hidden_states.  The TCS learns
        # to compensate for the absent per-layer gating.
        per_layer_input = torch.ones(1, device=device, dtype=dtype)

        # Shared KV state dict — layers with KV sharing store their
        # key/value tensors here so later shared layers can reuse them.
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for layer in self._decoder_layers:
            # Determine each layer's attention type for correct rotary freqs
            layer_type = getattr(layer, "layer_type", None)
            if layer_type is None:
                attn = getattr(layer, "self_attn", None)
                layer_type = getattr(attn, "layer_type", None) if attn else None

            # Compute or reuse cached position embeddings for this type
            position_embeddings = None
            if self._rotary_emb is not None:
                if layer_type not in pos_emb_cache:
                    pos_emb_cache[layer_type] = self._rotary_emb(
                        hidden_states, position_ids, layer_type=layer_type,
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
            # Gemma4TextDecoderLayer returns a single tensor, not a tuple.
            # Guard against both cases to be safe.
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_states = self._decoder_norm(hidden_states)
        logits = self._hidden_to_logits(hidden_states)
        return {"logits": logits, "loss": None}

    def _forward_from_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run transformer blocks on given embeddings."""
        # Gemma 4 per-layer input processing OOMs on inputs_embeds
        # without input_ids — use direct decoder layers instead.
        if self._decoder_layers is not None:
            return self._forward_decoder_layers(embeds, attention_mask)

        outputs = self._lm(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = self._compute_logits(outputs)
        return {"logits": logits, "loss": getattr(outputs, "loss", None)}

    def forward_compressed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        compression_ratio: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with token compression."""
        embeds = self._get_embeddings(input_ids)
        ratio = compression_ratio or self.cfg.compression.ratio

        if isinstance(self.tcs, MultiScaleCompressionLayer):
            compressed = self.tcs.compress(embeds, ratio)
            compressed = self.tcs.add_pos(compressed, ratio)
        else:
            compressed = self.tcs(embeds)
            M = compressed.size(1)
            pos_ids = torch.arange(M, device=compressed.device).unsqueeze(0)
            compressed = compressed + self.compressed_pos_emb(pos_ids)

        # Build compressed attention mask
        comp_mask = None
        if attention_mask is not None:
            B = attention_mask.size(0)
            M = compressed.size(1)
            comp_mask = torch.ones(B, M, device=attention_mask.device, dtype=attention_mask.dtype)

        # Compressed labels: we cannot directly map compressed positions to original labels.
        # For self-distillation, labels are applied via KL loss, not CE on compressed output.
        result = self._forward_from_embeds(compressed, attention_mask=comp_mask, labels=None)
        result["compressed_embeds"] = compressed
        return result

    def forward_uncompressed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Standard forward pass without compression (used as self-distillation teacher)."""
        # Disable LoRA for the teacher path to get true baseline behavior
        if isinstance(self.base_model, PeftModel):
            with self.base_model.disable_adapter_layers():
                outputs = self._lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
        else:
            outputs = self._lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        logits = self._compute_logits(outputs)
        return {"logits": logits, "loss": getattr(outputs, "loss", None)}

    def _align_teacher_logits_to_student(
        self,
        teacher_logits: torch.Tensor,
        student_len: int,
    ) -> torch.Tensor:
        """Align teacher sequence length to student length via index sampling."""
        teacher_len = teacher_logits.size(1)
        if teacher_len == student_len:
            return teacher_logits

        indices = torch.linspace(
            0,
            teacher_len - 1,
            student_len,
            device=teacher_logits.device,
        ).long()
        return teacher_logits.index_select(1, indices)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mode: str = "compressed",
        compression_ratio: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Unified forward pass.

        Args:
            mode: "compressed", "uncompressed", or "both" (for self-distillation).
            compression_ratio: Override compression ratio (for multi-scale).
        """
        if mode == "uncompressed" or self.tcs is None:
            return self.forward_uncompressed(input_ids, attention_mask, labels)

        if mode == "compressed":
            return self.forward_compressed(
                input_ids, attention_mask, labels, compression_ratio
            )

        # mode == "both": self-distillation
        # Student (compressed, with grad)
        student_out = self.forward_compressed(
            input_ids, attention_mask, labels, compression_ratio
        )
        student_logits = student_out["logits"]

        # Teacher (uncompressed, no grad), aligned early to reduce return size.
        with torch.no_grad():
            teacher_out = self.forward_uncompressed(input_ids, attention_mask, labels)
            teacher_logits = self._align_teacher_logits_to_student(
                teacher_out["logits"],
                student_logits.size(1),
            )

        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits.detach(),
            "compressed_embeds": student_out.get("compressed_embeds"),
        }

    def save_trainable(self, save_dir: str | Path) -> None:
        """Save only trainable components (TCS + LoRA adapters)."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save TCS weights
        if self.tcs is not None:
            tcs_state = {k: v for k, v in self.state_dict().items() if "tcs" in k or "compressed_pos_emb" in k or "decompressor" in k}
            torch.save(tcs_state, save_dir / "tcs_weights.pt")

        # Save LoRA adapters (PEFT format)
        if isinstance(self.base_model, PeftModel):
            self.base_model.save_pretrained(save_dir / "lora_adapters")

        logger.info("Saved trainable weights to %s", save_dir)

    def load_trainable(self, load_dir: str | Path) -> None:
        """Load trainable components (TCS + LoRA adapters)."""
        load_dir = Path(load_dir)

        # Load TCS weights
        tcs_path = load_dir / "tcs_weights.pt"
        if tcs_path.exists() and self.tcs is not None:
            tcs_state = torch.load(tcs_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.load_state_dict(tcs_state, strict=False)
            logger.info(
                "Loaded TCS weights from %s (missing=%d, unexpected=%d)",
                tcs_path, len(missing), len(unexpected),
            )

        # Load LoRA adapters
        lora_path = load_dir / "lora_adapters"
        if lora_path.exists() and isinstance(self.base_model, PeftModel):
            self.base_model = PeftModel.from_pretrained(
                self.base_model.get_base_model(),
                str(lora_path),
            )
            logger.info("Loaded LoRA adapters from %s", lora_path)

    def get_trainable_params(self, phase: str) -> list[nn.Parameter]:
        """Get parameters that should be optimized for a given training phase."""
        params = []

        if phase == "tcs-pretrain":
            # Only TCS parameters
            if self.tcs is not None:
                params.extend(self.tcs.parameters())
                if hasattr(self, "compressed_pos_emb"):
                    params.extend(self.compressed_pos_emb.parameters())
                if hasattr(self, "decompressor"):
                    params.extend(self.decompressor.parameters())
        elif phase in ("tcs-lora", "tool-calling"):
            # TCS + LoRA parameters
            if self.tcs is not None:
                params.extend(self.tcs.parameters())
                if hasattr(self, "compressed_pos_emb"):
                    params.extend(self.compressed_pos_emb.parameters())
                if hasattr(self, "decompressor"):
                    params.extend(self.decompressor.parameters())
            # LoRA params (already marked requires_grad by peft)
            if isinstance(self.base_model, PeftModel):
                for p in self.base_model.parameters():
                    if p.requires_grad:
                        params.append(p)
        else:
            # All trainable params
            params = [p for p in self.parameters() if p.requires_grad]

        return params
