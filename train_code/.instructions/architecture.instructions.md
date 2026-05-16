---
applyTo: 'src/model.py,src/losses.py,src/config.py,src/tools/**'
---

# Model Architecture Skill

## TCS (Token Compression Sub-network)
- Inserted between Gemma 4's embedding layer output and first transformer block.
- Uses `nn.Conv1d(d_model, d_model, kernel_size=7, stride=ratio, padding=3)` + GELU + pointwise Conv1d + RMSNorm.
- Must handle variable input lengths — output length is `ceil(N / ratio)`.
- Compression layer is always trainable. Never freeze it.

## LoRA Integration
- Use HuggingFace `peft` library. Do NOT implement LoRA from scratch.
- Apply to attention projections only: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Default rank=16, alpha=32, dropout=0.05.
- The base model must be loaded with `torch.float16` or `torch.bfloat16` and `device_map="auto"`.

## Face Pipeline (Mobile-Optimized, Separate from LLM)
The face recognition pipeline is **completely independent** from the LLM:
- **Detection:** MediaPipe Face Detection (<1MB, works on ARM CPUs, <10ms).
- **Embeddings:** MobileFaceNet (1MB ONNX model, 128-d vectors, <30ms on ARM).
- **Search:** sqlite-vec cosine distance search, threshold 0.85.
- **NOT** CLIP, SigLIP, ArcFace-R100 — these are too large for mobile.
- The face pipeline outputs a `face_id` which the LLM uses via tool calling.
- All face models must export to ONNX for mobile deployment (ONNX Runtime Mobile).

## Forward Pass Pattern
```python
def forward(self, input_ids, attention_mask=None, labels=None):
    # 1. Get embeddings from frozen Gemma 4
    embeds = self.base_model.get_input_embeddings()(input_ids)
    
    # 2. Compress via TCS
    compressed = self.tcs(embeds)  # (B, N, D) → (B, M, D)
    
    # 3. Add compressed positional embeddings
    compressed = compressed + self.compressed_pos_emb[:, :compressed.size(1)]
    
    # 4. Pass through transformer blocks (with LoRA)
    # Must inject compressed embeddings into the model's forward
```

## Self-Distillation Pattern
- Run forward WITHOUT TCS (full tokens) → get teacher logits (detached, no grad)
- Run forward WITH TCS (compressed) → get student logits
- Loss = α · T² · KL(student ‖ teacher) + (1-α) · CE(student, labels)

## Positional Embeddings
- Compressed sequences need their own positional embeddings.
- Size: `(1, max_compressed_len, d_model)` where `max_compressed_len = ceil(max_seq_len / ratio)`.
- Initialize from truncated/interpolated original positional embeddings when possible.

## Memory Optimization
- Use `torch.utils.checkpoint.checkpoint()` on transformer blocks.
- Load base model with `load_in_8bit=True` via bitsandbytes when VRAM is tight.
- Use `torch.amp.autocast('cuda', dtype=torch.bfloat16)` for forward/backward.
