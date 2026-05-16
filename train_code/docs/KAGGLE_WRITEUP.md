# Memory Lane: When AI Remembers So They Don't Have To

*An offline Alzheimer's companion powered by Gemma 4 E2B with Token Compression, LoRA, and intelligent task routing — fine-tuned with Unsloth, deployed via LiteRT*

---

## Motivation

Over 55 million people worldwide live with dementia [[1]](#ref-1), with nearly 10 million new cases every year [[2]](#ref-2). Every 3 seconds, someone new is diagnosed [[1]](#ref-1). They gradually lose the ability to recognize their own children, remember where they are, follow a medication schedule, or recall the shared experiences that define their most important relationships. Caregivers — mostly family members, who provide on average 5 hours of care per day [[2]](#ref-2) — spend exhausting hours repeating the same information, managing complex pill schedules, and carrying the emotional weight of being forgotten by someone they love.

Current AI assistants fail these patients for three fundamental reasons. First, they require constant internet connectivity — unreliable for elderly users, especially in low- and middle-income countries where over 60% of dementia patients reside [[2]](#ref-2). Second, they are passive: the patient has to remember to open an app and ask for help, which is precisely what Alzheimer's prevents them from doing. Third, sending sensitive medical and facial data to the cloud raises serious privacy concerns under varying data protection jurisdictions.

There's a story from a Brazilian influencer named Renato Cariani about his father who had Alzheimer's. His father forgot how to smoke — his body was in withdrawal, craving nicotine, but his mind couldn't remember what smoking was or how to do it. That image of someone trapped between a body that needs something and a mind that can't remember what — that's who Lane is for.

We didn't want to build another note-taking app that sits idle on a phone. We wanted to build something that **actively participates** in the patient's daily life — recognizing faces, logging encounters automatically, reminding about medication unprompted, and speaking in a warm voice when the patient is confused and scared. All running on-device, all offline, all private.

---

## Solution Approach

### Lane: The Memory Agent

**Lane** (from *Memory Lane*) is an AI agent that runs entirely on the patient's phone. It combines Gemma 4 E2B with a novel Token Compression Sub-network (TCS) for efficient on-device inference, LoRA [[3]](#ref-3) adaptation for compressed token understanding, face recognition via MobileFaceNet [[6]](#ref-6), and a local SQLite database that serves as the patient's persistent external memory.

The interaction model is voice-first. The patient says "Oi Lane" (or presses a physical button), and Lane listens, processes, and responds through Google's offline TTS — no screen reading required. When a face appears on camera, Lane proactively identifies the person and shares context without being asked.

The key architectural decisions were driven by one constraint: **everything must run offline on a phone**.

### Token Compression Sub-network (TCS) — Original Research

The core technical contribution is the Token Compression Sub-network, a novel technique from our own research currently being prepared for publication (preprint on arXiv with GPT-2 validation results).

The TCS is a trainable Conv1D layer inserted between Gemma 4's embedding layer and its first transformer block. It compresses a sequence of N tokens into N/r tokens before self-attention:

```
Input Embeddings (B, 512, 2048)
         │
         ▼
Conv1d(2048, 2048, kernel=7, stride=4, padding=3) → GELU → Conv1d(1×1) → RMSNorm
         │
         ▼
Compressed (B, 128, 2048)   ← 4× fewer tokens
         │
         ▼
Gemma 4 Transformer Blocks  (now processing 128 tokens instead of 512)
```

With 4× token compression, self-attention FLOPs drop by 16× (O(N²) → O(N/4)²). The TCS adds only ~3M parameters (0.15% of Gemma 4 E2B), is fully differentiable, and trains end-to-end via self-distillation [[4]](#ref-4) where the uncompressed path through Gemma 4 serves as the teacher and the compressed path as the student — no external teacher model needed.

The research trajectory: validated on TinyStories + GPT-2 first, now adapted for Gemma 4 E2B with LoRA for this hackathon, with a full paper planned with Gemma 4 benchmarks.

### Self-Distillation Training

Training happens in three phases, all on Kaggle (2× T4, 30GB RAM, 19.5GB disk):

**Phase 1 — TCS Pre-training** (~30 min, 2000 steps): Only the TCS parameters are trainable (~3M). The loss is pure KL divergence between the compressed student logits and the uncompressed teacher logits (α=1.0). Learning rate starts at 1e-3 because TCS is randomly initialized.

**Phase 2 — TCS + LoRA Joint Training** (~2-4h, 10,000 steps): Now both TCS and LoRA [[3]](#ref-3) adapters are trainable (~23M params). LoRA is applied to q_proj, k_proj, v_proj, o_proj with rank=16 and alpha=32. The loss becomes balanced: 50% KL distillation + 50% cross-entropy (α=0.5). A critical implementation detail: since compressed sequences have different lengths than the teacher, we align teacher logits to compressed positions by deterministic index sampling and then compute KL and log-normalizers in vocabulary chunks. When we want Unsloth's 4-bit path, this phase can run through `FastLanguageModel` on a single T4; however, when we want to occupy both Kaggle GPUs, we switch to the Accelerate DDP entrypoint instead. Learning rate drops to 2e-4 for LoRA stability.

**Phase 3 — Tool Calling Fine-tune** (~1-2h, 3000 steps): The model is fine-tuned on synthetic tool-calling conversations generated by Gemma 4 E2B itself. Standard cross-entropy is applied only on assistant tokens, while system, user, and tool turns remain masked with -100. Furthermore, because Phase 3 still runs through the compressed TCS path, those unmasked targets are deterministically aligned to the compressed positions before cross-entropy, preventing sequence-length mismatches when the compression ratio is greater than 1. When we prioritize 4-bit efficiency, this phase runs through the single-GPU Unsloth entrypoint; in contrast, when we need both Kaggle T4s active, we launch the DDP entrypoint. Learning rate drops further to 3e-5.

### Synthetic Data Generation with Gemma 4

We built a `GemmaDataGenerator` that uses Gemma 4 E2B (the instruction-tuned model) to generate diverse training conversations. The pipeline:

1. **Prompt Gemma 4** to generate 10-30 diverse user queries per category (e.g., "Generate 20 different ways a confused elderly person might ask about someone who just walked in")
2. **Parse the output** into individual queries, falling back to handcrafted templates if the model output is insufficient
3. **Deterministically construct** the tool call JSON and tool response (no hallucination risk — these are built from structured data)
4. **Batch-generate** warm assistant responses using Gemma 4 with temperature=0.8 and top_p=0.92

The result is ~25K training examples across 5 weighted categories:
- Person Recognition: 40% (face detection → identify person)
- Orientation: 20% ("Where am I?" / scene description)
- Medication: 15% (schedule queries)
- Memory Recall: 15% ("Tell me about Maria")
- Caregiver Alerts: 10% (confusion/wandering detection)

Each example is a 5-turn conversation in Gemma 4's chat format:
```
<start_of_turn>system [system prompt with 9 tool definitions]
<start_of_turn>user [patient's question]
<start_of_turn>model [tool call JSON]
<start_of_turn>tool [tool response JSON]
<start_of_turn>model [warm, simple response to the patient]
```

The synthetic dataset includes 24 persons (family members, caregivers, neighbors, friends — each with relationship, bio, and face_id), 8 medications across 4 time periods (with physical descriptions like "small white round pill"), and 10 locations (with comma-separated feature keywords for visual matching).

### Intelligent Task Routing (Cactus)

Not every query needs the full LLM. Our `TaskRouter` classifies incoming inputs using lightweight regex patterns (zero added latency) and routes to the optimal processing path:

| Input | Route | Compression | Latency Budget |
|---|---|---|---|
| Face detected + embedding | MobileFaceNet → sqlite-vec → LLM | r=4 | 500ms |
| "Bom dia" (greeting) | Gemma 4 + TCS | r=8 (fastest) | 500ms |
| "Que remédio?" (medication) | SQLite direct lookup — **skip LLM** | — | 300ms |
| "Quem é a Maria?" (memory) | Gemma 4 + TCS | r=2 (quality) | 2000ms |
| "Estou perdido" (alert) | Rule-based instant response + alert | — | 100ms |

This means Lane uses **five different processing paths** depending on the task: face pipeline, three LLM compression levels (r=2, r=4, r=8), SQLite direct queries, and rule-based logic. The router returns a `RoutingDecision` with task type, route, compression ratio, confidence score, and latency budget — the `RoutedCompanion` orchestrator then executes the appropriate handler.

The routing patterns are bilingual (Portuguese + English) since our target users are Brazilian elderly patients who may switch between languages mid-sentence.

### 9 SQLite Tools via Native Function Calling

Lane uses Gemma 4's native function calling format to interact with a local SQLite database (with sqlite-vec extension for vector similarity search). The 9 tools:

| Tool | Purpose | Type |
|---|---|---|
| `read_person(face_id)` | Look up person by face match, return bio + memories | Read |
| `write_encounter(person, context)` | Log who visited, what happened, when | Write |
| `get_medication(time_of_day)` | Get pill schedule for morning/afternoon/evening/night | Read |
| `describe_location(features)` | Identify room from visual feature keywords | Read |
| `alert_caregiver(type, details)` | Send confusion/wandering/fall alert | Write |
| `get_agenda(day_of_week)` | Get appointments and scheduled events | Read |
| `save_preference(category, key, value)` | Store patient preferences and habits | Write |
| `get_preferences(category)` | Retrieve preferences (food, music, hobbies, etc.) | Read |
| `get_routine(time_of_day)` | Get daily routine steps | Read |

Critically, Lane doesn't just read from the database — it **actively writes**. Every encounter, every conversation fragment, every medication confirmation is recorded. The database becomes a living, growing memory for the patient.

### Face Recognition Pipeline

Face recognition runs as a separate lightweight module, completely independent from the LLM:

1. **MediaPipe Face Detection** via BlazeFace [[5]](#ref-5) (<1MB, sub-millisecond on mobile GPUs) — detects faces in camera frames
2. **MobileFaceNet** [[6]](#ref-6) (4.0MB float32 / ~1MB INT8-quantized ONNX, 128-dimensional embeddings, 18ms on mobile [[6]](#ref-6)) — generates face embeddings
3. **sqlite-vec** [[8]](#ref-8) cosine search — matches against known faces with threshold 0.85

We chose MobileFaceNet [[6]](#ref-6) over CLIP [[7]](#ref-7) (~400MB) or ArcFace-R100 [[9]](#ref-9) (~250MB) because it's under 4MB (or ~1MB quantized), runs in 18ms on mobile CPUs [[6]](#ref-6), and produces compact 128-d vectors (4× less storage than 512-d alternatives). For a database of ~50 known faces, MobileFaceNet's 99.55% accuracy on LFW [[6]](#ref-6) is more than sufficient.

### Mobile App

The app is built with React Native (Expo) by our team member Juan Suzart. The design is intentionally minimal: a camera view and a record button. The interaction is voice-first — the patient shouldn't need to read or tap anything. We use Google's offline TTS and STT models to stay within the Google ecosystem. Activation is via wake word ("Oi Lane") with a physical button as fallback for accessibility.

---

## Development Process

### Team

**Acauã Rangel Brazil** — ML engineer and researcher. Responsible for the TCS research (original contribution), model training pipeline, agent orchestration, Kaggle training infrastructure, tool calling system, and the routing layer.

**Juan Benjamin Suzart** — Mobile developer. Responsible for the React Native (Expo) app, UI/UX design in Figma, TTS/STT integration, and camera/face pipeline on device.

We're based in Salvador, Bahia, Brazil.

### Challenges We Faced

#### Preserving base knowledge through compression

The biggest challenge was not losing Gemma 4's base knowledge when compressing tokens 4×. When you take 512 tokens and squeeze them into 128, the attention patterns change fundamentally — key-value distributions shift, positional relationships break, and the transformer has never seen these "super-tokens" before.

Our first attempts with TCS alone (no LoRA) showed the model could compress and decompress tokens but lost coherent generation. The KL divergence between compressed and uncompressed outputs remained high even after convergence.

The solution was the two-phase approach: first train TCS in isolation to learn good compression (Phase 1), then add LoRA [[3]](#ref-3) adapters to teach the transformer blocks how to attend over compressed representations (Phase 2). LoRA adapts q_proj, k_proj, v_proj, and o_proj — precisely the layers that need to learn new attention patterns for compressed tokens. This brought the self-distillation loss to acceptable levels while preserving Gemma 4's fluency and world knowledge.

A critical implementation detail: the compressed and uncompressed sequences have different lengths, so you can't directly compute KL divergence between them. We solve this by aligning teacher logits to compressed token positions with deterministic index sampling, followed by chunked KL and chunked log-normalizer computation over the vocabulary to control VRAM spikes. This was a non-obvious solution that took several failed attempts before landing.

An additional systems detail is applied in distributed training: for `mode="both"`, the model aligns teacher logits to student length before returning outputs, so Accelerate/DDP output conversion does not inflate full-length teacher logits into a large fp32 allocation.

#### Synthetic data quality for tool calling

Training tool calling requires structured conversations where the model learns when to call tools, what arguments to pass, and how to format warm responses after receiving tool results. The quality of this synthetic data directly determines whether Lane feels like a caring companion or a database query terminal.

Our initial approach of handcrafting examples hit a ceiling around 200 conversations — not enough variety for robust training. We then built a `GemmaDataGenerator` that uses Gemma 4 E2B itself to generate diverse user queries, while keeping tool calls and responses deterministic (built from structured data, never hallucinated). This gave us 25K+ examples with natural language diversity while maintaining structural correctness.

The key insight was separating what the model generates (user queries and final responses — where variety matters) from what we construct programmatically (tool call JSON and tool results — where correctness matters).

#### Kaggle disk constraints

Kaggle gives you 19.5GB of disk. A single Gemma 4 E2B checkpoint is ~4GB. Our training pipeline produces intermediate checkpoints every 500-1000 steps. Without careful management, disk fills up mid-training and the run crashes.

We implemented an aggressive checkpoint rotation system: keep only the last 3 checkpoints during training, and after each phase finishes, delete all intermediate checkpoints (the `checkpoints/` and `best/` directories), keeping only the `final/` directory with the merged TCS + LoRA weights (~50MB). This brought peak disk usage to ~7GB, leaving 12GB headroom.

The training script logs `du -sh` and `df -h` between every phase as a safety measure.

To reduce repeated startup time, the text preprocessing stage (raw text filtering, tokenization, and fixed-length chunk creation for Phases 1-2) is cached on disk in `data/text_cache/`. When dataset/tokenizer/sequence settings match, subsequent runs reuse cached chunks instead of rebuilding them. Additionally, if that primary cache path is unavailable because `data/` is not a directory in the runtime environment, the pipeline now falls back to `GEMMA_TEXT_CACHE_DIR` (or the system temp directory) instead of crashing.

Moreover, the data loader now receives sequence length from the model configuration (`model_cfg.max_seq_len`) instead of the micro-batch setting, which prevents pathological tiny-chunk generation (e.g., chunk length equal to batch size) during Phase 2 startup.

For distributed failures, the DDP entrypoint is wrapped with Torch Elastic `@record` and rank-local diagnostics, so root-cause tracebacks and CUDA memory context are emitted directly instead of only a generic `ChildFailedError` wrapper.

To stabilize multi-GPU reduction behavior, the Accelerate/DDP setup explicitly enables `find_unused_parameters=True`, which is required because some optional TCS submodules can exist in the wrapped model while remaining outside the active loss path in a given training phase.

Furthermore, we added a bounded early-step OOM recovery path in `train_ddp`: when CUDA OOM occurs at startup, the script retries with a reduced micro-batch and a compensating gradient-accumulation factor, while preserving the intended effective batch size.

#### Making Unsloth work with TCS

Unsloth optimizes LoRA training with fused CUDA kernels, but it has its own model loading and PEFT integration. Our TCS layer sits outside Unsloth's optimization scope — it's a custom module injected between the embedding layer and the first transformer block.

We solved this by creating an `UnslothWithTCS` wrapper that delegates LoRA optimization to Unsloth while managing TCS independently. TCS parameters are small enough (~3M) that they don't benefit from Unsloth's optimizations anyway. The wrapper exposes the same `forward()` interface (compressed/uncompressed/both modes) and `save_trainable()` / `get_trainable_params()` methods as the standard model, so the training loop code is identical.

Moreover, for Gemma 4 multimodal configs, the wrapper resolves model width from `hidden_size` or `text_config.hidden_size`, which avoids startup failures when the top-level config omits a direct `hidden_size` field.

In addition, to simplify phase handoff, the wrapper now loads TCS checkpoints from either nested dictionaries (`tcs`, `pos_emb`, `decompressor`) or flat trainable state dictionaries (`tcs.*`, `compressed_pos_emb.*`, `decompressor.*`).

Furthermore, when Unsloth exposes a multimodal processor (instead of a plain tokenizer), the pipeline explicitly routes tokenization through the processor's nested text tokenizer, preventing `encode()` mismatches in text dataset preparation.

Additionally, the Unsloth training loop now resolves AMP settings against the active CUDA capability, automatically downgrading `bf16` requests to `fp16` on devices like T4 that do not support bfloat16.

Moreover, the Gemini teacher cache path now treats any API-stage runtime failure as recoverable and immediately switches to local teacher generation, preserving training continuity instead of aborting the run.

Furthermore, for the Unsloth compressed path, we now bypass Gemma 4's multimodal per-layer input expansion when `inputs_embeds` are provided by executing decoder layers directly on compressed text embeddings, which avoids extreme transient allocations that can exceed T4 VRAM by orders of magnitude.

In addition, this mitigation now resolves text backbones through recursively nested wrapper objects (Unsloth + PEFT), making the bypass resilient to wrapper layout changes across dependency versions.

Additionally, to remain compatible with recent Unsloth versions that disable raw logits by default, distillation phases now set `UNSLOTH_RETURN_LOGITS=1` automatically before Unsloth model loading.

At the moment, this Unsloth entrypoint remains single-GPU. Consequently, multi-GPU Kaggle runs still go through `src.train_ddp.py`, while `src.train_unsloth.py` is reserved for 4-bit single-GPU LoRA passes.

Phase 1 (TCS-only, no LoRA) doesn't use Unsloth at all — there's no benefit. Unsloth kicks in for Phases 2 and 3, where the 2× speedup on LoRA backward passes cuts training time roughly in half.

---

## Technical Summary

| Component | Technology | Size/Impact |
|---|---|---|
| Base LLM | Gemma 4 E2B (frozen, 2B params) | Multilingual, native tool calling |
| Token Compression | TCS — Conv1D stride=4 (original research) | 4× token reduction, 16× FLOP savings, ~3M params |
| Adaptation | LoRA [[3]](#ref-3) rank=16, alpha=32 | ~20M trainable (1.1% of model) |
| Self-Distillation | KL + CE loss [[4]](#ref-4), no external teacher | Uncompressed path = teacher |
| Fast Training | Unsloth FastLanguageModel | 2× speed, 60% less VRAM |
| Task Routing | TaskRouter — 5 processing paths | Regex classification, zero overhead |
| Mobile Export | LiteRT via ai-edge-torch | int8 quantized for on-device |
| Face Detection | MediaPipe / BlazeFace [[5]](#ref-5) | <1MB, sub-ms on mobile GPU |
| Face Embedding | MobileFaceNet [[6]](#ref-6) ONNX | 4MB (1MB INT8), 128-d, 18ms |
| Vector Search | sqlite-vec [[8]](#ref-8) cosine distance | Threshold 0.85, <1ms |
| Memory Database | SQLite + WAL mode, 9 tools | Persistent offline storage |
| Mobile App | React Native (Expo) | Cross-platform, voice-first |
| Speech | Google Offline TTS/STT | Google ecosystem, works offline |

Furthermore, the LiteRT export path reconstructs the TCS embedding width directly from saved checkpoints and accepts both nested and flat checkpoint layouts. Consequently, models trained through either the standard wrapper or the Unsloth wrapper can be exported without hard-coding a single hidden size.

### Training Pipeline

| Phase | What | Steps | Loss | LR | Trainable | Backend |
|---|---|---|---|---|---|---|
| 1 | TCS Pre-train | 2,000 | KL (α=1.0) | 1e-3 | ~3M (TCS) | Accelerate |
| 2 | TCS + LoRA | 10,000 | KL+CE (α=0.5) | 2e-4 | ~23M | Unsloth |
| 3 | Tool Calling | 3,000 | CE | 3e-5 | ~23M | Unsloth |
| 4 | Export | — | — | — | — | ai-edge-torch |

### Database Schema (8 tables)

```
persons        → id, name, relationship, bio
face_embeddings → id, person_id, embedding (128-d BLOB)
memories       → id, person_id, content, memory_type
medication     → id, medication_name, description, time_of_day, dosage
encounters     → id, person_id, context, location, timestamp
locations      → id, name, description, features
agenda         → id, title, description, day_of_week, time, recurring
preferences    → id, category, key, value (UNIQUE category+key)
routines       → id, name, time_of_day, steps, notes
```

---

## What's Next

| Timeframe | Goal |
|---|---|
| Post-hackathon | Academic publication on TCS technique (arXiv → peer review) |
| 3 months | Beta app on Play Store, partnership with geriatric clinics in Salvador |
| 6 months | Integration with smart glasses / wearable cameras |
| 1 year | Clinical pilot study with real Alzheimer's patients |
| Long-term | Adapt TCS for other edge AI use cases (sign language, elder care) |

---

*Team Memory Lane — Acauã Rangel Brazil & Juan Benjamin Suzart — Salvador, Bahia, Brazil*

---

## References

<a id="ref-1"></a>**[1]** Alzheimer's Disease International. "Dementia Facts & Figures." ADI, 2024. Available at: https://www.alzint.org/about/dementia-facts-figures/

<a id="ref-2"></a>**[2]** World Health Organization. "Dementia — Key Facts." WHO Fact Sheet, 31 March 2025. Available at: https://www.who.int/news-room/fact-sheets/detail/dementia

<a id="ref-3"></a>**[3]** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv preprint arXiv:2106.09685*. https://arxiv.org/abs/2106.09685

<a id="ref-4"></a>**[4]** Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." *NIPS 2014 Deep Learning Workshop. arXiv preprint arXiv:1503.02531*. https://arxiv.org/abs/1503.02531

<a id="ref-5"></a>**[5]** Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs." *CVPR Workshop on Computer Vision for Augmented and Virtual Reality. arXiv preprint arXiv:1907.05047*. https://arxiv.org/abs/1907.05047

<a id="ref-6"></a>**[6]** Chen, S., Liu, Y., Gao, X., & Han, Z. (2018). "MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices." *CCBR 2018. arXiv preprint arXiv:1804.07573*. https://arxiv.org/abs/1804.07573

<a id="ref-7"></a>**[7]** Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021. arXiv preprint arXiv:2103.00020*. https://arxiv.org/abs/2103.00020

<a id="ref-8"></a>**[8]** Willison, A. "sqlite-vec: A SQLite Extension for Vector Search." GitHub. Available at: https://github.com/asg017/sqlite-vec

<a id="ref-9"></a>**[9]** Deng, J., Guo, J., Yang, J., Xue, N., Kotsia, I., & Zafeiriou, S. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." *IEEE TPAMI, 2022. arXiv preprint arXiv:1801.07698*. https://arxiv.org/abs/1801.07698
