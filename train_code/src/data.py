"""
Dataset loading and synthetic data generation for the Memory Companion.

Phase 1-2: General text from wikitext / OpenWebText for self-distillation.
Phase 3: Synthetic tool calling conversations for function calling fine-tune.

Data generation uses Gemma 4 to produce diverse, natural conversations
instead of relying on external APIs (GPT/Claude).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM

from src.config import TrainingConfig, ToolCallingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# General Text Dataset (Phase 1 & 2)
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Tokenized text dataset for self-distillation training.
    Loads from HuggingFace datasets and creates fixed-length chunks.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        split: str = "train",
        max_seq_len: int = 512,
        max_samples: int | None = None,
        cache_dir: str = "data/text_cache",
        use_cache: bool = True,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.chunks: list[torch.Tensor] = []

        cache_path = self._build_cache_path(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
        )

        if use_cache:
            cache_path = self._resolve_cache_path(cache_path)
            if cache_path is None:
                use_cache = False

        if use_cache:
            loaded = self._load_from_cache(cache_path)
            if loaded:
                return

        logger.info("Loading dataset %s/%s split=%s", dataset_name, dataset_config, split)
        raw = load_dataset(dataset_name, dataset_config, split=split)

        # Tokenize all text into one long sequence, then chunk
        all_ids: list[int] = []
        for i, example in enumerate(raw):
            text = example.get("text", "")
            if not text or len(text.strip()) < 10:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            if max_samples and i >= max_samples:
                break

        # Create fixed-length chunks
        for start in range(0, len(all_ids) - max_seq_len, max_seq_len):
            chunk = all_ids[start : start + max_seq_len]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        logger.info(
            "Created %d chunks of length %d from %d tokens",
            len(self.chunks), max_seq_len, len(all_ids),
        )

        if use_cache:
            self._save_to_cache(cache_path)

    def _resolve_cache_path(self, cache_path: Path) -> Path | None:
        if self._ensure_cache_dir(cache_path.parent):
            return cache_path

        configured_fallback = os.getenv("GEMMA_TEXT_CACHE_DIR", "").strip()
        fallback_root = Path(configured_fallback) if configured_fallback else Path(tempfile.gettempdir()) / "gemma4_text_cache"
        fallback_path = fallback_root / cache_path.name

        if not self._ensure_cache_dir(fallback_path.parent):
            logger.warning("Text cache disabled; no writable cache directory is available")
            return None

        logger.warning(
            "Primary text cache directory unavailable (%s). Falling back to %s",
            cache_path.parent,
            fallback_path.parent,
        )
        return fallback_path

    def _ensure_cache_dir(self, cache_dir: Path) -> bool:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create text cache directory %s (%s)", cache_dir, exc)
            return False

        if cache_dir.is_dir():
            return True

        logger.warning("Text cache path exists but is not a directory: %s", cache_dir)
        return False

    def _build_cache_path(
        self,
        cache_dir: str,
        dataset_name: str,
        dataset_config: str,
        split: str,
        max_seq_len: int,
        max_samples: int | None,
    ) -> Path:
        tokenizer_name = str(getattr(self.tokenizer, "name_or_path", type(self.tokenizer).__name__))
        key_payload = {
            "v": 1,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "max_seq_len": max_seq_len,
            "max_samples": max_samples,
            "tokenizer": tokenizer_name,
        }
        encoded = json.dumps(key_payload, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return Path(cache_dir) / f"text_chunks_{digest}.pt"

    def _load_from_cache(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False

        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning("Failed to load text cache %s (%s)", cache_path, exc)
            return False

        chunks_tensor = payload.get("chunks")
        if not isinstance(chunks_tensor, torch.Tensor):
            return False
        if chunks_tensor.ndim != 2:
            return False

        self.chunks = [row.clone().to(dtype=torch.long) for row in chunks_tensor]
        logger.info("Loaded %d chunks from cache %s", len(self.chunks), cache_path)
        return True

    def _save_to_cache(self, cache_path: Path) -> None:
        if not self.chunks:
            return

        if not self._ensure_cache_dir(cache_path.parent):
            return
        lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
        lock_fd = None
        start_time = time.time()

        while lock_fd is None and (time.time() - start_time) < 120:
            try:
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if cache_path.exists():
                    return
                time.sleep(0.2)

        if lock_fd is None:
            logger.warning("Could not acquire text cache lock for %s", cache_path)
            return

        os.close(lock_fd)

        try:
            if cache_path.exists():
                return

            chunks_tensor = torch.stack(self.chunks).cpu()
            payload = {
                "chunks": chunks_tensor,
                "created_at": int(time.time()),
            }
            tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(cache_path)
            logger.info("Saved text cache to %s", cache_path)
        except Exception as exc:
            logger.warning("Failed to save text cache %s (%s)", cache_path, exc)
        finally:
            try:
                if lock_path.exists():
                    lock_path.unlink()
            except Exception:
                logger.warning("Could not remove cache lock %s", lock_path)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = self.chunks[idx]
        # Labels are shifted by 1 (next token prediction)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last position

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Tool Calling Dataset (Phase 3)
# ---------------------------------------------------------------------------

class ToolCallingDataset(Dataset):
    """
    Dataset of synthetic tool calling conversations.
    Loaded from a JSONL file generated by generate_synthetic_data().
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str = "data/tool_calling_train.jsonl",
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[dict] = []

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Tool calling data not found at {data_path}. "
                "Run `poetry run generate-data` first."
            )

        with open(path) as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        logger.info("Loaded %d tool calling examples from %s", len(self.examples), data_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]
        messages = example["messages"]

        # Format messages into a single string
        text = self._format_conversation(messages)

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels: mask system and user turns, only train on assistant + tool responses
        labels = self._create_labels(text, input_ids, messages)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format chat messages into a flat string."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
            elif role == "tool":
                parts.append(f"<start_of_turn>tool\n{content}<end_of_turn>")
        return "\n".join(parts)

    def _create_labels(
        self,
        text: str,
        input_ids: torch.Tensor,
        messages: list[dict],
    ) -> torch.Tensor:
        """
        Create labels that only compute loss on assistant responses.
        System/user/tool turns are masked with -100.
        """
        labels = input_ids.clone()

        # Simple approach: find assistant turn boundaries and mask everything else.
        # Decode each token position to find which role it belongs to.
        # For efficiency, we mask based on string position mapping.
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        # Find positions of model turns
        model_start = "<start_of_turn>model\n"
        model_end = "<end_of_turn>"

        # By default mask everything
        labels[:] = -100

        pos = 0
        while True:
            start = decoded.find(model_start, pos)
            if start == -1:
                break
            content_start = start + len(model_start)
            end = decoded.find(model_end, content_start)
            if end == -1:
                end = len(decoded)

            # Map string positions to token positions
            prefix = decoded[:content_start]
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            content = decoded[content_start:end]
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)

            tok_start = len(prefix_tokens)
            tok_end = min(tok_start + len(content_tokens), len(labels))

            # Unmask assistant content
            labels[tok_start:tok_end] = input_ids[tok_start:tok_end]

            pos = end + len(model_end)

        return labels


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

# Scenario templates for generating synthetic tool calling conversations
_SCENARIOS = [
    # Person recognition (40%)
    {
        "category": "person_recognition",
        "user_templates": [
            "Someone just walked in. Who is that?",
            "There's a woman at the door. Do I know her?",
            "A man is sitting next to me. Who is he?",
            "I see someone familiar but I can't remember their name.",
            "Who is this person visiting me?",
            "This person says they know me. Is that true?",
            "A young girl just gave me a hug. Who is she?",
            "Someone brought me flowers. Do you know them?",
            "There's someone waving at me. Should I wave back?",
            "I think I know that face. Can you help me?",
            "Somebody is at the gate. Is it safe to open?",
            "A man with glasses just came in. Do I know him?",
            "Who is the lady making food in the kitchen?",
            "Someone just called my name. Who was that?",
            "There are people in the living room. Who are they?",
        ],
        "tool_name": "read_person",
        "weight": 0.4,
    },
    # Orientation (20%)
    {
        "category": "orientation",
        "user_templates": [
            "Where am I right now?",
            "I don't recognize this place. Where am I?",
            "How do I get to the bathroom from here?",
            "What room is this?",
            "I woke up and I'm confused. Where am I?",
            "Is this my house?",
            "I'm lost. Can you tell me where I am?",
            "This room looks different. What happened?",
            "Where is the kitchen? I'm hungry.",
            "I was walking and now I don't know where I ended up.",
            "Am I still at home?",
            "Which way is the bedroom? I want to rest.",
        ],
        "tool_name": "describe_location",
        "weight": 0.2,
    },
    # Medication (15%)
    {
        "category": "medication",
        "user_templates": [
            "Do I need to take any medicine now?",
            "What pills do I take in the morning?",
            "I think I forgot my medication. What should I take?",
            "Which pill is the small white one?",
            "Is it time for my medicine?",
            "Did I already take my pills today?",
            "I have some pills here but I don't remember what they're for.",
            "How many pills do I take at night?",
            "The pink pill — what is that one for?",
            "Rosa said I need my medicine. Which one?",
            "Should I take anything with lunch?",
        ],
        "tool_name": "get_medication",
        "weight": 0.15,
    },
    # Memory sharing (15%)
    {
        "category": "memory",
        "user_templates": [
            "Tell me about Maria. Who is she?",
            "Do I have grandchildren?",
            "Tell me about my wife.",
            "What did I use to do with my son?",
            "Who used to visit me on Sundays?",
            "Do I have a brother or sister?",
            "Who is the person in that photo on the wall?",
            "Did I use to have a job? What did I do?",
            "Someone mentioned João. Who is that?",
            "Was I married? Tell me about her.",
            "I dreamed about a woman. I think she was important to me.",
            "Who brings me cake on my birthday?",
        ],
        "tool_name": "read_person",
        "weight": 0.15,
    },
    # Caregiver alert (10%)
    {
        "category": "alert",
        "user_templates": [
            "I'm very confused and scared.",
            "I don't know where I am and I want to go home.",
            "I've been walking around and I can't find my room.",
            "I feel dizzy and unsteady.",
            "I can't remember anything today. Something is wrong.",
            "I fell down and I can't get up.",
            "I feel bad. Something hurts but I don't know what.",
            "I'm scared. I don't recognize anything.",
            "Please help me. I'm all alone.",
            "I've been calling but nobody comes.",
            "I think I took the wrong pill.",
            "My chest feels tight. I need help.",
        ],
        "tool_name": "alert_caregiver",
        "weight": 0.1,
    },
]

# Sample persons for data generation
_SAMPLE_PERSONS = [
    # Family — close relatives
    {"face_id": "face_001", "name": "Maria", "relationship": "granddaughter", "bio": "Maria is 8 years old. She loves drawing and visits every Sunday. You used to take her fishing at the lake."},
    {"face_id": "face_002", "name": "Ana", "relationship": "wife", "bio": "Ana is your wife of 45 years. She loved gardening and cooking. Her favorite flower was sunflower."},
    {"face_id": "face_003", "name": "João", "relationship": "son", "bio": "João is your oldest son. He is an engineer and lives in São Paulo. He calls every Wednesday."},
    {"face_id": "face_004", "name": "Beatriz", "relationship": "daughter", "bio": "Beatriz is your daughter. She is a school teacher and lives nearby. She brings you fresh bread on Fridays."},
    {"face_id": "face_005", "name": "Carlos", "relationship": "son", "bio": "Carlos is your younger son. He is a musician and plays guitar. He named his band after your dog."},
    {"face_id": "face_006", "name": "Lucas", "relationship": "grandson", "bio": "Lucas is 12 years old. He is João's son and loves playing soccer. He wants to be a pilot when he grows up."},
    {"face_id": "face_007", "name": "Helena", "relationship": "sister", "bio": "Helena is your younger sister. She lives in Belo Horizonte and visits during holidays. You grew up together on a farm."},
    {"face_id": "face_008", "name": "Roberto", "relationship": "brother", "bio": "Roberto is your older brother. He was a fisherman. You two built a treehouse together as kids."},
    # Extended family
    {"face_id": "face_009", "name": "Clara", "relationship": "daughter-in-law", "bio": "Clara is João's wife. She is a nurse and always checks your blood pressure when she visits."},
    {"face_id": "face_010", "name": "Miguel", "relationship": "great-grandson", "bio": "Miguel is 2 years old. He is Lucas's baby brother. He always laughs when you make funny faces."},
    {"face_id": "face_011", "name": "Teresa", "relationship": "niece", "bio": "Teresa is Helena's daughter. She is a lawyer in Brasília. She calls you 'Tio Querido'."},
    # Caregivers & medical
    {"face_id": "face_012", "name": "Rosa", "relationship": "caregiver", "bio": "Rosa has been your caregiver for 3 years. She is very kind and makes your favorite soup."},
    {"face_id": "face_013", "name": "Fernanda", "relationship": "caregiver", "bio": "Fernanda is your weekend caregiver. She helps you with exercises in the morning and reads to you."},
    {"face_id": "face_014", "name": "Dr. Silva", "relationship": "neurologist", "bio": "Dr. Silva is your neurologist. He visits once a month to check on you."},
    {"face_id": "face_015", "name": "Dr. Oliveira", "relationship": "cardiologist", "bio": "Dr. Oliveira is your heart doctor. She comes every three months. She always brings a puzzle for you."},
    {"face_id": "face_016", "name": "Enfermeira Lucia", "relationship": "nurse", "bio": "Lucia is the home nurse who comes twice a week. She checks your medication and blood sugar."},
    # Friends & neighbors
    {"face_id": "face_017", "name": "Pedro", "relationship": "neighbor", "bio": "Pedro lives next door. You play chess together on Saturday afternoons."},
    {"face_id": "face_018", "name": "Antônio", "relationship": "best friend", "bio": "Antônio is your best friend since childhood. You were in the army together. He brings you coffee on Tuesdays."},
    {"face_id": "face_019", "name": "Dona Margarida", "relationship": "neighbor", "bio": "Dona Margarida lives across the street. She brings you homemade cake on your birthday every year."},
    {"face_id": "face_020", "name": "Padre Francisco", "relationship": "parish priest", "bio": "Padre Francisco is from the local church. He visits on Sundays and you enjoy talking about philosophy."},
    {"face_id": "face_021", "name": "Sérgio", "relationship": "former colleague", "bio": "Sérgio worked with you at the factory for 20 years. You retired on the same day. He visits monthly."},
    {"face_id": "face_022", "name": "Lúcia", "relationship": "friend", "bio": "Lúcia is Ana's best friend. She comes every Thursday for afternoon tea and brings photo albums."},
    {"face_id": "face_023", "name": "Rafael", "relationship": "grandson-in-law", "bio": "Rafael is married to Teresa. He is a chef and always brings you special desserts when he visits."},
    {"face_id": "face_024", "name": "Dona Ilda", "relationship": "neighbor", "bio": "Dona Ilda is your oldest neighbor. She's 92 and remembers your wedding. She lives two houses down."},
]

_SAMPLE_MEDICATIONS = {
    "morning": [
        {"name": "Donepezil", "description": "small white round pill", "dosage": "5mg", "notes": "Take with breakfast"},
        {"name": "Memantine", "description": "oval beige pill", "dosage": "10mg", "notes": "Take with breakfast"},
        {"name": "Lisinopril", "description": "small pink pill", "dosage": "10mg", "notes": "Take for blood pressure, with water"},
    ],
    "afternoon": [
        {"name": "Vitamin D", "description": "yellow soft capsule", "dosage": "1000 IU", "notes": "Take with lunch"},
        {"name": "Calcium", "description": "large white round tablet", "dosage": "500mg", "notes": "Take with food for bone health"},
    ],
    "evening": [
        {"name": "Aspirin", "description": "small white pill", "dosage": "81mg", "notes": "Take with dinner for heart health"},
    ],
    "night": [
        {"name": "Melatonin", "description": "small blue pill", "dosage": "3mg", "notes": "Take 30 minutes before bed"},
        {"name": "Omeprazole", "description": "purple and gray capsule", "dosage": "20mg", "notes": "Take before bed on empty stomach"},
    ],
}

_SAMPLE_LOCATIONS = {
    "stove,refrigerator,table": {"name": "Kitchen", "description": "Your kitchen where you have breakfast every morning. Ana's recipe notebook is on the shelf."},
    "sofa,television,bookshelf": {"name": "Living Room", "description": "The main room where you watch TV and read. Your favorite chair is by the window."},
    "bed,wardrobe,mirror": {"name": "Bedroom", "description": "Your bedroom. The bathroom is through the door on the left. Ana's photo is on the nightstand."},
    "plants,bench,sunflowers": {"name": "Garden", "description": "The garden where Ana planted her sunflowers. Pedro's house is visible over the fence."},
    "sink,towel,tiles": {"name": "Bathroom", "description": "Your bathroom. The handrail is on the right side. Your toothbrush is the blue one."},
    "desk,lamp,photos": {"name": "Study", "description": "Your old study room. It has your books, family photos, and the chess set Pedro gave you."},
    "dining table,china cabinet,chandelier": {"name": "Dining Room", "description": "The dining room where the family gathers on Sundays. Ana's china set is in the cabinet."},
    "washing machine,clothes line,iron": {"name": "Laundry Room", "description": "The laundry room at the back of the house. Rosa organizes your clothes here."},
    "rocking chair,porch,street view": {"name": "Front Porch", "description": "The front porch where you sit in the afternoon. Dona Margarida's house is across the street."},
    "tools,workbench,wood": {"name": "Workshop", "description": "Your workshop in the garage. You used to build birdhouses here with João."},
}


# ---------------------------------------------------------------------------
# Gemma 4 Data Generator
# ---------------------------------------------------------------------------

# Female names for pronoun selection in generated queries
_FEMALE_NAMES = {
    "Maria", "Ana", "Beatriz", "Helena", "Clara", "Teresa",
    "Rosa", "Fernanda", "Dr. Oliveira", "Enfermeira Lucia",
    "Dona Margarida", "Lúcia", "Dona Ilda",
}


def _pronoun(person: dict) -> str:
    """Return 'she' or 'he' based on the person's name."""
    return "she" if person["name"] in _FEMALE_NAMES else "he"


class GemmaDataGenerator:
    """
    Uses Gemma 4 to generate diverse, natural synthetic training data.

    Gemma generates user queries and assistant responses while tool calls
    and tool responses remain deterministic (valid JSON structures).
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-e2b-it",
        device: str = "cuda",
        torch_dtype: str = "float16",
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.92,
    ):
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        logger.info("Loading Gemma model %s for data generation...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype_map.get(torch_dtype, torch.bfloat16),
            device_map=device if device == "auto" else {"": device},
        )
        self.model.eval()
        logger.info("Gemma model loaded on %s", device)

    def cleanup(self) -> None:
        """Free GPU memory held by the local model."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """Run a single generation with Gemma."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )
        # Decode only the newly generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.no_grad()
    def _generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for multiple prompts sequentially.

        Batched generation with padding causes NaN/Inf CUDA asserts
        in float16 that corrupt the CUDA context irrecoverably.
        Sequential generation avoids padding entirely.
        """
        results = []
        for prompt in prompts:
            results.append(self._generate(prompt))
        return results

    def generate_user_queries(self, category: str, num: int = 10) -> list[str]:
        """Generate diverse user queries for a scenario category using Gemma."""
        category_descriptions = {
            "person_recognition": (
                "The patient sees someone and needs help identifying who the person is. "
                "They may be confused, curious, or scared."
            ),
            "orientation": (
                "The patient is disoriented and doesn't know where they are. "
                "They may be confused about the room, the house, or their surroundings."
            ),
            "medication": (
                "The patient needs help with their medication. They may have forgotten "
                "whether they took their pills, or want to know what to take."
            ),
            "memory": (
                "The patient wants to recall memories about a family member or friend. "
                "They want to know about relationships and shared experiences."
            ),
            "alert": (
                "The patient is distressed, confused, lost, or in potential danger. "
                "They may need immediate help from a caregiver."
            ),
        }

        desc = category_descriptions.get(category, "The patient needs help.")
        prompt = (
            "<start_of_turn>user\n"
            "You are helping generate training data for an Alzheimer's memory companion device.\n"
            f"Scenario: {desc}\n\n"
            f"Generate exactly {num} diverse, realistic things an elderly Alzheimer's patient "
            "might say in this situation. Each should be 1-2 sentences, natural spoken language, "
            "varying in emotion (calm, confused, scared, curious). "
            f"Output a numbered list from 1 to {num}. One per line. "
            "Do NOT add explanations or commentary.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            "1."
        )
        raw = "1." + self._generate(prompt)
        queries = self._parse_numbered_list(raw, num)
        # Fallback to templates if Gemma output is insufficient
        if len(queries) < 3:
            logger.warning(
                "Gemma generated only %d queries for '%s', supplementing with templates",
                len(queries), category,
            )
            for scenario in _SCENARIOS:
                if scenario["category"] == category:
                    queries.extend(scenario["user_templates"])
                    break
        return queries

    def generate_assistant_response(
        self,
        user_query: str,
        tool_name: str,
        tool_result: dict,
        context_hint: str,
    ) -> str:
        """Generate a warm, natural assistant response given tool results."""
        prompt = (
            "<start_of_turn>user\n"
            "You are a warm, caring memory companion for an Alzheimer's patient.\n"
            "The patient said: \"{user_query}\"\n"
            "You called the tool '{tool_name}' and got this result:\n"
            "{tool_result_json}\n\n"
            "Context: {context_hint}\n\n"
            "Write a warm, reassuring response (2-4 sentences). Use simple language. "
            "Be gentle and kind. Include specific details from the tool result. "
            "Do NOT fabricate information not in the tool result.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        ).format(
            user_query=user_query,
            tool_name=tool_name,
            tool_result_json=json.dumps(tool_result, indent=2),
            context_hint=context_hint,
        )
        response = self._generate(prompt)
        # Clean up: take first paragraph, remove any meta-commentary
        response = response.split("\n\n")[0].strip()
        if not response or len(response) < 10:
            return context_hint  # Fallback to hint
        return response

    @staticmethod
    def _parse_numbered_list(text: str, expected: int) -> list[str]:
        """Parse a numbered list from Gemma output.

        Handles multiple formats:
        - Numbered: "1. ...", "1) ...", "1: ...", "1- ..."
        - Bulleted: "- ...", "* ...", "• ..."
        - Quoted lines: '"..."'
        - Plain lines (fallback)
        """
        lines = text.strip().split("\n")
        queries: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try numbered patterns: "1. ...", "1) ...", "1: ...", "1- ..."
            match = re.match(r"^\s*\d+[\.\)\:\-]\s*(.+)", line)
            if match:
                q = match.group(1).strip().strip('"').strip("'")
                if len(q) > 5:
                    queries.append(q)
                continue
            # Try bullet patterns: "- ...", "* ...", "• ..."
            match = re.match(r"^\s*[\-\*\•]\s+(.+)", line)
            if match:
                q = match.group(1).strip().strip('"').strip("'")
                if len(q) > 5:
                    queries.append(q)
                continue
            # Try quoted lines: "..."
            match = re.match(r'^\s*["\'](.{6,})["\']\s*$', line)
            if match:
                queries.append(match.group(1).strip())
                continue
            # Fallback: plain line that looks like a sentence
            if len(line) > 10 and not line.startswith(("#", "Scenario", "Here", "Sure")):
                queries.append(line.strip('"').strip("'"))
        return queries[:expected]

    def generate_example(self, category: str, tc_cfg: ToolCallingConfig) -> dict:
        """Generate a single training example using Gemma for natural language parts."""
        if category in ("person_recognition", "memory"):
            return self._gen_person_example(category, tc_cfg)
        elif category == "orientation":
            return self._gen_orientation_example(tc_cfg)
        elif category == "medication":
            return self._gen_medication_example(tc_cfg)
        elif category == "alert":
            return self._gen_alert_example(tc_cfg)
        else:
            raise ValueError(f"Unknown category: {category}")

    def generate_examples_batch(
        self,
        categories: list[str],
        tc_cfg: ToolCallingConfig,
        batch_size: int = 8,
    ) -> list[dict]:
        """Generate multiple examples with batched GPU calls for higher utilization.

        Instead of generating responses one-at-a-time, this:
        1. Prepares all tool calls / metadata on CPU (fast)
        2. Batches the assistant response prompts
        3. Sends them to GPU in one batch call
        """
        # Phase 1: Prepare all partial examples on CPU (no GPU needed)
        partials: list[dict] = []
        prompts: list[str] = []

        for category in categories:
            partial = self._prepare_example_partial(category, tc_cfg)
            partials.append(partial)
            prompts.append(partial["_response_prompt"])

        # Phase 2: Batch generate all assistant responses on GPU
        all_responses: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = self._generate_batch(batch_prompts)
            all_responses.extend(batch_responses)

        # Phase 3: Assemble final examples on CPU
        results: list[dict] = []
        for partial, response in zip(partials, all_responses):
            response = response.split("\n\n")[0].strip()
            if not response or len(response) < 10:
                response = partial["_hint"]
            results.append({
                "messages": [
                    {"role": "system", "content": tc_cfg.system_prompt},
                    {"role": "user", "content": partial["user_msg"]},
                    {"role": "assistant", "content": f"{tc_cfg.tool_call_start}\n{partial['tool_call']}\n{tc_cfg.tool_call_end}"},
                    {"role": "tool", "content": f"{tc_cfg.tool_response_start}\n{partial['tool_response']}\n{tc_cfg.tool_response_end}"},
                    {"role": "assistant", "content": response},
                ]
            })
        return results

    def _prepare_example_partial(self, category: str, tc_cfg: ToolCallingConfig) -> dict:
        """Prepare example metadata and response prompt without calling GPU.

        User queries are contextualized with the chosen entity to ensure
        coherence between what the patient says and the tool call/result.
        """
        if category in ("person_recognition", "memory"):
            person = random.choice(_SAMPLE_PERSONS)
            # Contextualize: replace generic queries with person-specific ones
            raw_query = random.choice(self._user_query_cache.get(category, _SCENARIOS[0]["user_templates"]))
            # If the query mentions a specific name that doesn't match, make it generic
            other_names = [p["name"] for p in _SAMPLE_PERSONS if p["name"] != person["name"]]
            for name in other_names:
                if name in raw_query:
                    raw_query = raw_query.replace(name, "this person")
            # For person_recognition: patient sees someone but doesn't know who
            # For memory: patient asks about a specific person
            if category == "memory" and person["name"] not in raw_query:
                user_msg = random.choice([
                    f"Tell me about {person['name']}. Who is {_pronoun(person)}?",
                    f"Do I know someone called {person['name']}?",
                    f"What can you tell me about {person['name']}?",
                    f"Who is {person['name']} to me?",
                    raw_query,
                ])
            else:
                user_msg = raw_query
            tool_call = json.dumps({"name": "read_person", "arguments": {"face_id": person["face_id"]}})
            tool_result = {"found": True, "name": person["name"], "relationship": person["relationship"], "bio": person["bio"], "memories": []}
            tool_response = json.dumps({"name": "read_person", "result": tool_result})
            hint = f"That's {person['name']}, your {person['relationship']}. {person['bio']}"
            tool_name = "read_person"
        elif category == "orientation":
            features_key = random.choice(list(_SAMPLE_LOCATIONS.keys()))
            location = _SAMPLE_LOCATIONS[features_key]
            user_msg = random.choice(self._user_query_cache.get("orientation", _SCENARIOS[1]["user_templates"]))
            tool_call = json.dumps({"name": "describe_location", "arguments": {"room_features": features_key}})
            tool_result = {"found": True, "name": location["name"], "description": location["description"]}
            tool_response = json.dumps({"name": "describe_location", "result": tool_result})
            hint = f"You're in your {location['name'].lower()}. {location['description']} You're safe at home."
            tool_name = "describe_location"
        elif category == "medication":
            time_of_day = random.choice(["morning", "afternoon", "evening"])
            user_msg = random.choice(self._user_query_cache.get("medication", _SCENARIOS[2]["user_templates"]))
            tool_call = json.dumps({"name": "get_medication", "arguments": {"time_of_day": time_of_day}})
            meds = [{"name": "Donepezil", "dosage": "10mg", "notes": "Take with food"}]
            tool_result = {"medications": meds}
            tool_response = json.dumps({"name": "get_medication", "result": tool_result})
            hint = f"It's {time_of_day} and you need to take Donepezil 10mg with food."
            tool_name = "get_medication"
        elif category == "alert":
            alert_type = random.choice(["wandering", "fall_risk", "confusion"])
            user_msg = random.choice(self._user_query_cache.get("alert", _SCENARIOS[4]["user_templates"]))
            tool_call = json.dumps({"name": "alert_caregiver", "arguments": {"alert_type": alert_type, "details": user_msg}})
            tool_result = {"sent": True, "caregiver": "Rosa", "method": "push_notification"}
            tool_response = json.dumps({"name": "alert_caregiver", "result": tool_result})
            hint = "I've let Rosa know, and she'll be with you shortly. You're safe at home."
            tool_name = "alert_caregiver"
        else:
            raise ValueError(f"Unknown category: {category}")

        # Build the response prompt (to be sent to GPU in batch)
        response_prompt = (
            "<start_of_turn>user\n"
            "You are a warm, caring memory companion for an Alzheimer's patient.\n"
            f"The patient said: \"{user_msg}\"\n"
            f"You called the tool '{tool_name}' and got this result:\n"
            f"{json.dumps(tool_result, indent=2)}\n\n"
            f"Context: {hint}\n\n"
            "Write a warm, reassuring response (2-4 sentences). Use simple language. "
            "Be gentle and kind. Include specific details from the tool result. "
            "Do NOT fabricate information not in the tool result.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        return {
            "user_msg": user_msg,
            "tool_call": tool_call,
            "tool_response": tool_response,
            "_response_prompt": response_prompt,
            "_hint": hint,
        }

    def _gen_person_example(self, category: str, tc_cfg: ToolCallingConfig) -> dict:
        person = random.choice(_SAMPLE_PERSONS)
        raw_query = random.choice(
            self._user_query_cache.get(category, _SCENARIOS[0]["user_templates"])
        )
        # Fix coherence: remove mismatched names from query
        for name in [p["name"] for p in _SAMPLE_PERSONS if p["name"] != person["name"]]:
            if name in raw_query:
                raw_query = raw_query.replace(name, "this person")
        if category == "memory" and person["name"] not in raw_query:
            user_msg = random.choice([
                f"Tell me about {person['name']}. Who is {_pronoun(person)}?",
                f"Do I know someone called {person['name']}?",
                f"What can you tell me about {person['name']}?",
                raw_query,
            ])
        else:
            user_msg = raw_query
        tool_call = json.dumps({"name": "read_person", "arguments": {"face_id": person["face_id"]}})
        tool_result = {
            "found": True, "name": person["name"],
            "relationship": person["relationship"], "bio": person["bio"], "memories": [],
        }
        tool_response = json.dumps({"name": "read_person", "result": tool_result})

        hint = f"That's {person['name']}, your {person['relationship']}. {person['bio']}"
        response = self.generate_assistant_response(user_msg, "read_person", tool_result, hint)

        return {
            "messages": [
                {"role": "system", "content": tc_cfg.system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"{tc_cfg.tool_call_start}\n{tool_call}\n{tc_cfg.tool_call_end}"},
                {"role": "tool", "content": f"{tc_cfg.tool_response_start}\n{tool_response}\n{tc_cfg.tool_response_end}"},
                {"role": "assistant", "content": response},
            ]
        }

    def _gen_orientation_example(self, tc_cfg: ToolCallingConfig) -> dict:
        features_key = random.choice(list(_SAMPLE_LOCATIONS.keys()))
        location = _SAMPLE_LOCATIONS[features_key]
        user_msg = random.choice(
            self._user_query_cache.get("orientation", _SCENARIOS[1]["user_templates"])
        )
        tool_call = json.dumps({"name": "describe_location", "arguments": {"room_features": features_key}})
        tool_result = {"found": True, "name": location["name"], "description": location["description"]}
        tool_response = json.dumps({"name": "describe_location", "result": tool_result})

        hint = f"You're in your {location['name'].lower()}. {location['description']} You're safe at home."
        response = self.generate_assistant_response(user_msg, "describe_location", tool_result, hint)

        return {
            "messages": [
                {"role": "system", "content": tc_cfg.system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"{tc_cfg.tool_call_start}\n{tool_call}\n{tc_cfg.tool_call_end}"},
                {"role": "tool", "content": f"{tc_cfg.tool_response_start}\n{tool_response}\n{tc_cfg.tool_response_end}"},
                {"role": "assistant", "content": response},
            ]
        }

    def _gen_medication_example(self, tc_cfg: ToolCallingConfig) -> dict:
        time_of_day = random.choice(["morning", "afternoon", "night"])
        meds = _SAMPLE_MEDICATIONS[time_of_day]
        user_msg = random.choice(
            self._user_query_cache.get("medication", _SCENARIOS[2]["user_templates"])
        )
        tool_call = json.dumps({"name": "get_medication", "arguments": {"time_of_day": time_of_day}})
        tool_result = {
            "medications": [
                {"name": m["name"], "description": m["description"], "dosage": m["dosage"], "notes": m["notes"]}
                for m in meds
            ]
        }
        tool_response = json.dumps({"name": "get_medication", "result": tool_result})

        if meds:
            med_names = ", ".join(m["name"] for m in meds)
            hint = f"Time for your {time_of_day} medication: {med_names}. {meds[0]['notes']}."
        else:
            hint = f"No medication scheduled for the {time_of_day}."
        response = self.generate_assistant_response(user_msg, "get_medication", tool_result, hint)

        return {
            "messages": [
                {"role": "system", "content": tc_cfg.system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"{tc_cfg.tool_call_start}\n{tool_call}\n{tc_cfg.tool_call_end}"},
                {"role": "tool", "content": f"{tc_cfg.tool_response_start}\n{tool_response}\n{tc_cfg.tool_response_end}"},
                {"role": "assistant", "content": response},
            ]
        }

    def _gen_alert_example(self, tc_cfg: ToolCallingConfig) -> dict:
        user_msg = random.choice(
            self._user_query_cache.get("alert", _SCENARIOS[4]["user_templates"])
        )
        alert_type = random.choice(["confusion", "wandering", "fall_risk"])
        tool_call = json.dumps({"name": "alert_caregiver", "arguments": {"alert_type": alert_type, "details": user_msg}})
        tool_result = {"success": True, "alert_id": f"alert_{random.randint(100,999)}", "type": alert_type}
        tool_response = json.dumps({"name": "alert_caregiver", "result": tool_result})

        hint = (
            "I understand you're feeling unsettled. That's okay. "
            "I've let Rosa know, and she'll be with you shortly. You're safe at home."
        )
        response = self.generate_assistant_response(user_msg, "alert_caregiver", tool_result, hint)

        return {
            "messages": [
                {"role": "system", "content": tc_cfg.system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"{tc_cfg.tool_call_start}\n{tool_call}\n{tc_cfg.tool_call_end}"},
                {"role": "tool", "content": f"{tc_cfg.tool_response_start}\n{tool_response}\n{tc_cfg.tool_response_end}"},
                {"role": "assistant", "content": response},
            ]
        }

    def warm_up_query_cache(self, queries_per_category: int = 30) -> None:
        """Pre-generate a pool of diverse user queries for each category."""
        self._user_query_cache: dict[str, list[str]] = {}
        for scenario in _SCENARIOS:
            cat = scenario["category"]
            logger.info("Generating %d user queries for '%s'...", queries_per_category, cat)
            generated = self.generate_user_queries(cat, num=queries_per_category)
            # Combine generated + template queries for maximum diversity
            combined = list(set(generated + scenario["user_templates"]))
            self._user_query_cache[cat] = combined
            logger.info("  → %d unique queries for '%s'", len(combined), cat)


# ---------------------------------------------------------------------------
# Gemini API Generator (uses google-genai SDK for larger models)
# ---------------------------------------------------------------------------

class GeminiApiGenerator(GemmaDataGenerator):
    """
    Data generator that uses the Gemini API for a larger Gemma model.

    Inherits all high-level logic from GemmaDataGenerator but replaces
    the local model inference with remote API calls via google-genai SDK.
    Suitable for larger models (e.g. gemma-4-31b-it) that cannot fit in
    local VRAM.

    Supports rate-limited concurrent requests to maximize throughput
    while staying within the API RPM limit (default: 14 RPM safe margin).
    """

    def __init__(
        self,
        model_name: str = "gemma-4-31b-it",
        api_key: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.92,
        rpm_limit: int = 14,
    ):
        # Skip GemmaDataGenerator.__init__ — no local model needed
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = "api"
        self._user_query_cache: dict[str, list[str]] = {}

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Gemini API mode. "
                "Install with: poetry add google-genai"
            ) from exc

        resolved_key = api_key or _resolve_gemini_api_key()
        if not resolved_key:
            raise ValueError(
                "No API key found. Set KAGGLE_KEY, GOOGLE_API_KEY, or "
                "GEMINI_API_KEY environment variable, or pass --api-key."
            )

        self._client = genai.Client(api_key=resolved_key)
        self._types = types
        self._model_name = model_name

        # Rate limiter: token-bucket with rpm_limit tokens per 60s
        self._rpm_limit = rpm_limit
        self._rate_lock = threading.Lock()
        self._request_times: list[float] = []

        logger.info(
            "GeminiApiGenerator initialized with model=%s, rpm_limit=%d",
            model_name, rpm_limit,
        )

    @property
    def model(self) -> None:
        """Compatibility shim — API generator has no local model."""
        return None

    def _wait_for_rate_limit(self) -> None:
        """Block until a request slot is available within the RPM window."""
        with self._rate_lock:
            now = time.monotonic()
            window = 60.0
            # Purge timestamps older than the window
            self._request_times = [
                t for t in self._request_times if now - t < window
            ]
            if len(self._request_times) >= self._rpm_limit:
                # Wait until the oldest request in the window expires
                sleep_time = window - (now - self._request_times[0]) + 0.1
                if sleep_time > 0:
                    logger.debug("Rate limit reached, waiting %.1fs", sleep_time)
                    # Release lock while sleeping so other threads can check
                    self._rate_lock.release()
                    try:
                        time.sleep(sleep_time)
                    finally:
                        self._rate_lock.acquire()
                    # Re-purge after sleep
                    now = time.monotonic()
                    self._request_times = [
                        t for t in self._request_times if now - t < window
                    ]
            self._request_times.append(time.monotonic())

    def _generate(self, prompt: str) -> str:
        """Call the Gemini API for a single generation with rate limiting."""
        from google.genai import errors as genai_errors

        self._wait_for_rate_limit()

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_new_tokens,
                ),
            )
            return (response.text or "").strip()
        except genai_errors.APIError:
            raise  # Propagate for fallback handling

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses concurrently via API with rate limiting.

        Uses a thread pool to fire up to rpm_limit concurrent requests,
        with each request respecting the token-bucket rate limiter.
        This achieves ~14 RPM throughput vs ~3 RPM sequential.
        """
        if len(prompts) <= 1:
            return [self._generate(p) for p in prompts]

        results: list[str | None] = [None] * len(prompts)
        max_workers = min(len(prompts), self._rpm_limit)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._generate, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # Propagates exceptions

        return results  # type: ignore[return-value]

    def cleanup(self) -> None:
        """Close the API client."""
        if hasattr(self, "_client"):
            self._client.close()


class QuotaExhaustedError(Exception):
    """Raised when the Gemini API quota is exhausted."""


def _resolve_gemini_api_key() -> str | None:
    """Resolve the API key from multiple sources (in priority order).

    1. Environment variables: GEMINI_API_KEY, GOOGLE_API_KEY, KAGGLE_KEY
    2. Kaggle Secrets vault (available inside Kaggle notebooks)
    """
    import os
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "KAGGLE_KEY"):
        key = os.environ.get(var)
        if key:
            return key

    # Fallback: Kaggle Secrets vault (only available in Kaggle runtime)
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("KAGGLE_KEY")
        if key:
            return key
    except Exception:
        pass

    return None


def _is_quota_error(exc: Exception) -> bool:
    """Check if an API error indicates exhausted credits.

    Uses duck typing: any exception with a .code attribute matching
    429/403/503 is treated as a quota error. This avoids tight coupling
    to google.genai.errors.APIError while remaining compatible with it.
    """
    code = getattr(exc, "code", None)
    if code is None:
        return False
    return code in (429, 403, 503)


def _create_generator(
    *,
    use_api: bool = False,
    api_model: str = "gemma-4-31b-it",
    api_key: str | None = None,
    local_model: str = "google/gemma-4-e2b-it",
    device: str = "cuda",
    torch_dtype: str = "float16",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.92,
    rpm_limit: int = 14,
) -> tuple[GemmaDataGenerator, bool]:
    """
    Factory that creates the best available generator.

    Returns (generator, is_api) tuple. The is_api flag indicates whether
    the returned generator is API-based (True) or local (False).
    Callers use this to decide fallback strategy.
    """
    if use_api:
        try:
            gen = GeminiApiGenerator(
                model_name=api_model,
                api_key=api_key,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                rpm_limit=rpm_limit,
            )
            # Smoke test: one small generation to verify credits
            gen._generate("Say hello in one word.")
            logger.info("Gemini API is available — using %s for generation", api_model)
            return gen, True
        except Exception as exc:
            if _is_quota_error(exc):
                logger.warning(
                    "Gemini API quota exhausted on startup, falling back to parallel multi-GPU"
                )
            else:
                logger.warning(
                    "Gemini API unavailable (%s), falling back to parallel multi-GPU", exc
                )

    local_gen = GemmaDataGenerator(
        model_name=local_model,
        device=device,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return local_gen, False


def _generate_example_with_fallback(
    gen: GemmaDataGenerator,
    category: str,
    tc_cfg: ToolCallingConfig,
    *,
    local_model: str = "google/gemma-4-e2b-it",
    device: str = "cuda",
    torch_dtype: str = "float16",
) -> tuple[dict, GemmaDataGenerator]:
    """
    Generate one example, falling back to local GPU if API quota runs out.

    Returns the example and the (possibly swapped) generator.
    """
    try:
        return gen.generate_example(category, tc_cfg), gen
    except Exception as exc:
        if not isinstance(gen, GeminiApiGenerator) or not _is_quota_error(exc):
            raise

        logger.warning("Gemini API quota exhausted mid-generation, switching to local GPU")
        gen.cleanup()
        local_gen = GemmaDataGenerator(
            model_name=local_model,
            device=device,
            torch_dtype=torch_dtype,
        )
        local_gen._user_query_cache = gen._user_query_cache
        local_gen.warm_up_query_cache()
        return local_gen.generate_example(category, tc_cfg), local_gen


def _generate_batch_with_fallback(
    gen: GemmaDataGenerator,
    categories: list[str],
    tc_cfg: ToolCallingConfig,
    batch_size: int = 8,
    *,
    local_model: str = "google/gemma-4-e2b-it",
    device: str = "cuda",
    torch_dtype: str = "float16",
) -> tuple[list[dict], GemmaDataGenerator]:
    """
    Generate a batch of examples, falling back to local GPU if API quota runs out.

    Returns the examples and the (possibly swapped) generator.
    """
    try:
        examples = gen.generate_examples_batch(categories, tc_cfg, batch_size=batch_size)
        return examples, gen
    except Exception as exc:
        if not isinstance(gen, GeminiApiGenerator) or not _is_quota_error(exc):
            raise

        logger.warning("Gemini API quota exhausted mid-batch, switching to local GPU")
        gen.cleanup()
        local_gen = GemmaDataGenerator(
            model_name=local_model,
            device=device,
            torch_dtype=torch_dtype,
        )
        local_gen._user_query_cache = gen._user_query_cache
        local_gen.warm_up_query_cache()
        examples = local_gen.generate_examples_batch(categories, tc_cfg, batch_size=batch_size)
        return examples, local_gen


def _generate_person_recognition_example(tc_cfg: ToolCallingConfig) -> dict:
    """Generate a person recognition conversation."""
    person = random.choice(_SAMPLE_PERSONS)
    user_msg = random.choice(_SCENARIOS[0]["user_templates"])

    tool_call = json.dumps({"name": "read_person", "arguments": {"face_id": person["face_id"]}})
    tool_response = json.dumps({
        "name": "read_person",
        "result": {
            "found": True,
            "name": person["name"],
            "relationship": person["relationship"],
            "bio": person["bio"],
            "memories": [],
        },
    })

    # Generate warm response
    response = (
        f"That's {person['name']}, your {person['relationship']}! "
        f"{person['bio']}"
    )

    return {
        "messages": [
            {"role": "system", "content": tc_cfg.system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"{tc_cfg.tool_call_start}\n{tool_call}\n{tc_cfg.tool_call_end}"},
            {"role": "tool", "content": f"{tc_cfg.tool_response_start}\n{tool_response}\n{tc_cfg.tool_response_end}"},
            {"role": "assistant", "content": response},
        ]
    }


def _generate_orientation_example(tc_cfg: ToolCallingConfig) -> dict:
    """Generate an orientation conversation."""
    features_key = random.choice(list(_SAMPLE_LOCATIONS.keys()))
    location = _SAMPLE_LOCATIONS[features_key]
    user_msg = random.choice(_SCENARIOS[1]["user_templates"])

    tool_call = json.dumps({"name": "describe_location", "arguments": {"room_features": features_key}})
    tool_response = json.dumps({
        "name": "describe_location",
        "result": {"found": True, "name": location["name"], "description": location["description"]},
    })

    response = f"You're in your {location['name'].lower()}. {location['description']} You're safe at home."

    return {
        "messages": [
            {"role": "system", "content": tc_cfg.system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"{tc_cfg.tool_call_start}\n{tool_call}\n{tc_cfg.tool_call_end}"},
            {"role": "tool", "content": f"{tc_cfg.tool_response_start}\n{tool_response}\n{tc_cfg.tool_response_end}"},
            {"role": "assistant", "content": response},
        ]
    }


def _generate_medication_example(tc_cfg: ToolCallingConfig) -> dict:
    """Generate a medication conversation."""
    time_of_day = random.choice(["morning", "afternoon", "night"])
    meds = _SAMPLE_MEDICATIONS[time_of_day]
    user_msg = random.choice(_SCENARIOS[2]["user_templates"])

    tool_call = json.dumps({"name": "get_medication", "arguments": {"time_of_day": time_of_day}})
    tool_response = json.dumps({
        "name": "get_medication",
        "result": {"medications": [{"name": m["name"], "description": m["description"], "dosage": m["dosage"], "notes": m["notes"]} for m in meds]},
    })

    if meds:
        med_descriptions = [f"the {m['description']} ({m['name']}, {m['dosage']})" for m in meds]
        response = f"It's time for your {time_of_day} medication. You need to take {', and '.join(med_descriptions)}. {meds[0]['notes']}."
    else:
        response = f"You don't have any medication scheduled for the {time_of_day}. Your next medication is later."

    return {
        "messages": [
            {"role": "system", "content": tc_cfg.system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"{tc_cfg.tool_call_start}\n{tool_call}\n{tc_cfg.tool_call_end}"},
            {"role": "tool", "content": f"{tc_cfg.tool_response_start}\n{tool_response}\n{tc_cfg.tool_response_end}"},
            {"role": "assistant", "content": response},
        ]
    }


def _generate_alert_example(tc_cfg: ToolCallingConfig) -> dict:
    """Generate a caregiver alert conversation."""
    user_msg = random.choice(_SCENARIOS[4]["user_templates"])
    alert_types = ["confusion", "wandering", "fall_risk"]
    alert_type = random.choice(alert_types)

    tool_call = json.dumps({"name": "alert_caregiver", "arguments": {"alert_type": alert_type, "details": user_msg}})
    tool_response = json.dumps({
        "name": "alert_caregiver",
        "result": {"success": True, "alert_id": "alert_123", "type": alert_type},
    })

    response = (
        "I understand you're feeling unsettled. That's okay — it happens sometimes. "
        "I've let Rosa know, and she'll be with you shortly. "
        "You're safe at home. Would you like me to tell you about where you are?"
    )

    return {
        "messages": [
            {"role": "system", "content": tc_cfg.system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"{tc_cfg.tool_call_start}\n{tool_call}\n{tc_cfg.tool_call_end}"},
            {"role": "tool", "content": f"{tc_cfg.tool_response_start}\n{tool_response}\n{tc_cfg.tool_response_end}"},
            {"role": "assistant", "content": response},
        ]
    }


_GENERATORS = {
    "person_recognition": _generate_person_recognition_example,
    "orientation": _generate_orientation_example,
    "medication": _generate_medication_example,
    "memory": _generate_person_recognition_example,  # Reuses person data
    "alert": _generate_alert_example,
}


def _parallel_worker(
    rank: int,
    num_examples: int,
    split: str,
    output_path: str,
    scenarios_weighted: list[str],
    gemma_model: str,
    torch_dtype: str,
    queries_per_category: int,
    seed: int,
    batch_size: int = 8,
) -> None:
    """Worker process for parallel data generation on a specific GPU.

    Uses batched generation to maximize GPU utilization and minimize
    CPU-GPU synchronization overhead.
    """
    # Each worker gets its own seed for different random outputs
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    device = f"cuda:{rank}"
    tc_cfg = ToolCallingConfig()

    logging.basicConfig(level=logging.INFO)
    worker_logger = logging.getLogger(f"datagen.worker.{rank}")
    worker_logger.info("Worker %d starting on %s, generating %d %s examples (batch=%d)",
                       rank, device, num_examples, split, batch_size)

    gen = GemmaDataGenerator(
        model_name=gemma_model,
        device=device,
        torch_dtype=torch_dtype,
    )
    gen.warm_up_query_cache(queries_per_category=queries_per_category)

    generated = 0
    with open(output_path, "w", buffering=8192) as f:
        while generated < num_examples:
            # Prepare a batch of categories
            remaining = num_examples - generated
            current_batch = min(batch_size, remaining)
            categories = [random.choice(scenarios_weighted) for _ in range(current_batch)]

            # Batched generation: 1 GPU call for all responses
            examples = gen.generate_examples_batch(categories, tc_cfg, batch_size=current_batch)

            for example in examples:
                f.write(json.dumps(example) + "\n")
            generated += len(examples)

            if generated % 10 < current_batch:
                worker_logger.info("[GPU %d][%s] %d/%d examples", rank, split, generated, num_examples)

    worker_logger.info("Worker %d finished: %d examples → %s", rank, num_examples, output_path)

    # Free GPU memory
    del gen.model
    del gen
    torch.cuda.empty_cache()


def generate_synthetic_data_parallel(
    output_dir: str = "data",
    num_train: int = 25000,
    num_val: int = 1500,
    seed: int = 42,
    gemma_model: str = "google/gemma-4-e2b-it",
    torch_dtype: str = "float16",
    queries_per_category: int = 30,
    num_gpus: int = 2,
) -> None:
    """
    Generate synthetic data in parallel across multiple GPUs.

    Spawns one worker per GPU, each generating a portion of examples.
    Results are merged into final JSONL files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    available_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, available_gpus)
    if num_gpus < 2:
        logger.warning("Only %d GPU(s) available, falling back to single-GPU generation", available_gpus)
        generate_synthetic_data(
            output_dir=output_dir, num_train=num_train, num_val=num_val, seed=seed,
            use_gemma=True, gemma_model=gemma_model, torch_dtype=torch_dtype,
            queries_per_category=queries_per_category,
        )
        return

    # Build weighted scenario list (shared across workers)
    scenarios_weighted: list[str] = []
    for scenario in _SCENARIOS:
        count = int(scenario["weight"] * 100)
        scenarios_weighted.extend([scenario["category"]] * count)

    mp.set_start_method("spawn", force=True)

    for split, num_total in [("train", num_train), ("val", num_val)]:
        # Split work across GPUs
        examples_per_gpu = num_total // num_gpus
        remainder = num_total % num_gpus

        temp_files: list[str] = []
        processes: list[mp.Process] = []

        for rank in range(num_gpus):
            n = examples_per_gpu + (1 if rank < remainder else 0)
            temp_path = str(out / f"_tmp_{split}_gpu{rank}.jsonl")
            temp_files.append(temp_path)

            p = mp.Process(
                target=_parallel_worker,
                args=(
                    rank, n, split, temp_path, scenarios_weighted,
                    gemma_model, torch_dtype, queries_per_category,
                    seed,
                ),
            )
            processes.append(p)

        # Start all workers
        for p in processes:
            p.start()

        # Wait for completion
        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker process exited with code {p.exitcode}")

        # Merge temp files into final output
        final_path = out / f"tool_calling_{split}.jsonl"
        with open(final_path, "w") as fout:
            for temp_path in temp_files:
                tp = Path(temp_path)
                if tp.exists():
                    with open(tp) as fin:
                        for line in fin:
                            fout.write(line)
                    tp.unlink()  # Clean up temp file

        logger.info("Merged %d %s examples → %s", num_total, split, final_path)

    print(f"Parallel generation complete: {num_train} train, {num_val} val in {out}/ ({num_gpus} GPUs)")


def generate_synthetic_data(
    output_dir: str = "data",
    num_train: int = 25000,
    num_val: int = 1500,
    seed: int = 42,
    use_gemma: bool = True,
    gemma_model: str = "google/gemma-4-e2b-it",
    device: str = "cuda",
    torch_dtype: str = "float16",
    queries_per_category: int = 30,
    use_api: bool = False,
    api_model: str = "gemma-4-31b-it",
    api_key: str | None = None,
    num_gpus: int = 2,
    rpm_limit: int = 14,
) -> None:
    """
    Generate synthetic tool calling training data.

    By default uses Gemma 4 to produce diverse, natural conversations.
    Falls back to template-based generation with --no-gemma.

    When use_api=True, tries the Gemini API with a larger model
    first. Requests are parallelized up to rpm_limit per minute.
    If API credits run out (at startup or mid-generation),
    automatically falls back to multi-GPU parallel generation using
    all available GPUs.

    Args:
        output_dir: Output directory for JSONL files.
        num_train: Number of training examples.
        num_val: Number of validation examples.
        seed: Random seed.
        use_gemma: If True, use Gemma 4 for generation. If False, use templates.
        gemma_model: HuggingFace model ID for local data generation.
        device: Device for Gemma inference ('cuda', 'cpu', 'auto').
        torch_dtype: Dtype for Gemma model ('bfloat16', 'float16', 'float32').
        queries_per_category: Number of user queries to pre-generate per category.
        use_api: If True, try Gemini API with a larger model first.
        api_model: Model name for the Gemini API.
        api_key: API key override (otherwise resolved from env vars).
        num_gpus: Number of GPUs for parallel fallback (default: 2).
        rpm_limit: Maximum requests per minute for API mode (default: 14).
    """
    random.seed(seed)
    tc_cfg = ToolCallingConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    gemma_gen: GemmaDataGenerator | None = None
    using_api = False
    if use_gemma:
        if use_api:
            gemma_gen, using_api = _create_generator(
                use_api=True,
                api_model=api_model,
                api_key=api_key,
                local_model=gemma_model,
                device=device,
                torch_dtype=torch_dtype,
                rpm_limit=rpm_limit,
            )
            # If API failed at startup, delegate to parallel generation
            if not using_api:
                gemma_gen.cleanup()
                logger.info("Delegating to parallel generation with %d GPUs", num_gpus)
                generate_synthetic_data_parallel(
                    output_dir=output_dir,
                    num_train=num_train,
                    num_val=num_val,
                    seed=seed,
                    gemma_model=gemma_model,
                    torch_dtype=torch_dtype,
                    queries_per_category=queries_per_category,
                    num_gpus=num_gpus,
                )
                return
        else:
            gemma_gen = GemmaDataGenerator(
                model_name=gemma_model,
                device=device,
                torch_dtype=torch_dtype,
            )
        gemma_gen.warm_up_query_cache(queries_per_category=queries_per_category)
    else:
        logger.info("Using template-based generation (--no-gemma)")

    # Build weighted scenario list
    scenarios_weighted: list[str] = []
    for scenario in _SCENARIOS:
        count = int(scenario["weight"] * 100)
        scenarios_weighted.extend([scenario["category"]] * count)

    # Batch size for API mode: process rpm_limit examples concurrently
    api_batch_size = rpm_limit if using_api else 1

    for split, num_examples in [("train", num_train), ("val", num_val)]:
        path = out / f"tool_calling_{split}.jsonl"
        generated = 0
        with open(path, "w") as f:
            while generated < num_examples:
                batch_count = min(api_batch_size, num_examples - generated)
                categories = [random.choice(scenarios_weighted) for _ in range(batch_count)]

                if gemma_gen is not None and using_api and batch_count > 1:
                    # Batched concurrent API generation
                    try:
                        examples = gemma_gen.generate_examples_batch(
                            categories, tc_cfg, batch_size=batch_count,
                        )
                    except Exception as exc:
                        if not isinstance(gemma_gen, GeminiApiGenerator) or not _is_quota_error(exc):
                            raise
                        logger.warning(
                            "Gemini API quota exhausted after %d/%d %s examples, "
                            "finishing with %d-GPU parallel",
                            generated, num_examples, split, num_gpus,
                        )
                        gemma_gen.cleanup()
                        f.flush()
                        remaining_this_split = num_examples - generated
                        _finish_with_parallel(
                            output_dir=output_dir,
                            path=path,
                            split=split,
                            remaining=remaining_this_split,
                            remaining_val=num_val if split == "train" else 0,
                            num_val=num_val,
                            seed=seed,
                            gemma_model=gemma_model,
                            torch_dtype=torch_dtype,
                            queries_per_category=queries_per_category,
                            num_gpus=num_gpus,
                        )
                        return
                    for example in examples:
                        f.write(json.dumps(example) + "\n")
                        generated += 1
                elif gemma_gen is not None:
                    # Single example (local GPU or batch_count == 1)
                    try:
                        example = gemma_gen.generate_example(categories[0], tc_cfg)
                    except Exception as exc:
                        if not isinstance(gemma_gen, GeminiApiGenerator) or not _is_quota_error(exc):
                            raise
                        logger.warning(
                            "Gemini API quota exhausted after %d/%d %s examples, "
                            "finishing with %d-GPU parallel",
                            generated, num_examples, split, num_gpus,
                        )
                        gemma_gen.cleanup()
                        f.flush()
                        remaining_this_split = num_examples - generated
                        _finish_with_parallel(
                            output_dir=output_dir,
                            path=path,
                            split=split,
                            remaining=remaining_this_split,
                            remaining_val=num_val if split == "train" else 0,
                            num_val=num_val,
                            seed=seed,
                            gemma_model=gemma_model,
                            torch_dtype=torch_dtype,
                            queries_per_category=queries_per_category,
                            num_gpus=num_gpus,
                        )
                        return
                    f.write(json.dumps(example) + "\n")
                    generated += 1
                else:
                    # Template mode
                    for cat in categories:
                        generator = _GENERATORS[cat]
                        example = generator(tc_cfg)
                        f.write(json.dumps(example) + "\n")
                        generated += 1

                if generated % 100 == 0 or generated == num_examples:
                    logger.info("[%s] Generated %d/%d examples", split, generated, num_examples)

        logger.info("Generated %d %s examples → %s", num_examples, split, path)

    # Free resources
    if gemma_gen is not None:
        gemma_gen.cleanup()

    print(f"Synthetic data generated: {num_train} train, {num_val} val in {out}/")


def _finish_with_parallel(
    *,
    output_dir: str,
    path: Path,
    split: str,
    remaining: int,
    remaining_val: int,
    num_val: int,
    seed: int,
    gemma_model: str,
    torch_dtype: str,
    queries_per_category: int,
    num_gpus: int,
) -> None:
    """
    Finish remaining examples using multi-GPU parallel generation.

    Generates remaining examples for the current split into a temp file,
    then appends to the existing partial file. If the current split is
    'train', also generates the full val split.
    """
    out = Path(output_dir)
    temp_path = out / f"_tmp_remaining_{split}.jsonl"

    # Generate remaining examples for current split
    generate_synthetic_data_parallel(
        output_dir=output_dir,
        num_train=remaining if split == "train" else 0,
        num_val=remaining if split == "val" else num_val,
        seed=seed + 1000,  # Different seed to avoid duplicating already-generated
        gemma_model=gemma_model,
        torch_dtype=torch_dtype,
        queries_per_category=queries_per_category,
        num_gpus=num_gpus,
    )

    # The parallel function wrote to the standard output paths.
    # For the current split, we need to merge: existing partial + new parallel output.
    parallel_path = out / f"tool_calling_{split}.jsonl"
    if parallel_path.exists() and path != parallel_path:
        # Append parallel output to partial file
        with open(path, "a") as f_out:
            with open(parallel_path) as f_in:
                for line in f_in:
                    f_out.write(line)

    logger.info("Finished remaining %d %s examples with %d-GPU parallel", remaining, split, num_gpus)
    print(f"Synthetic data generated with API+parallel fallback in {out}/")


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def build_dataloader(
    tokenizer: PreTrainedTokenizer,
    cfg: TrainingConfig,
    max_seq_len: int = 512,
    split: str = "train",
) -> DataLoader:
    """Build the appropriate DataLoader for a training phase."""
    if cfg.phase in ("tcs-pretrain", "tcs-lora"):
        max_samples = None
        if split == "validation":
            max_samples = cfg.max_eval_samples

        dataset = TextDataset(
            tokenizer=tokenizer,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            split=split,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
        )
    elif cfg.phase == "tool-calling":
        data_file = f"data/tool_calling_{split}.jsonl"
        dataset = ToolCallingDataset(
            tokenizer=tokenizer,
            data_path=data_file,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(f"Unknown phase: {cfg.phase}")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for synthetic data generation."""
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--num-train", type=int, default=25000, help="Number of training examples")
    parser.add_argument("--num-val", type=int, default=1500, help="Number of validation examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Gemma generation options (default: enabled)
    parser.add_argument(
        "--no-gemma", action="store_true",
        help="Disable Gemma-based generation, use templates instead (faster, no GPU)",
    )
    parser.add_argument(
        "--gemma-model", default="google/gemma-4-e2b-it",
        help="HuggingFace model ID for Gemma data generation",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for Gemma inference (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--torch-dtype", default="float16", choices=["bfloat16", "float16", "float32"],
        help="Dtype for Gemma model",
    )
    parser.add_argument(
        "--queries-per-category", type=int, default=30,
        help="Number of user queries to pre-generate per scenario category",
    )

    # Parallel generation options
    parser.add_argument(
        "--parallel", action="store_true",
        help="Enable parallel generation across multiple GPUs (default: single GPU)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=2,
        help="Number of GPUs for parallel generation (default: 2)",
    )

    # Gemini API options (larger model via API, fallback to local GPU)
    parser.add_argument(
        "--gemini-api", action="store_true",
        help="Use Gemini API for a larger model. Falls back to local GPU if credits run out.",
    )
    parser.add_argument(
        "--api-model", default="gemma-4-31b-it",
        help="Model name for the Gemini API (default: gemma-4-31b-it)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for Gemini (default: from GEMINI_API_KEY / GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--rpm", type=int, default=14,
        help="Max requests per minute for API mode (default: 14, limit is 15)",
    )

    args = parser.parse_args()

    if args.gemini_api:
        # API mode: start with Gemini API, fallback to local multi-GPU
        generate_synthetic_data(
            output_dir=args.output_dir,
            num_train=args.num_train,
            num_val=args.num_val,
            seed=args.seed,
            use_gemma=True,
            gemma_model=args.gemma_model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            queries_per_category=args.queries_per_category,
            use_api=True,
            api_model=args.api_model,
            api_key=args.api_key,
            num_gpus=args.num_gpus,
            rpm_limit=args.rpm,
        )
    elif args.parallel and not args.no_gemma:
        generate_synthetic_data_parallel(
            output_dir=args.output_dir,
            num_train=args.num_train,
            num_val=args.num_val,
            seed=args.seed,
            gemma_model=args.gemma_model,
            torch_dtype=args.torch_dtype,
            queries_per_category=args.queries_per_category,
            num_gpus=args.num_gpus,
        )
    else:
        generate_synthetic_data(
            output_dir=args.output_dir,
            num_train=args.num_train,
            num_val=args.num_val,
            seed=args.seed,
            use_gemma=not args.no_gemma,
            gemma_model=args.gemma_model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            queries_per_category=args.queries_per_category,
            use_api=False,
            api_model=args.api_model,
            api_key=args.api_key,
        )


if __name__ == "__main__":  # pragma: no cover
    main()


