"""
Cactus-compatible intelligent task router for the Memory Companion.

Routes incoming tasks to the optimal model/compression level based on
input complexity, task type, and device constraints. This is the core
of the "local-first mobile wearable that intelligently routes between models."

Routing decisions:
  - Face detected → MobileFaceNet (1MB ONNX) → sqlite-vec lookup → LLM with context
  - Simple greeting/orientation → Gemma 4 with TCS r=8 (fastest, 64× FLOP savings)
  - Medication query → SQLite direct lookup (skip LLM entirely for speed)
  - Complex reasoning/memory → Gemma 4 with TCS r=2 (highest quality)
  - Caregiver alert → rule-based trigger (no LLM needed)
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch

from src.config import (
    DatabaseConfig,
    FaceEmbeddingConfig,
    InferenceConfig,
    ModelConfig,
    ToolCallingConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Classification
# ---------------------------------------------------------------------------

class TaskType(Enum):
    """Categories of tasks the companion can handle."""
    FACE_RECOGNITION = auto()   # Camera detected a face → identify person
    ORIENTATION = auto()        # "Where am I?" / scene description
    MEDICATION = auto()         # Medication schedule queries
    MEMORY_RECALL = auto()      # "Tell me about X" / detailed memory retrieval
    GREETING = auto()           # Simple greetings and small talk
    CAREGIVER_ALERT = auto()    # Confusion/wandering/emergency detection
    GENERAL = auto()            # Anything else


class ModelRoute(Enum):
    """Where to route the task for processing."""
    FACE_PIPELINE = auto()      # MobileFaceNet ONNX → sqlite-vec
    LLM_FAST = auto()           # Gemma 4 + TCS r=8 (fastest)
    LLM_BALANCED = auto()       # Gemma 4 + TCS r=4 (default)
    LLM_QUALITY = auto()        # Gemma 4 + TCS r=2 (highest quality)
    SQLITE_DIRECT = auto()      # Direct DB query, no LLM needed
    RULE_BASED = auto()         # Hardcoded logic (alerts, simple responses)


@dataclass
class RoutingDecision:
    """Result of the task router's classification."""
    task_type: TaskType
    route: ModelRoute
    compression_ratio: int | None = None  # Only for LLM routes
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_budget_ms: int = 2000  # Max time allowed for this task


# ---------------------------------------------------------------------------
# Keyword patterns for fast classification (no LLM needed)
# ---------------------------------------------------------------------------

# Portuguese + English patterns (the patient may speak either)
_GREETING_PATTERNS = re.compile(
    r"\b(oi|olá|bom dia|boa tarde|boa noite|hello|hi|hey|good morning"
    r"|good afternoon|good evening|como vai|tudo bem)\b",
    re.IGNORECASE,
)

_MEDICATION_PATTERNS = re.compile(
    r"\b(remédio|medicamento|medicação|medication|medicine|pill|pills"
    r"|comprimido|tomar|dose|dosage|horário|schedule"
    r"|manhã|morning|tarde|afternoon|noite|evening|night"
    r"|agenda|compromisso|appointment|rotina|routine"
    r"|o que (eu )?faço|what do i do|minha agenda|my schedule)\b",
    re.IGNORECASE,
)

_ORIENTATION_PATTERNS = re.compile(
    r"\b(onde estou|where am i|que lugar|what place|que horas|what time"
    r"|que dia|what day|que cômodo|what room|cozinha|kitchen"
    r"|quarto|bedroom|banheiro|bathroom|sala|living room)\b",
    re.IGNORECASE,
)

_ALERT_PATTERNS = re.compile(
    r"\b(perdido|lost|confuso|confused|não sei|don't know|scared|medo"
    r"|help|socorro|ajuda|cadê|where is|caí|fell|fall|dor|pain|hurt)\b",
    re.IGNORECASE,
)

_MEMORY_PATTERNS = re.compile(
    r"\b(quem é|who is|me conta|tell me about|lembra|remember"
    r"|história|story|quando|when did|como era|what was"
    r"|família|family|filho|son|filha|daughter|neto|grandson"
    r"|neta|granddaughter|esposa|wife|marido|husband)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Task Router
# ---------------------------------------------------------------------------

class TaskRouter:
    """
    Intelligent router that classifies incoming tasks and routes them to
    the optimal model/compression level.

    The router uses a lightweight keyword-based classifier (no ML overhead)
    to avoid adding latency. For ambiguous inputs, it defaults to the
    balanced LLM route (r=4).

    This is the Cactus integration point: a local-first wearable that
    routes between MobileFaceNet, SQLite, rule-based logic, and the LLM
    at different compression levels.
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        inf_cfg: InferenceConfig | None = None,
    ):
        self.model_cfg = model_cfg or ModelConfig()
        self.inf_cfg = inf_cfg or InferenceConfig()

        # Available compression ratios (from multi-scale config)
        self.ratios = self.model_cfg.compression.ratios if self.model_cfg.compression.multi_scale else [
            self.model_cfg.compression.ratio,
        ]

        # Routing stats for logging
        self._stats: dict[str, int] = {rt.name: 0 for rt in ModelRoute}

    def classify(
        self,
        user_input: str,
        has_face: bool = False,
        face_embedding: list[float] | None = None,
    ) -> RoutingDecision:
        """
        Classify the input and return a routing decision.

        Args:
            user_input: Text from the patient (speech-to-text or typed).
            has_face: Whether the camera detected a face in the current frame.
            face_embedding: 128-d MobileFaceNet embedding if a face was detected.

        Returns:
            RoutingDecision with task type, route, and compression ratio.
        """
        # Priority 1: Face detected → always route to face pipeline first
        if has_face and face_embedding is not None:
            decision = RoutingDecision(
                task_type=TaskType.FACE_RECOGNITION,
                route=ModelRoute.FACE_PIPELINE,
                compression_ratio=4,  # After face lookup, use balanced LLM
                confidence=0.95,
                metadata={"face_embedding_dim": len(face_embedding)},
                latency_budget_ms=500,  # Face lookup should be fast
            )
            self._log_route(decision)
            return decision

        # Priority 2: Caregiver alerts (safety-critical, rule-based)
        if _ALERT_PATTERNS.search(user_input):
            decision = RoutingDecision(
                task_type=TaskType.CAREGIVER_ALERT,
                route=ModelRoute.RULE_BASED,
                confidence=0.85,
                latency_budget_ms=100,  # Alerts must be instant
            )
            self._log_route(decision)
            return decision

        # Priority 3: Medication (direct DB lookup, fast)
        if _MEDICATION_PATTERNS.search(user_input):
            decision = RoutingDecision(
                task_type=TaskType.MEDICATION,
                route=ModelRoute.SQLITE_DIRECT,
                compression_ratio=8,  # Fallback to fast LLM if DB needs formatting
                confidence=0.9,
                latency_budget_ms=300,
            )
            self._log_route(decision)
            return decision

        # Priority 4: Simple greetings (fast LLM, high compression)
        if _GREETING_PATTERNS.search(user_input):
            decision = RoutingDecision(
                task_type=TaskType.GREETING,
                route=ModelRoute.LLM_FAST,
                compression_ratio=8,
                confidence=0.9,
                latency_budget_ms=500,
            )
            self._log_route(decision)
            return decision

        # Priority 5: Orientation questions (balanced)
        if _ORIENTATION_PATTERNS.search(user_input):
            decision = RoutingDecision(
                task_type=TaskType.ORIENTATION,
                route=ModelRoute.LLM_BALANCED,
                compression_ratio=4,
                confidence=0.85,
                latency_budget_ms=1000,
            )
            self._log_route(decision)
            return decision

        # Priority 6: Memory recall (quality-first, low compression)
        if _MEMORY_PATTERNS.search(user_input):
            decision = RoutingDecision(
                task_type=TaskType.MEMORY_RECALL,
                route=ModelRoute.LLM_QUALITY,
                compression_ratio=2,
                confidence=0.85,
                latency_budget_ms=2000,
            )
            self._log_route(decision)
            return decision

        # Default: balanced LLM
        decision = RoutingDecision(
            task_type=TaskType.GENERAL,
            route=ModelRoute.LLM_BALANCED,
            compression_ratio=4,
            confidence=0.5,
            latency_budget_ms=1500,
        )
        self._log_route(decision)
        return decision

    def _log_route(self, decision: RoutingDecision) -> None:
        self._stats[decision.route.name] += 1
        logger.debug(
            "Routed to %s (task=%s, ratio=%s, confidence=%.2f)",
            decision.route.name,
            decision.task_type.name,
            decision.compression_ratio,
            decision.confidence,
        )

    def get_stats(self) -> dict[str, int]:
        """Return routing statistics for monitoring."""
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Routed Inference Orchestrator
# ---------------------------------------------------------------------------

class RoutedCompanion:
    """
    Full inference pipeline with intelligent routing.

    Combines:
    - TaskRouter: classifies input → decides which model/compression
    - MobileFaceNet: face detection + embedding (ONNX)
    - Gemma4WithTCS: LLM at variable compression ratios
    - SQLite: direct DB queries for medication/person lookups
    - Rule-based: caregiver alerts without LLM

    This is the Cactus entry point.
    """

    def __init__(
        self,
        companion,          # MemoryCompanion instance (lazy import to avoid circular)
        router: TaskRouter,
        db_conn: sqlite3.Connection,
    ):
        self.companion = companion
        self.router = router
        self.db = db_conn

    def respond(
        self,
        user_input: str,
        has_face: bool = False,
        face_embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Process an input through the routing pipeline.

        Returns:
            Dict with: response (str), route (str), task_type (str),
            latency_ms (float), compression_ratio (int | None).
        """
        start = time.perf_counter()

        # Step 1: Route the task
        decision = self.router.classify(user_input, has_face, face_embedding)

        # Step 2: Execute based on route
        if decision.route == ModelRoute.FACE_PIPELINE:
            response = self._handle_face(user_input, face_embedding, decision)
        elif decision.route == ModelRoute.SQLITE_DIRECT:
            response = self._handle_sqlite_direct(user_input, decision)
        elif decision.route == ModelRoute.RULE_BASED:
            response = self._handle_rule_based(user_input, decision)
        else:
            # LLM routes (fast / balanced / quality)
            response = self._handle_llm(user_input, decision)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "response": response,
            "route": decision.route.name,
            "task_type": decision.task_type.name,
            "compression_ratio": decision.compression_ratio,
            "latency_ms": round(elapsed_ms, 1),
            "within_budget": elapsed_ms <= decision.latency_budget_ms,
        }

    def _handle_face(
        self,
        user_input: str,
        face_embedding: list[float] | None,
        decision: RoutingDecision,
    ) -> str:
        """Face detected → look up person → enrich prompt → LLM response."""
        if face_embedding is None:
            return self._handle_llm(user_input, decision)

        # Search for matching face in DB
        from src.tools.sqlite_tools import tool_read_person
        person = self._search_face_embedding(face_embedding)

        if person and person.get("found", False):
            # Enrich the prompt with person context
            enriched = (
                f"[Face recognized: {person.get('name', 'unknown')}, "
                f"{person.get('relationship', '')}] "
                f"{user_input}"
            )
            return self.companion.respond(enriched)

        # Face not recognized → still answer with LLM
        enriched = f"[Unknown face detected] {user_input}"
        return self.companion.respond(enriched)

    def _search_face_embedding(self, embedding: list[float]) -> dict[str, Any] | None:
        """Search sqlite-vec for matching face embedding."""
        try:
            import struct
            blob = struct.pack(f"{len(embedding)}f", *embedding)
            row = self.db.execute(
                """
                SELECT p.id, p.name, p.relationship, p.bio,
                       vec_distance_cosine(fe.embedding, ?) as distance
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.id
                WHERE distance < 0.15
                ORDER BY distance ASC
                LIMIT 1
                """,
                (blob,),
            ).fetchone()
            if row:
                return {
                    "found": True,
                    "id": row["id"],
                    "name": row["name"],
                    "relationship": row["relationship"],
                    "bio": row["bio"],
                    "distance": row["distance"],
                }
        except Exception as e:
            logger.warning("Face search failed: %s", e)
        return {"found": False}

    def _handle_sqlite_direct(
        self,
        user_input: str,
        decision: RoutingDecision,
    ) -> str:
        """Direct DB lookup for medication schedule — skip LLM for speed."""
        # Detect time of day from input
        input_lower = user_input.lower()
        time_of_day = "morning"  # default

        for tod in ("morning", "manhã", "afternoon", "tarde", "evening", "noite", "night"):
            if tod in input_lower:
                time_of_day = {
                    "manhã": "morning",
                    "tarde": "afternoon",
                    "noite": "evening",
                }.get(tod, tod)
                break

        # Direct query
        from src.tools.sqlite_tools import tool_get_medication
        result = tool_get_medication(self.db, time_of_day)

        if result.get("medications"):
            meds = result["medications"]
            lines = []
            for med in meds:
                lines.append(
                    f"- {med['medication_name']}: {med.get('dosage', '')} "
                    f"({med.get('notes', '')})"
                )
            return (
                f"Here are your {time_of_day} medications:\n"
                + "\n".join(lines)
            )

        # No medications found → fall back to LLM for a helpful response
        return self.companion.respond(user_input)

    def _handle_rule_based(
        self,
        user_input: str,
        decision: RoutingDecision,
    ) -> str:
        """
        Safety-critical alerts: rule-based, no LLM latency.
        Immediately triggers caregiver alert AND provides calming response.
        """
        # Determine alert type
        input_lower = user_input.lower()
        if any(w in input_lower for w in ("perdido", "lost", "cadê", "where")):
            alert_type = "wandering"
            calm = (
                "You're safe. You're at home. Take a deep breath. "
                "I'm letting your family know you need some company."
            )
        elif any(w in input_lower for w in ("caí", "fell", "fall", "dor", "pain", "hurt")):
            alert_type = "fall_risk"
            calm = (
                "Stay still and don't try to get up alone. "
                "I'm calling for help right now."
            )
        else:
            alert_type = "confusion"
            calm = (
                "It's okay to feel confused sometimes. You're safe. "
                "Let me help you. I'm letting your family know."
            )

        # Fire alert (non-blocking)
        try:
            from src.tools.sqlite_tools import tool_alert_caregiver
            tool_alert_caregiver(self.db, alert_type, user_input[:200])
        except Exception as e:
            logger.error("Failed to send caregiver alert: %s", e)

        return calm

    def _handle_llm(
        self,
        user_input: str,
        decision: RoutingDecision,
    ) -> str:
        """Route to LLM with the appropriate compression ratio."""
        # Override the companion's default compression ratio
        original_ratio = self.companion.inf_cfg.compression_ratio
        if decision.compression_ratio:
            self.companion.inf_cfg.compression_ratio = decision.compression_ratio

        try:
            return self.companion.respond(user_input)
        finally:
            self.companion.inf_cfg.compression_ratio = original_ratio
