"""Tests for the Cactus-compatible task router."""

import pytest
import sqlite3
from unittest.mock import MagicMock, patch

from src.routing import (
    ModelRoute,
    RoutedCompanion,
    RoutingDecision,
    TaskRouter,
    TaskType,
)
from src.config import InferenceConfig


@pytest.fixture
def router():
    return TaskRouter()


class TestTaskClassification:
    """Test that inputs are classified to the correct task type and route."""

    def test_face_detected_routes_to_face_pipeline(self, router):
        decision = router.classify("Who is this?", has_face=True, face_embedding=[0.1] * 128)
        assert decision.task_type == TaskType.FACE_RECOGNITION
        assert decision.route == ModelRoute.FACE_PIPELINE

    def test_face_without_embedding_still_routes_face(self, router):
        """has_face=True but no embedding → should NOT route to face pipeline."""
        decision = router.classify("Who is this?", has_face=True, face_embedding=None)
        # Without embedding, it should fall through to text classification
        assert decision.route != ModelRoute.FACE_PIPELINE

    def test_greeting_routes_to_fast_llm(self, router):
        for greeting in ["Oi!", "Bom dia", "Hello", "Hey", "Tudo bem?"]:
            decision = router.classify(greeting)
            assert decision.task_type == TaskType.GREETING
            assert decision.route == ModelRoute.LLM_FAST
            assert decision.compression_ratio == 8

    def test_medication_routes_to_sqlite(self, router):
        for med_query in [
            "Que remédio eu tomo de manhã?",
            "What medication do I take?",
            "Qual o horário do comprimido?",
        ]:
            decision = router.classify(med_query)
            assert decision.task_type == TaskType.MEDICATION
            assert decision.route == ModelRoute.SQLITE_DIRECT

    def test_orientation_routes_to_balanced_llm(self, router):
        for orient in ["Onde estou?", "Where am I?", "Que lugar é esse?"]:
            decision = router.classify(orient)
            assert decision.task_type == TaskType.ORIENTATION
            assert decision.route == ModelRoute.LLM_BALANCED
            assert decision.compression_ratio == 4

    def test_memory_recall_routes_to_quality_llm(self, router):
        for memory_q in ["Quem é a Maria?", "Tell me about my family", "Who is João?"]:
            decision = router.classify(memory_q)
            assert decision.task_type == TaskType.MEMORY_RECALL
            assert decision.route == ModelRoute.LLM_QUALITY
            assert decision.compression_ratio == 2

    def test_alert_routes_to_rule_based(self, router):
        for alert in ["Estou perdido", "I'm lost", "Me ajuda", "I fell down"]:
            decision = router.classify(alert)
            assert decision.task_type == TaskType.CAREGIVER_ALERT
            assert decision.route == ModelRoute.RULE_BASED

    def test_unknown_input_routes_to_balanced(self, router):
        decision = router.classify("xyzzy foobarbaz")
        assert decision.task_type == TaskType.GENERAL
        assert decision.route == ModelRoute.LLM_BALANCED
        assert decision.compression_ratio == 4

    def test_alert_priority_over_greeting(self, router):
        """Alert keywords should take priority over greeting patterns."""
        decision = router.classify("Oi, estou perdido")
        assert decision.task_type == TaskType.CAREGIVER_ALERT

    def test_face_priority_over_everything(self, router):
        """Face detection always takes highest priority."""
        decision = router.classify("Estou perdido", has_face=True, face_embedding=[0.1] * 128)
        assert decision.task_type == TaskType.FACE_RECOGNITION


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_default_latency_budget(self):
        d = RoutingDecision(task_type=TaskType.GENERAL, route=ModelRoute.LLM_BALANCED)
        assert d.latency_budget_ms == 2000

    def test_compression_ratio_none_for_non_llm(self, router):
        decision = router.classify("Estou perdido")
        assert decision.route == ModelRoute.RULE_BASED
        assert decision.compression_ratio is None


class TestRouterStats:
    """Test routing statistics tracking."""

    def test_stats_accumulate(self, router):
        router.classify("Bom dia")
        router.classify("Bom dia")
        router.classify("Onde estou?")
        stats = router.get_stats()
        assert stats["LLM_FAST"] == 2
        assert stats["LLM_BALANCED"] == 1

    def test_stats_start_at_zero(self, router):
        stats = router.get_stats()
        assert all(v == 0 for v in stats.values())


# ---------------------------------------------------------------------------
# RoutedCompanion
# ---------------------------------------------------------------------------

class TestRoutedCompanion:
    """Test the full routing pipeline with mocked companion and DB."""

    @pytest.fixture
    def routed(self):
        companion = MagicMock()
        companion.respond = MagicMock(return_value="LLM response")
        companion.inf_cfg = InferenceConfig()
        router = TaskRouter()
        db = MagicMock(spec=sqlite3.Connection)
        return RoutedCompanion(companion, router, db)

    def test_respond_greeting_goes_to_llm(self, routed):
        result = routed.respond("Bom dia!")
        assert result["route"] == "LLM_FAST"
        assert result["compression_ratio"] == 8
        assert "response" in result
        assert "latency_ms" in result
        assert "within_budget" in result

    def test_respond_alert_rule_based(self, routed):
        result = routed.respond("Estou perdido e com medo")
        assert result["route"] == "RULE_BASED"
        assert "response" in result
        # Rule-based should return calming text, not from LLM
        assert "safe" in result["response"].lower() or "okay" in result["response"].lower()

    def test_respond_medication_sqlite_direct(self, routed):
        with patch("src.tools.sqlite_tools.tool_get_medication", return_value={
            "medications": [{"medication_name": "Donepezil", "dosage": "5mg", "notes": "With food"}]
        }):
            result = routed.respond("Que remédio eu tomo de manhã?")
        assert result["route"] == "SQLITE_DIRECT"

    def test_respond_medication_no_meds_fallback(self, routed):
        with patch("src.tools.sqlite_tools.tool_get_medication", return_value={"medications": []}):
            result = routed.respond("Que remédio eu tomo de manhã?")
        assert result["route"] == "SQLITE_DIRECT"
        # Falls back to LLM response
        assert result["response"] == "LLM response"

    def test_respond_face_pipeline(self, routed):
        embedding = [0.1] * 128

        # Mock _search_face_embedding
        with patch.object(routed, "_search_face_embedding", return_value={
            "found": True, "name": "Maria", "relationship": "granddaughter"
        }):
            result = routed.respond("Who is that?", has_face=True, face_embedding=embedding)
        assert result["route"] == "FACE_PIPELINE"

    def test_respond_face_no_embedding(self, routed):
        result = routed.respond("Who is that?", has_face=True, face_embedding=None)
        # Without embedding, falls through to text classification
        assert result["route"] != "FACE_PIPELINE"

    def test_respond_face_not_recognized(self, routed):
        embedding = [0.1] * 128
        with patch.object(routed, "_search_face_embedding", return_value={"found": False}):
            result = routed.respond("Who is that?", has_face=True, face_embedding=embedding)
        assert result["route"] == "FACE_PIPELINE"

    def test_handle_rule_based_wandering(self, routed):
        response = routed._handle_rule_based(
            "I'm lost and can't find my room",
            RoutingDecision(task_type=TaskType.CAREGIVER_ALERT, route=ModelRoute.RULE_BASED),
        )
        assert "safe" in response.lower()

    def test_handle_rule_based_fall(self, routed):
        response = routed._handle_rule_based(
            "I fell down and it hurts",
            RoutingDecision(task_type=TaskType.CAREGIVER_ALERT, route=ModelRoute.RULE_BASED),
        )
        assert "still" in response.lower() or "help" in response.lower()

    def test_handle_rule_based_confusion(self, routed):
        response = routed._handle_rule_based(
            "I'm very confused today",
            RoutingDecision(task_type=TaskType.CAREGIVER_ALERT, route=ModelRoute.RULE_BASED),
        )
        assert "okay" in response.lower() or "safe" in response.lower()

    def test_handle_rule_based_alert_failure(self, routed):
        """Alert tool failure should not prevent calming response."""
        with patch("src.tools.sqlite_tools.tool_alert_caregiver", side_effect=Exception("DB error")):
            response = routed._handle_rule_based(
                "I'm confused",
                RoutingDecision(task_type=TaskType.CAREGIVER_ALERT, route=ModelRoute.RULE_BASED),
            )
        assert isinstance(response, str)

    def test_handle_llm_restores_ratio(self, routed):
        original_ratio = routed.companion.inf_cfg.compression_ratio
        decision = RoutingDecision(
            task_type=TaskType.ORIENTATION,
            route=ModelRoute.LLM_BALANCED,
            compression_ratio=8,
        )
        routed._handle_llm("Where am I?", decision)
        # Should restore original ratio
        assert routed.companion.inf_cfg.compression_ratio == original_ratio

    def test_handle_sqlite_time_detection(self, routed):
        """Test time-of-day detection from user input."""
        for tod_word, expected_route in [
            ("morning", "SQLITE_DIRECT"),
            ("manhã", "SQLITE_DIRECT"),
            ("afternoon", "SQLITE_DIRECT"),
            ("noite", "SQLITE_DIRECT"),
        ]:
            with patch("src.tools.sqlite_tools.tool_get_medication", return_value={
                "medications": [{"medication_name": "Test", "dosage": "1mg", "notes": ""}]
            }):
                result = routed.respond(f"What medication do I take in the {tod_word}?")
            assert result["route"] == "SQLITE_DIRECT"

    def test_search_face_embedding_failure(self, routed):
        """DB error during face search returns not found."""
        routed.db.execute = MagicMock(side_effect=Exception("DB error"))
        result = routed._search_face_embedding([0.1] * 128)
        assert result["found"] is False
