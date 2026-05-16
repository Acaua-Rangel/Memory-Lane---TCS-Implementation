"""
Tests for SQLite tools and tool execution engine.
"""

import json
import sqlite3
import pytest

from src.config import ToolCallingConfig
from src.tools.sqlite_tools import (
    execute_tool_call,
    format_tool_response,
    init_database,
    parse_tool_call,
    seed_demo_data,
    tool_read_person,
    tool_get_medication,
    tool_describe_location,
    tool_write_encounter,
    tool_alert_caregiver,
)


@pytest.fixture
def db():
    """In-memory SQLite database with demo data."""
    conn = init_database(":memory:")
    seed_demo_data(conn)
    yield conn
    conn.close()


@pytest.fixture
def clean_db():
    """In-memory SQLite database without demo data."""
    conn = init_database(":memory:")
    yield conn
    conn.close()


class TestDatabaseInit:
    def test_tables_created(self, db):
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "persons" in table_names
        assert "memories" in table_names
        assert "medication_schedule" in table_names
        assert "encounters" in table_names
        assert "locations" in table_names


class TestReadPerson:
    def test_existing_person(self, db):
        result = tool_read_person(db, "face_001")
        assert result["found"] is True
        assert result["name"] == "Maria"
        assert result["relationship"] == "granddaughter"

    def test_nonexistent_person(self, db):
        result = tool_read_person(db, "face_999")
        assert result["found"] is False

    def test_person_with_memories(self, db):
        result = tool_read_person(db, "face_001")
        assert len(result["memories"]) > 0


class TestGetMedication:
    def test_morning_meds(self, db):
        result = tool_get_medication(db, "morning")
        assert len(result["medications"]) == 2
        names = {m["name"] for m in result["medications"]}
        assert "Donepezil" in names

    def test_evening_no_meds(self, db):
        result = tool_get_medication(db, "evening")
        assert len(result["medications"]) == 0

    def test_night_meds(self, db):
        result = tool_get_medication(db, "night")
        assert len(result["medications"]) == 1
        assert result["medications"][0]["name"] == "Melatonin"


class TestDescribeLocation:
    def test_kitchen(self, db):
        result = tool_describe_location(db, "stove,refrigerator,table")
        assert result["found"] is True
        assert result["name"] == "Kitchen"

    def test_partial_match(self, db):
        result = tool_describe_location(db, "sofa,television")
        assert result["found"] is True
        assert result["name"] == "Living Room"

    def test_no_match(self, db):
        result = tool_describe_location(db, "pool,diving_board")
        assert result["found"] is False


class TestWriteEncounter:
    def test_write_known_person(self, db):
        result = tool_write_encounter(db, "Maria", "Visited for Sunday lunch")
        assert result["success"] is True
        assert "encounter_id" in result

    def test_write_unknown_person(self, db):
        result = tool_write_encounter(db, "Unknown Person", "Arrived at door")
        assert result["success"] is True


class TestAlertCaregiver:
    def test_alert(self, db):
        result = tool_alert_caregiver(db, "confusion", "Patient seems disoriented")
        assert result["success"] is True
        assert result["type"] == "confusion"


class TestGetCurrentDatetime:
    def test_returns_all_fields(self, db):
        from src.tools.sqlite_tools import tool_get_current_datetime
        result = tool_get_current_datetime(db)
        assert "date" in result
        assert "time" in result
        assert "day_of_week" in result
        assert result["time_of_day"] in ("morning", "afternoon", "evening", "night")

    def test_dispatch_via_engine(self, db):
        result = execute_tool_call(db, "get_current_datetime", {})
        assert "date" in result
        assert "time_of_day" in result


class TestExecuteToolCall:
    def test_dispatch(self, db):
        result = execute_tool_call(db, "read_person", {"face_id": "face_001"})
        assert result["name"] == "Maria"

    def test_unknown_tool(self, db):
        with pytest.raises(ValueError, match="Unknown tool"):
            execute_tool_call(db, "nonexistent_tool", {})


class TestParseToolCall:
    def test_valid_call(self):
        tc_cfg = ToolCallingConfig()
        text = '<tool_call>\n{"name": "read_person", "arguments": {"face_id": "face_001"}}\n</tool_call>'
        result = parse_tool_call(text, tc_cfg)
        assert result is not None
        assert result[0] == "read_person"
        assert result[1]["face_id"] == "face_001"

    def test_no_tool_call(self):
        tc_cfg = ToolCallingConfig()
        result = parse_tool_call("Just a normal response.", tc_cfg)
        assert result is None

    def test_malformed_json(self):
        tc_cfg = ToolCallingConfig()
        text = "<tool_call>\nnot valid json\n</tool_call>"
        result = parse_tool_call(text, tc_cfg)
        assert result is None

    def test_missing_name_field(self):
        """Tool call JSON without a 'name' field should return None."""
        tc_cfg = ToolCallingConfig()
        text = '<tool_call>\n{"arguments": {"face_id": "face_001"}}\n</tool_call>'
        result = parse_tool_call(text, tc_cfg)
        assert result is None

    def test_arguments_not_dict(self):
        """Tool call with arguments as non-dict should return None."""
        tc_cfg = ToolCallingConfig()
        text = '<tool_call>\n{"name": "read_person", "arguments": "invalid"}\n</tool_call>'
        result = parse_tool_call(text, tc_cfg)
        assert result is None


class TestFormatToolResponse:
    def test_format(self):
        tc_cfg = ToolCallingConfig()
        result = format_tool_response(
            "read_person",
            {"name": "Maria", "relationship": "granddaughter"},
            tc_cfg,
        )
        assert "<tool_response>" in result
        assert "</tool_response>" in result
        parsed = json.loads(
            result.replace("<tool_response>\n", "").replace("\n</tool_response>", "")
        )
        assert parsed["name"] == "read_person"
        assert parsed["result"]["name"] == "Maria"


# ---------------------------------------------------------------------------
# Agenda, preferences, routine tools
# ---------------------------------------------------------------------------

class TestGetAgenda:
    def test_agenda_no_data(self, clean_db):
        from src.tools.sqlite_tools import tool_get_agenda
        result = tool_get_agenda(clean_db, "Monday")
        assert result["items"] == []
        assert "Monday" in result["message"]

    def test_agenda_all_days_no_data(self, clean_db):
        from src.tools.sqlite_tools import tool_get_agenda
        result = tool_get_agenda(clean_db)
        assert result["items"] == []

    def test_agenda_with_data(self, clean_db):
        from src.tools.sqlite_tools import tool_get_agenda
        clean_db.execute(
            "INSERT INTO agenda (id, title, description, time, day_of_week, recurring) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("a1", "Doctor visit", "Check-up", "10:00", "Monday", 0),
        )
        clean_db.commit()
        result = tool_get_agenda(clean_db, "Monday")
        assert len(result["items"]) == 1
        assert result["items"][0]["title"] == "Doctor visit"

    def test_agenda_all_days(self, db):
        from src.tools.sqlite_tools import tool_get_agenda
        db.execute(
            "INSERT INTO agenda (id, title, description, time, day_of_week, recurring) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("a2", "Walk", "Morning walk", "07:00", "Tuesday", 1),
        )
        db.commit()
        result = tool_get_agenda(db)
        assert len(result["items"]) >= 1


class TestSaveAndGetPreferences:
    def test_save_preference(self, db):
        from src.tools.sqlite_tools import tool_save_preference
        result = tool_save_preference(db, "food", "favorite", "pasta")
        assert result["success"] is True
        assert result["category"] == "food"
        assert result["key"] == "favorite"

    def test_get_preferences_by_category(self, clean_db):
        from src.tools.sqlite_tools import tool_save_preference, tool_get_preferences
        tool_save_preference(clean_db, "music", "genre", "classical")
        result = tool_get_preferences(clean_db, "music")
        assert len(result["preferences"]) == 1
        assert result["preferences"][0]["value"] == "classical"

    def test_get_all_preferences(self, db):
        from src.tools.sqlite_tools import tool_save_preference, tool_get_preferences
        tool_save_preference(db, "food", "drink", "coffee")
        tool_save_preference(db, "music", "artist", "Bach")
        result = tool_get_preferences(db)
        assert len(result["preferences"]) >= 2

    def test_save_preference_upsert(self, clean_db):
        from src.tools.sqlite_tools import tool_save_preference, tool_get_preferences
        tool_save_preference(clean_db, "food", "fav", "pasta")
        tool_save_preference(clean_db, "food", "fav", "rice")
        result = tool_get_preferences(clean_db, "food")
        assert len(result["preferences"]) == 1
        assert result["preferences"][0]["value"] == "rice"


class TestGetRoutine:
    def test_no_routines(self, clean_db):
        from src.tools.sqlite_tools import tool_get_routine
        result = tool_get_routine(clean_db, "morning")
        assert result["routines"] == []

    def test_with_routine_data(self, clean_db):
        from src.tools.sqlite_tools import tool_get_routine
        clean_db.execute(
            "INSERT INTO routines (id, name, time_of_day, steps, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            ("r1", "Wake up", "morning", "Stretch, brush teeth", "Do before breakfast"),
        )
        clean_db.commit()
        result = tool_get_routine(clean_db, "morning")
        assert len(result["routines"]) == 1
        assert result["routines"][0]["name"] == "Wake up"


class TestExecuteToolCallDispatch:
    def test_dispatch_get_agenda(self, db):
        result = execute_tool_call(db, "get_agenda", {"day_of_week": "Monday"})
        assert "items" in result

    def test_dispatch_save_preference(self, db):
        result = execute_tool_call(db, "save_preference", {"category": "food", "key": "fav", "value": "rice"})
        assert result["success"] is True

    def test_dispatch_get_preferences(self, db):
        result = execute_tool_call(db, "get_preferences", {})
        assert "preferences" in result

    def test_dispatch_get_routine(self, db):
        result = execute_tool_call(db, "get_routine", {"time_of_day": "morning"})
        assert "routines" in result

    def test_dispatch_get_current_datetime(self, db):
        result = execute_tool_call(db, "get_current_datetime", {})
        assert "date" in result
