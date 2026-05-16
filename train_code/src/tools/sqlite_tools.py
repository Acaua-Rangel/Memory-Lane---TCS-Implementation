"""
SQLite tool definitions and execution engine.

Tools allow the model to read/write patient data (persons, memories,
medication schedules, encounters) via structured function calls.
Face embeddings use sqlite-vec for vector similarity search.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DatabaseConfig, ToolCallingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS persons (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    relationship TEXT,
    bio TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id TEXT PRIMARY KEY,
    person_id TEXT REFERENCES persons(id),
    embedding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    person_id TEXT REFERENCES persons(id),
    content TEXT NOT NULL,
    memory_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS medication_schedule (
    id TEXT PRIMARY KEY,
    medication_name TEXT NOT NULL,
    description TEXT,
    time_of_day TEXT,
    dosage TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS encounters (
    id TEXT PRIMARY KEY,
    person_id TEXT REFERENCES persons(id),
    context TEXT,
    location TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS locations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    features TEXT
);

CREATE TABLE IF NOT EXISTS agenda (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    day_of_week TEXT,
    time TEXT,
    recurring INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS preferences (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category, key)
);

CREATE TABLE IF NOT EXISTS routines (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    time_of_day TEXT,
    steps TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize the SQLite database with schema and WAL mode."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()

    logger.info("Database initialized at %s", db_path)
    return conn


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

def tool_read_person(conn: sqlite3.Connection, face_id: str) -> dict[str, Any]:
    """Look up a person by ID and return their info + memories."""
    row = conn.execute(
        "SELECT * FROM persons WHERE id = ?", (face_id,)
    ).fetchone()

    if row is None:
        return {"found": False, "message": "No person found with that ID."}

    memories = conn.execute(
        "SELECT content, memory_type FROM memories WHERE person_id = ? ORDER BY created_at DESC LIMIT 5",
        (face_id,),
    ).fetchall()

    return {
        "found": True,
        "name": row["name"],
        "relationship": row["relationship"],
        "bio": row["bio"],
        "memories": [{"content": m["content"], "type": m["memory_type"]} for m in memories],
    }


def tool_write_encounter(
    conn: sqlite3.Connection, person_name: str, context: str
) -> dict[str, Any]:
    """Log an encounter with a person."""
    # Find person by name
    person = conn.execute(
        "SELECT id FROM persons WHERE name = ? COLLATE NOCASE", (person_name,)
    ).fetchone()

    person_id = person["id"] if person else None
    encounter_id = str(uuid.uuid4())

    conn.execute(
        "INSERT INTO encounters (id, person_id, context, timestamp) VALUES (?, ?, ?, ?)",
        (encounter_id, person_id, context, datetime.now().isoformat()),
    )
    conn.commit()

    return {"success": True, "encounter_id": encounter_id}


def tool_get_medication(
    conn: sqlite3.Connection, time_of_day: str
) -> dict[str, Any]:
    """Get medication schedule for a time of day."""
    meds = conn.execute(
        "SELECT medication_name, description, dosage, notes FROM medication_schedule WHERE time_of_day = ?",
        (time_of_day,),
    ).fetchall()

    if not meds:
        return {"medications": [], "message": f"No medications scheduled for {time_of_day}."}

    return {
        "medications": [
            {
                "name": m["medication_name"],
                "description": m["description"],
                "dosage": m["dosage"],
                "notes": m["notes"],
            }
            for m in meds
        ]
    }


def tool_describe_location(
    conn: sqlite3.Connection, room_features: str
) -> dict[str, Any]:
    """Identify location based on visual features."""
    features = [f.strip().lower() for f in room_features.split(",")]

    # Simple keyword matching against stored locations
    locations = conn.execute("SELECT * FROM locations").fetchall()

    best_match = None
    best_score = 0

    for loc in locations:
        stored_features = set(f.strip().lower() for f in (loc["features"] or "").split(","))
        score = len(set(features) & stored_features)
        if score > best_score:
            best_score = score
            best_match = loc

    if best_match and best_score > 0:
        return {
            "found": True,
            "name": best_match["name"],
            "description": best_match["description"],
        }

    return {"found": False, "message": "Location not recognized from those features."}


def tool_alert_caregiver(
    conn: sqlite3.Connection, alert_type: str, details: str
) -> dict[str, Any]:
    """Send alert to caregiver (logs to encounters table)."""
    alert_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO encounters (id, context, location, timestamp) VALUES (?, ?, ?, ?)",
        (alert_id, f"ALERT [{alert_type}]: {details}", "system", datetime.now().isoformat()),
    )
    conn.commit()

    logger.warning("Caregiver alert [%s]: %s", alert_type, details)
    return {"success": True, "alert_id": alert_id, "type": alert_type}


def tool_get_agenda(
    conn: sqlite3.Connection, day_of_week: str | None = None
) -> dict[str, Any]:
    """Get the patient's agenda/schedule for a given day or all days."""
    if day_of_week:
        rows = conn.execute(
            "SELECT title, description, time, day_of_week, recurring "
            "FROM agenda WHERE day_of_week = ? COLLATE NOCASE ORDER BY time",
            (day_of_week,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT title, description, time, day_of_week, recurring "
            "FROM agenda ORDER BY day_of_week, time"
        ).fetchall()

    if not rows:
        msg = f"No agenda items for {day_of_week}." if day_of_week else "No agenda items."
        return {"items": [], "message": msg}

    return {
        "items": [
            {
                "title": r["title"],
                "description": r["description"],
                "time": r["time"],
                "day": r["day_of_week"],
                "recurring": bool(r["recurring"]),
            }
            for r in rows
        ]
    }


def tool_save_preference(
    conn: sqlite3.Connection, category: str, key: str, value: str
) -> dict[str, Any]:
    """Save or update a patient preference (food, music, habits, etc.)."""
    pref_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO preferences (id, category, key, value) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(category, key) DO UPDATE SET value = excluded.value",
        (pref_id, category, key, value),
    )
    conn.commit()
    return {"success": True, "category": category, "key": key}


def tool_get_preferences(
    conn: sqlite3.Connection, category: str | None = None
) -> dict[str, Any]:
    """Get patient preferences, optionally filtered by category."""
    if category:
        rows = conn.execute(
            "SELECT category, key, value FROM preferences WHERE category = ? COLLATE NOCASE",
            (category,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT category, key, value FROM preferences ORDER BY category"
        ).fetchall()

    return {
        "preferences": [
            {"category": r["category"], "key": r["key"], "value": r["value"]}
            for r in rows
        ]
    }


def tool_get_routine(
    conn: sqlite3.Connection, time_of_day: str
) -> dict[str, Any]:
    """Get the patient's daily routine for a time of day."""
    rows = conn.execute(
        "SELECT name, steps, notes FROM routines WHERE time_of_day = ? COLLATE NOCASE",
        (time_of_day,),
    ).fetchall()

    if not rows:
        return {"routines": [], "message": f"No routines for {time_of_day}."}

    return {
        "routines": [
            {"name": r["name"], "steps": r["steps"], "notes": r["notes"]}
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# Tool Execution Engine
# ---------------------------------------------------------------------------

def tool_get_current_datetime(
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Return the current date, time, day of week, and time-of-day period."""
    now = datetime.now()
    hour = now.hour
    if hour < 12:
        period = "morning"
    elif hour < 18:
        period = "afternoon"
    elif hour < 21:
        period = "evening"
    else:
        period = "night"

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "day_of_week": now.strftime("%A").lower(),
        "time_of_day": period,
    }


_TOOL_REGISTRY: dict[str, callable] = {
    "read_person": tool_read_person,
    "write_encounter": tool_write_encounter,
    "get_medication": tool_get_medication,
    "describe_location": tool_describe_location,
    "alert_caregiver": tool_alert_caregiver,
    "get_agenda": tool_get_agenda,
    "save_preference": tool_save_preference,
    "get_preferences": tool_get_preferences,
    "get_routine": tool_get_routine,
    "get_current_datetime": tool_get_current_datetime,
}


def execute_tool_call(
    conn: sqlite3.Connection, tool_name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """
    Execute a tool call against the SQLite database.

    Args:
        conn: SQLite connection.
        tool_name: Name of the tool to execute.
        arguments: Tool arguments as a dict.

    Returns:
        Tool execution result as a dict.

    Raises:
        ValueError: If tool_name is not in the registry.
    """
    if tool_name not in _TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool: {tool_name}. Available: {list(_TOOL_REGISTRY.keys())}"
        )

    tool_fn = _TOOL_REGISTRY[tool_name]
    return tool_fn(conn, **arguments)


def parse_tool_call(text: str, tc_cfg: ToolCallingConfig) -> tuple[str, dict] | None:
    """
    Parse a tool call from model output text.

    Returns (tool_name, arguments) or None if no valid tool call found.
    """
    start = text.find(tc_cfg.tool_call_start)
    end = text.find(tc_cfg.tool_call_end)

    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start + len(tc_cfg.tool_call_start):end].strip()

    try:
        call = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool call JSON: %s", json_str)
        return None

    name = call.get("name")
    args = call.get("arguments", {})

    if not name or not isinstance(args, dict):
        return None

    return name, args


def format_tool_response(
    tool_name: str, result: dict[str, Any], tc_cfg: ToolCallingConfig
) -> str:
    """Format a tool execution result for feeding back to the model."""
    response = {"name": tool_name, "result": result}
    return f"{tc_cfg.tool_response_start}\n{json.dumps(response)}\n{tc_cfg.tool_response_end}"


# ---------------------------------------------------------------------------
# Seed Data (for demo/testing)
# ---------------------------------------------------------------------------

def seed_demo_data(conn: sqlite3.Connection) -> None:
    """Insert demo data for testing and hackathon demo."""
    persons = [
        ("face_001", "Maria", "granddaughter", "Maria is 8 years old. She loves drawing and visits every Sunday. You used to take her fishing at the lake."),
        ("face_002", "Rosa", "caregiver", "Rosa has been your caregiver for 3 years. She is very kind and makes your favorite soup."),
        ("face_003", "João", "son", "João is your oldest son. He is an engineer and lives in São Paulo. He calls every Wednesday."),
        ("face_004", "Ana", "wife", "Ana is your wife of 45 years. She loved gardening and cooking. Her favorite flower was sunflower."),
    ]

    for pid, name, rel, bio in persons:
        conn.execute(
            "INSERT OR IGNORE INTO persons (id, name, relationship, bio) VALUES (?, ?, ?, ?)",
            (pid, name, rel, bio),
        )

    memories = [
        ("face_001", "You taught Maria to ride a bicycle last summer in the park.", "shared_experience"),
        ("face_001", "Maria's favorite color is purple. She always wears purple ribbons.", "preference"),
        ("face_003", "João graduated from engineering school in 2010. You were very proud.", "shared_experience"),
        ("face_004", "You and Ana got married on June 15, 1980, at the small church downtown.", "shared_experience"),
        ("face_004", "Ana's chocolate cake recipe is in the blue notebook in the kitchen drawer.", "routine"),
    ]

    for person_id, content, mtype in memories:
        conn.execute(
            "INSERT OR IGNORE INTO memories (id, person_id, content, memory_type) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), person_id, content, mtype),
        )

    medications = [
        ("Donepezil", "small white round pill", "morning", "5mg", "Take with breakfast"),
        ("Memantine", "oval beige pill", "morning", "10mg", "Take with breakfast"),
        ("Vitamin D", "yellow soft capsule", "afternoon", "1000 IU", "Take with lunch"),
        ("Melatonin", "small blue pill", "night", "3mg", "Take 30 minutes before bed"),
    ]

    for name, desc, tod, dosage, notes in medications:
        conn.execute(
            "INSERT OR IGNORE INTO medication_schedule (id, medication_name, description, time_of_day, dosage, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), name, desc, tod, dosage, notes),
        )

    locations = [
        ("Kitchen", "Your kitchen where you have breakfast", "stove,refrigerator,table,sink,microwave"),
        ("Living Room", "The main room where you watch TV", "sofa,television,bookshelf,window,lamp"),
        ("Bedroom", "Your bedroom where you sleep", "bed,wardrobe,mirror,nightstand,clock"),
        ("Garden", "The small garden where Ana planted flowers", "plants,bench,fence,sunflowers,path"),
        ("Bathroom", "The main bathroom next to your bedroom", "sink,mirror,towels,shower,tiles"),
    ]

    for name, desc, features in locations:
        conn.execute(
            "INSERT OR IGNORE INTO locations (id, name, description, features) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), name, desc, features),
        )

    # Agenda
    agenda_items = [
        ("Doctor appointment", "Checkup with Dr. Silva at the clinic", "wednesday", "10:00", 0),
        ("Maria visits", "Granddaughter Maria comes to visit", "sunday", "15:00", 1),
        ("João calls", "Son João calls from São Paulo", "wednesday", "19:00", 1),
        ("Physical therapy", "Session with therapist at home", "monday", "09:00", 1),
        ("Physical therapy", "Session with therapist at home", "thursday", "09:00", 1),
    ]

    for title, desc, day, time, recurring in agenda_items:
        conn.execute(
            "INSERT OR IGNORE INTO agenda (id, title, description, day_of_week, time, recurring) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), title, desc, day, time, recurring),
        )

    # Preferences
    preferences = [
        ("food", "favorite_meal", "Feijoada with rice and farofa"),
        ("food", "morning_drink", "Coffee with milk, no sugar"),
        ("food", "dislike", "Does not like fish"),
        ("music", "favorite_genre", "Bossa nova, especially Tom Jobim"),
        ("music", "favorite_song", "Garota de Ipanema"),
        ("hobby", "favorite_activity", "Watching football on TV, especially Bahia games"),
        ("hobby", "past_hobby", "Used to fish at the lake every weekend"),
        ("social", "best_friend", "Carlos, neighbor for 30 years, visits on Saturdays"),
    ]

    for category, key, value in preferences:
        conn.execute(
            "INSERT OR IGNORE INTO preferences (id, category, key, value) "
            "VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), category, key, value),
        )

    # Daily Routines
    routines = [
        ("Morning routine", "morning",
         "1. Wake up\n2. Go to bathroom\n3. Wash face and brush teeth\n4. Take morning medication with breakfast\n5. Have coffee with milk",
         "Rosa (caregiver) arrives at 7am to help"),
        ("Afternoon routine", "afternoon",
         "1. Lunch at 12pm\n2. Take afternoon vitamin\n3. Rest/nap until 2pm\n4. Watch TV or sit in garden",
         "Likes to sit in Ana's garden after lunch"),
        ("Evening routine", "evening",
         "1. Dinner at 6pm\n2. Watch the news\n3. Call family if scheduled\n4. Take night medication",
         "Prefers light dinner — soup or sandwich"),
        ("Night routine", "night",
         "1. Brush teeth\n2. Take melatonin\n3. Put on pajamas\n4. Go to bed by 9pm",
         "Sleeps better with the hallway light on"),
    ]

    for name, tod, steps, notes in routines:
        conn.execute(
            "INSERT OR IGNORE INTO routines (id, name, time_of_day, steps, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), name, tod, steps, notes),
        )

    conn.commit()
    logger.info("Demo data seeded into database.")
