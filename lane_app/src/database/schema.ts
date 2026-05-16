// SQL DDL para as 8 tabelas — espelha DatabaseConfig e sample data do projeto Python

export const SCHEMA_SQL = `
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS persons (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  relationship TEXT NOT NULL,
  bio TEXT NOT NULL,
  phone_number TEXT,
  is_caregiver INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
  id TEXT PRIMARY KEY,
  person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_person ON face_embeddings(person_id);

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  date TEXT,
  tags TEXT,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_person ON memories(person_id);

CREATE TABLE IF NOT EXISTS medication_schedule (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  dosage TEXT NOT NULL,
  time_of_day TEXT NOT NULL CHECK(time_of_day IN ('morning','afternoon','evening','night')),
  description TEXT NOT NULL,
  instructions TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS encounters (
  id TEXT PRIMARY KEY,
  person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  context TEXT NOT NULL,
  location TEXT,
  notes TEXT,
  timestamp INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_encounters_person ON encounters(person_id);
CREATE INDEX IF NOT EXISTS idx_encounters_time ON encounters(timestamp);

CREATE TABLE IF NOT EXISTS locations (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  features TEXT NOT NULL,
  navigation_hint TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agenda (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  day_of_week INTEGER NOT NULL CHECK(day_of_week BETWEEN 0 AND 6),
  time TEXT,
  location TEXT,
  is_recurring INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS preferences (
  id TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(category, key)
);

CREATE TABLE IF NOT EXISTS routine_steps (
  id TEXT PRIMARY KEY,
  time_of_day TEXT NOT NULL CHECK(time_of_day IN ('morning','afternoon','evening','night')),
  step_order INTEGER NOT NULL,
  title TEXT NOT NULL,
  description TEXT NOT NULL
);
`;
