import type { SQLiteDatabase } from 'expo-sqlite';
import type { EncounterRecord } from '../../types';
import { uuid } from '../utils';

export class EncounterRepository {
  constructor(private db: SQLiteDatabase) {}

  async findByPersonId(personId: string, limit = 5): Promise<EncounterRecord[]> {
    const rows = await this.db.getAllAsync<RawEncounter>(
      'SELECT * FROM encounters WHERE person_id = ? ORDER BY timestamp DESC LIMIT ?',
      [personId, limit],
    );
    return rows.map(toRecord);
  }

  async findLast(personId: string): Promise<EncounterRecord | null> {
    const row = await this.db.getFirstAsync<RawEncounter>(
      'SELECT * FROM encounters WHERE person_id = ? ORDER BY timestamp DESC LIMIT 1',
      [personId],
    );
    return row ? toRecord(row) : null;
  }

  async findRecent(hours = 24): Promise<EncounterRecord[]> {
    const since = Date.now() - hours * 60 * 60 * 1000;
    const rows = await this.db.getAllAsync<RawEncounter>(
      'SELECT * FROM encounters WHERE timestamp >= ? ORDER BY timestamp DESC',
      [since],
    );
    return rows.map(toRecord);
  }

  async insert(data: Omit<EncounterRecord, 'id' | 'timestamp'>): Promise<EncounterRecord> {
    const id = uuid();
    const timestamp = Date.now();
    await this.db.runAsync(
      `INSERT INTO encounters (id, person_id, context, location, notes, timestamp)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [id, data.personId, data.context, data.location ?? null, data.notes ?? null, timestamp],
    );
    return { id, ...data, timestamp };
  }
}

interface RawEncounter {
  id: string;
  person_id: string;
  context: string;
  location: string | null;
  notes: string | null;
  timestamp: number;
}

function toRecord(raw: RawEncounter): EncounterRecord {
  return {
    id: raw.id,
    personId: raw.person_id,
    context: raw.context,
    location: raw.location ?? undefined,
    notes: raw.notes ?? undefined,
    timestamp: raw.timestamp,
  };
}
