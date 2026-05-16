import type { SQLiteDatabase } from 'expo-sqlite';
import type { MemoryRecord } from '../../types';
import { uuid } from '../utils';

export class MemoryRepository {
  constructor(private db: SQLiteDatabase) {}

  async findByPersonId(personId: string, limit = 10): Promise<MemoryRecord[]> {
    const rows = await this.db.getAllAsync<RawMemory>(
      'SELECT * FROM memories WHERE person_id = ? ORDER BY created_at DESC LIMIT ?',
      [personId, limit],
    );
    return rows.map(toRecord);
  }

  async findById(id: string): Promise<MemoryRecord | null> {
    const row = await this.db.getFirstAsync<RawMemory>(
      'SELECT * FROM memories WHERE id = ?',
      [id],
    );
    return row ? toRecord(row) : null;
  }

  async insert(data: Omit<MemoryRecord, 'id' | 'createdAt'>): Promise<MemoryRecord> {
    const id = uuid();
    const now = Date.now();
    const tagsJson = data.tags ? JSON.stringify(data.tags) : null;
    await this.db.runAsync(
      `INSERT INTO memories (id, person_id, title, description, date, tags, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [id, data.personId, data.title, data.description, data.date ?? null, tagsJson, now],
    );
    return { id, ...data, createdAt: now };
  }

  async delete(id: string): Promise<void> {
    await this.db.runAsync('DELETE FROM memories WHERE id = ?', [id]);
  }
}

interface RawMemory {
  id: string;
  person_id: string;
  title: string;
  description: string;
  date: string | null;
  tags: string | null;
  created_at: number;
}

function toRecord(raw: RawMemory): MemoryRecord {
  return {
    id: raw.id,
    personId: raw.person_id,
    title: raw.title,
    description: raw.description,
    date: raw.date ?? undefined,
    tags: raw.tags ? (JSON.parse(raw.tags) as string[]) : undefined,
    createdAt: raw.created_at,
  };
}
