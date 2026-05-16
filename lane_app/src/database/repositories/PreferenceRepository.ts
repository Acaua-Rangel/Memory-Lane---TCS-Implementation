import type { SQLiteDatabase } from 'expo-sqlite';
import type { PreferenceRecord } from '../../types';
import { uuid } from '../utils';

export class PreferenceRepository {
  constructor(private db: SQLiteDatabase) {}

  async findByCategory(category: string): Promise<PreferenceRecord[]> {
    const rows = await this.db.getAllAsync<RawPreference>(
      'SELECT * FROM preferences WHERE category = ? ORDER BY key',
      [category],
    );
    return rows.map(toRecord);
  }

  async findOne(category: string, key: string): Promise<PreferenceRecord | null> {
    const row = await this.db.getFirstAsync<RawPreference>(
      'SELECT * FROM preferences WHERE category = ? AND key = ?',
      [category, key],
    );
    return row ? toRecord(row) : null;
  }

  async getValue(category: string, key: string): Promise<string | null> {
    const pref = await this.findOne(category, key);
    return pref?.value ?? null;
  }

  async set(category: string, key: string, value: string): Promise<PreferenceRecord> {
    const now = Date.now();
    const existing = await this.findOne(category, key);

    if (existing) {
      await this.db.runAsync(
        'UPDATE preferences SET value = ?, updated_at = ? WHERE category = ? AND key = ?',
        [value, now, category, key],
      );
      return { ...existing, value, updatedAt: now };
    }

    const id = uuid();
    await this.db.runAsync(
      'INSERT INTO preferences (id, category, key, value, updated_at) VALUES (?, ?, ?, ?, ?)',
      [id, category, key, value, now],
    );
    return { id, category, key, value, updatedAt: now };
  }

  async delete(category: string, key: string): Promise<void> {
    await this.db.runAsync(
      'DELETE FROM preferences WHERE category = ? AND key = ?',
      [category, key],
    );
  }
}

interface RawPreference {
  id: string;
  category: string;
  key: string;
  value: string;
  updated_at: number;
}

function toRecord(raw: RawPreference): PreferenceRecord {
  return {
    id: raw.id,
    category: raw.category,
    key: raw.key,
    value: raw.value,
    updatedAt: raw.updated_at,
  };
}
