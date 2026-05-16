import type { SQLiteDatabase } from 'expo-sqlite';
import type { PersonRecord } from '../../types';
import { uuid } from '../utils';

export class PersonRepository {
  constructor(private db: SQLiteDatabase) {}

  async findById(id: string): Promise<PersonRecord | null> {
    const row = await this.db.getFirstAsync<RawPerson>(
      'SELECT * FROM persons WHERE id = ?',
      [id],
    );
    return row ? toRecord(row) : null;
  }

  async findAll(): Promise<PersonRecord[]> {
    const rows = await this.db.getAllAsync<RawPerson>('SELECT * FROM persons ORDER BY name');
    return rows.map(toRecord);
  }

  async findCaregivers(): Promise<PersonRecord[]> {
    const rows = await this.db.getAllAsync<RawPerson>(
      'SELECT * FROM persons WHERE is_caregiver = 1 ORDER BY name',
    );
    return rows.map(toRecord);
  }

  async findByName(name: string): Promise<PersonRecord | null> {
    const row = await this.db.getFirstAsync<RawPerson>(
      'SELECT * FROM persons WHERE name LIKE ? LIMIT 1',
      [`%${name}%`],
    );
    return row ? toRecord(row) : null;
  }

  async insert(data: Omit<PersonRecord, 'id' | 'createdAt' | 'updatedAt'>): Promise<PersonRecord> {
    const now = Date.now();
    const id = uuid();
    await this.db.runAsync(
      `INSERT INTO persons (id, name, relationship, bio, phone_number, is_caregiver, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [id, data.name, data.relationship, data.bio, data.phoneNumber ?? null, data.isCaregiver ? 1 : 0, now, now],
    );
    return { id, ...data, createdAt: now, updatedAt: now };
  }

  async upsert(data: Omit<PersonRecord, 'createdAt' | 'updatedAt'>): Promise<void> {
    const now = Date.now();
    await this.db.runAsync(
      `INSERT INTO persons (id, name, relationship, bio, phone_number, is_caregiver, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         name = excluded.name,
         relationship = excluded.relationship,
         bio = excluded.bio,
         phone_number = excluded.phone_number,
         is_caregiver = excluded.is_caregiver,
         updated_at = excluded.updated_at`,
      [data.id, data.name, data.relationship, data.bio, data.phoneNumber ?? null, data.isCaregiver ? 1 : 0, now, now],
    );
  }

  async update(id: string, data: Partial<Omit<PersonRecord, 'id' | 'createdAt'>>): Promise<void> {
    const now = Date.now();
    await this.db.runAsync(
      `UPDATE persons SET
         name = COALESCE(?, name),
         relationship = COALESCE(?, relationship),
         bio = COALESCE(?, bio),
         phone_number = COALESCE(?, phone_number),
         is_caregiver = COALESCE(?, is_caregiver),
         updated_at = ?
       WHERE id = ?`,
      [
        data.name ?? null,
        data.relationship ?? null,
        data.bio ?? null,
        data.phoneNumber ?? null,
        data.isCaregiver !== undefined ? (data.isCaregiver ? 1 : 0) : null,
        now,
        id,
      ],
    );
  }

  async delete(id: string): Promise<void> {
    await this.db.runAsync('DELETE FROM persons WHERE id = ?', [id]);
  }

  async count(): Promise<number> {
    const row = await this.db.getFirstAsync<{ count: number }>('SELECT COUNT(*) as count FROM persons');
    return row?.count ?? 0;
  }
}

interface RawPerson {
  id: string;
  name: string;
  relationship: string;
  bio: string;
  phone_number: string | null;
  is_caregiver: number;
  created_at: number;
  updated_at: number;
}

function toRecord(raw: RawPerson): PersonRecord {
  return {
    id: raw.id,
    name: raw.name,
    relationship: raw.relationship,
    bio: raw.bio,
    phoneNumber: raw.phone_number ?? undefined,
    isCaregiver: raw.is_caregiver === 1,
    createdAt: raw.created_at,
    updatedAt: raw.updated_at,
  };
}
