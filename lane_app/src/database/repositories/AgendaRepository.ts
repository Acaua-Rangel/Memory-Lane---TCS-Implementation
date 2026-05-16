import type { SQLiteDatabase } from 'expo-sqlite';
import type { AgendaRecord } from '../../types';
import { uuid } from '../utils';

export class AgendaRepository {
  constructor(private db: SQLiteDatabase) {}

  async findByDayOfWeek(dayOfWeek: number): Promise<AgendaRecord[]> {
    const rows = await this.db.getAllAsync<RawAgenda>(
      'SELECT * FROM agenda WHERE day_of_week = ? ORDER BY time',
      [dayOfWeek],
    );
    return rows.map(toRecord);
  }

  async findToday(): Promise<AgendaRecord[]> {
    const today = new Date().getDay();
    return this.findByDayOfWeek(today);
  }

  async findAll(): Promise<AgendaRecord[]> {
    const rows = await this.db.getAllAsync<RawAgenda>(
      'SELECT * FROM agenda ORDER BY day_of_week, time',
    );
    return rows.map(toRecord);
  }

  async insert(data: Omit<AgendaRecord, 'id'>): Promise<AgendaRecord> {
    const id = uuid();
    await this.db.runAsync(
      `INSERT INTO agenda (id, title, description, day_of_week, time, location, is_recurring)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [id, data.title, data.description, data.dayOfWeek, data.time ?? null, data.location ?? null, data.isRecurring ? 1 : 0],
    );
    return { id, ...data };
  }

  async upsert(data: AgendaRecord): Promise<void> {
    await this.db.runAsync(
      `INSERT INTO agenda (id, title, description, day_of_week, time, location, is_recurring)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         title = excluded.title,
         description = excluded.description,
         day_of_week = excluded.day_of_week,
         time = excluded.time,
         location = excluded.location,
         is_recurring = excluded.is_recurring`,
      [data.id, data.title, data.description, data.dayOfWeek, data.time ?? null, data.location ?? null, data.isRecurring ? 1 : 0],
    );
  }

  async delete(id: string): Promise<void> {
    await this.db.runAsync('DELETE FROM agenda WHERE id = ?', [id]);
  }
}

interface RawAgenda {
  id: string;
  title: string;
  description: string;
  day_of_week: number;
  time: string | null;
  location: string | null;
  is_recurring: number;
}

function toRecord(raw: RawAgenda): AgendaRecord {
  return {
    id: raw.id,
    title: raw.title,
    description: raw.description,
    dayOfWeek: raw.day_of_week,
    time: raw.time ?? undefined,
    location: raw.location ?? undefined,
    isRecurring: raw.is_recurring === 1,
  };
}
