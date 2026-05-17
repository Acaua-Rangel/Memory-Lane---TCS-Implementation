import type { SQLiteDatabase } from 'expo-sqlite';
import type { MedicationRecord } from '../../types';
import { uuid } from '../utils';

export class MedicationRepository {
  constructor(private db: SQLiteDatabase) {}

  async findByTimeOfDay(timeOfDay: MedicationRecord['timeOfDay']): Promise<MedicationRecord[]> {
    const rows = await this.db.getAllAsync<RawMedication>(
      'SELECT * FROM medication_schedule WHERE time_of_day = ? AND is_active = 1',
      [timeOfDay],
    );
    return rows.map(toRecord);
  }

  async findAll(activeOnly = true): Promise<MedicationRecord[]> {
    const sql = activeOnly
      ? 'SELECT * FROM medication_schedule WHERE is_active = 1 ORDER BY time_of_day'
      : 'SELECT * FROM medication_schedule ORDER BY time_of_day';
    const rows = await this.db.getAllAsync<RawMedication>(sql);
    return rows.map(toRecord);
  }

  async findById(id: string): Promise<MedicationRecord | null> {
    const row = await this.db.getFirstAsync<RawMedication>(
      'SELECT * FROM medication_schedule WHERE id = ?',
      [id],
    );
    return row ? toRecord(row) : null;
  }

  async insert(data: Omit<MedicationRecord, 'id'>): Promise<MedicationRecord> {
    const id = uuid();
    await this.db.runAsync(
      `INSERT INTO medication_schedule (id, name, dosage, time_of_day, description, instructions, is_active)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [id, data.name, data.dosage, data.timeOfDay, data.description, data.instructions, data.isActive ? 1 : 0],
    );
    return { id, ...data };
  }

  async upsert(data: MedicationRecord): Promise<void> {
    await this.db.runAsync(
      `INSERT INTO medication_schedule (id, name, dosage, time_of_day, description, instructions, is_active)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         name = excluded.name,
         dosage = excluded.dosage,
         time_of_day = excluded.time_of_day,
         description = excluded.description,
         instructions = excluded.instructions,
         is_active = excluded.is_active`,
      [data.id, data.name, data.dosage, data.timeOfDay, data.description, data.instructions, data.isActive ? 1 : 0],
    );
  }

  async setActive(id: string, isActive: boolean): Promise<void> {
    await this.db.runAsync(
      'UPDATE medication_schedule SET is_active = ? WHERE id = ?',
      [isActive ? 1 : 0, id],
    );
  }

  async delete(id: string): Promise<void> {
    await this.db.runAsync('DELETE FROM medication_schedule WHERE id = ?', [id]);
  }
}

interface RawMedication {
  id: string;
  name: string;
  dosage: string;
  time_of_day: string;
  description: string;
  instructions: string;
  is_active: number;
}

function toRecord(raw: RawMedication): MedicationRecord {
  return {
    id: raw.id,
    name: raw.name,
    dosage: raw.dosage,
    timeOfDay: raw.time_of_day as MedicationRecord['timeOfDay'],
    description: raw.description,
    instructions: raw.instructions,
    isActive: raw.is_active === 1,
  };
}
