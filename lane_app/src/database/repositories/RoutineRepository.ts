import type { SQLiteDatabase } from 'expo-sqlite';
import type { RoutineStep } from '../../types';
import { uuid } from '../utils';

export class RoutineRepository {
  constructor(private db: SQLiteDatabase) {}

  async findByTimeOfDay(timeOfDay: RoutineStep['timeOfDay']): Promise<RoutineStep[]> {
    const rows = await this.db.getAllAsync<RawRoutine>(
      'SELECT * FROM routine_steps WHERE time_of_day = ? ORDER BY step_order',
      [timeOfDay],
    );
    return rows.map(toRecord);
  }

  async findAll(): Promise<RoutineStep[]> {
    const rows = await this.db.getAllAsync<RawRoutine>(
      'SELECT * FROM routine_steps ORDER BY time_of_day, step_order',
    );
    return rows.map(toRecord);
  }

  async insert(data: Omit<RoutineStep, 'id'>): Promise<RoutineStep> {
    const id = uuid();
    await this.db.runAsync(
      `INSERT INTO routine_steps (id, time_of_day, step_order, title, description)
       VALUES (?, ?, ?, ?, ?)`,
      [id, data.timeOfDay, data.order, data.title, data.description],
    );
    return { id, ...data };
  }

  async upsert(data: RoutineStep): Promise<void> {
    await this.db.runAsync(
      `INSERT INTO routine_steps (id, time_of_day, step_order, title, description)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         time_of_day = excluded.time_of_day,
         step_order = excluded.step_order,
         title = excluded.title,
         description = excluded.description`,
      [data.id, data.timeOfDay, data.order, data.title, data.description],
    );
  }

  async delete(id: string): Promise<void> {
    await this.db.runAsync('DELETE FROM routine_steps WHERE id = ?', [id]);
  }
}

interface RawRoutine {
  id: string;
  time_of_day: string;
  step_order: number;
  title: string;
  description: string;
}

function toRecord(raw: RawRoutine): RoutineStep {
  return {
    id: raw.id,
    timeOfDay: raw.time_of_day as RoutineStep['timeOfDay'],
    order: raw.step_order,
    title: raw.title,
    description: raw.description,
  };
}
