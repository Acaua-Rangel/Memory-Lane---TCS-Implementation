import type { SQLiteDatabase } from 'expo-sqlite';
import type { LocationRecord } from '../../types';
import { uuid } from '../utils';

export class LocationRepository {
  constructor(private db: SQLiteDatabase) {}

  async findAll(): Promise<LocationRecord[]> {
    const rows = await this.db.getAllAsync<RawLocation>('SELECT * FROM locations ORDER BY name');
    return rows.map(toRecord);
  }

  async findById(id: string): Promise<LocationRecord | null> {
    const row = await this.db.getFirstAsync<RawLocation>(
      'SELECT * FROM locations WHERE id = ?',
      [id],
    );
    return row ? toRecord(row) : null;
  }

  // Identifica localização por features visuais detectadas na cena
  async findByFeatures(detectedFeatures: string[]): Promise<LocationRecord | null> {
    const locations = await this.findAll();
    if (locations.length === 0) return null;

    let bestMatch: LocationRecord | null = null;
    let bestScore = 0;

    for (const loc of locations) {
      const matches = loc.features.filter((f) =>
        detectedFeatures.some((detected) =>
          detected.toLowerCase().includes(f.toLowerCase()) ||
          f.toLowerCase().includes(detected.toLowerCase()),
        ),
      ).length;

      const score = matches / Math.max(loc.features.length, 1);
      if (score > bestScore) {
        bestScore = score;
        bestMatch = loc;
      }
    }

    // Retorna apenas se pelo menos 1 feature coincide
    return bestScore > 0 ? bestMatch : null;
  }

  async insert(data: Omit<LocationRecord, 'id'>): Promise<LocationRecord> {
    const id = uuid();
    await this.db.runAsync(
      `INSERT INTO locations (id, name, description, features, navigation_hint)
       VALUES (?, ?, ?, ?, ?)`,
      [id, data.name, data.description, JSON.stringify(data.features), data.navigationHint],
    );
    return { id, ...data };
  }

  async upsert(data: LocationRecord): Promise<void> {
    await this.db.runAsync(
      `INSERT INTO locations (id, name, description, features, navigation_hint)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         name = excluded.name,
         description = excluded.description,
         features = excluded.features,
         navigation_hint = excluded.navigation_hint`,
      [data.id, data.name, data.description, JSON.stringify(data.features), data.navigationHint],
    );
  }
}

interface RawLocation {
  id: string;
  name: string;
  description: string;
  features: string;
  navigation_hint: string;
}

function toRecord(raw: RawLocation): LocationRecord {
  return {
    id: raw.id,
    name: raw.name,
    description: raw.description,
    features: JSON.parse(raw.features) as string[],
    navigationHint: raw.navigation_hint,
  };
}
