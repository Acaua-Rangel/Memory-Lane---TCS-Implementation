import * as SQLite from 'expo-sqlite';
import { SCHEMA_SQL } from './schema';
import { PersonRepository } from './repositories/PersonRepository';
import { FaceEmbeddingRepository } from './repositories/FaceEmbeddingRepository';
import { MemoryRepository } from './repositories/MemoryRepository';
import { MedicationRepository } from './repositories/MedicationRepository';
import { EncounterRepository } from './repositories/EncounterRepository';
import { LocationRepository } from './repositories/LocationRepository';
import { AgendaRepository } from './repositories/AgendaRepository';
import { PreferenceRepository } from './repositories/PreferenceRepository';
import { RoutineRepository } from './repositories/RoutineRepository';
import { DATABASE } from '../config/constants';
import {
  SAMPLE_PERSONS,
  SAMPLE_MEDICATIONS,
  SAMPLE_LOCATIONS,
  SAMPLE_AGENDA,
  SAMPLE_ROUTINE,
} from '../config/sampleData';
import { uuid } from './utils';

export class LaneDatabase {
  private db!: SQLite.SQLiteDatabase;

  // Repositórios públicos — usados pelo ToolExecutor e pelo app
  persons!: PersonRepository;
  faceEmbeddings!: FaceEmbeddingRepository;
  memories!: MemoryRepository;
  medications!: MedicationRepository;
  encounters!: EncounterRepository;
  locations!: LocationRepository;
  agenda!: AgendaRepository;
  preferences!: PreferenceRepository;
  routines!: RoutineRepository;

  private static instance: LaneDatabase | null = null;

  static getInstance(): LaneDatabase {
    if (!LaneDatabase.instance) {
      LaneDatabase.instance = new LaneDatabase();
    }
    return LaneDatabase.instance;
  }

  async open(): Promise<void> {
    this.db = await SQLite.openDatabaseAsync(DATABASE.NAME);
    await this.runMigrations();
    this.initRepositories();
  }

  private async runMigrations(): Promise<void> {
    // Executa DDL — cada CREATE TABLE IF NOT EXISTS é idempotente
    await this.db.execAsync(SCHEMA_SQL);
  }

  private initRepositories(): void {
    this.persons = new PersonRepository(this.db);
    this.faceEmbeddings = new FaceEmbeddingRepository(this.db);
    this.memories = new MemoryRepository(this.db);
    this.medications = new MedicationRepository(this.db);
    this.encounters = new EncounterRepository(this.db);
    this.locations = new LocationRepository(this.db);
    this.agenda = new AgendaRepository(this.db);
    this.preferences = new PreferenceRepository(this.db);
    this.routines = new RoutineRepository(this.db);
  }

  // Popula o banco com dados de exemplo na primeira execução
  async seedIfEmpty(): Promise<void> {
    const personCount = await this.persons.count();
    if (personCount > 0) return; // Já semeado

    await this.db.withTransactionAsync(async () => {
      // Pessoas
      for (const p of SAMPLE_PERSONS) {
        await this.persons.upsert({ ...p, createdAt: Date.now(), updatedAt: Date.now() });
      }

      // Medicamentos
      for (const m of SAMPLE_MEDICATIONS) {
        await this.medications.insert({ ...m, id: uuid() } as Parameters<typeof this.medications.insert>[0]);
      }

      // Localizações
      for (const l of SAMPLE_LOCATIONS) {
        await this.locations.insert(l);
      }

      // Agenda
      for (const a of SAMPLE_AGENDA) {
        await this.agenda.insert(a);
      }

      // Rotina
      for (const r of SAMPLE_ROUTINE) {
        await this.routines.insert(r);
      }
    });
  }

  // Transação explícita para operações compostas
  async withTransaction<T>(fn: () => Promise<T>): Promise<T> {
    return this.db.withTransactionAsync(fn);
  }

  async close(): Promise<void> {
    await this.db.closeAsync();
  }

  isOpen(): boolean {
    return !!this.db;
  }
}
