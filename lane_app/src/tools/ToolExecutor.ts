// Executa os 9 tools do Lane contra o LaneDatabase.
// Espelha sqlite_tools.py do projeto de treinamento.

import type { LaneDatabase } from '../database/LaneDatabase';
import type { ParsedToolCall, ToolResult, AlertType } from '../types';
import type { CaregiverAlertService } from '../alerts/CaregiverAlertService';

export class ToolExecutor {
  constructor(
    private db: LaneDatabase,
    private alertService: CaregiverAlertService,
  ) {}

  async execute(call: ParsedToolCall): Promise<ToolResult> {
    try {
      const data = await this.dispatch(call);
      return { toolName: call.name, success: true, data };
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      console.error(`[ToolExecutor] ${call.name} failed:`, error);
      return { toolName: call.name, success: false, error };
    }
  }

  async executeAll(calls: ParsedToolCall[]): Promise<ToolResult[]> {
    return Promise.all(calls.map((c) => this.execute(c)));
  }

  private async dispatch(call: ParsedToolCall): Promise<unknown> {
    const p = call.parameters as Record<string, unknown>;

    switch (call.name) {
      case 'read_person':
        return this.readPerson(p.face_id as string);

      case 'write_encounter':
        return this.writeEncounter(
          p.person as string,
          p.context as string,
          p.location as string | undefined,
        );

      case 'get_medication':
        return this.getMedication(p.time_of_day as string);

      case 'describe_location':
        return this.describeLocation(p.features as string[]);

      case 'alert_caregiver':
        return this.alertCaregiver(p.type as AlertType, p.details as string);

      case 'get_agenda':
        return this.getAgenda(p.day_of_week as string);

      case 'save_preference':
        return this.savePreference(
          p.category as string,
          p.key as string,
          p.value as string,
        );

      case 'get_preferences':
        return this.getPreferences(p.category as string);

      case 'get_routine':
        return this.getRoutine(p.time_of_day as string);

      default:
        throw new Error(`Unknown tool: ${call.name as string}`);
    }
  }

  // ── Tool 1 ─────────────────────────────────────────────────────
  private async readPerson(faceId: string) {
    const person = await this.db.persons.findById(faceId);
    if (!person) return { found: false, message: 'Pessoa não encontrada no banco.' };

    const [memories, lastEncounter] = await Promise.all([
      this.db.memories.findByPersonId(faceId, 3),
      this.db.encounters.findLast(faceId),
    ]);

    return {
      found: true,
      person,
      recentMemories: memories,
      lastEncounter,
    };
  }

  // ── Tool 2 ─────────────────────────────────────────────────────
  private async writeEncounter(person: string, context: string, location?: string) {
    // Tenta encontrar pelo ID primeiro, depois pelo nome
    let personRecord = await this.db.persons.findById(person);
    if (!personRecord) {
      personRecord = await this.db.persons.findByName(person);
    }
    if (!personRecord) {
      return { recorded: false, message: `Pessoa "${person}" não encontrada.` };
    }

    const encounter = await this.db.encounters.insert({
      personId: personRecord.id,
      context,
      location,
    });

    return { recorded: true, encounterId: encounter.id };
  }

  // ── Tool 3 ─────────────────────────────────────────────────────
  private async getMedication(timeOfDay: string) {
    const validTimes = ['morning', 'afternoon', 'evening', 'night'];
    const time = validTimes.includes(timeOfDay)
      ? (timeOfDay as 'morning' | 'afternoon' | 'evening' | 'night')
      : 'morning';

    const meds = await this.db.medications.findByTimeOfDay(time);
    if (meds.length === 0) {
      return { found: false, timeOfDay: time, message: 'Nenhum medicamento neste horário.' };
    }
    return { found: true, timeOfDay: time, medications: meds };
  }

  // ── Tool 4 ─────────────────────────────────────────────────────
  private async describeLocation(features: string[]) {
    const location = await this.db.locations.findByFeatures(features);
    if (!location) {
      return {
        found: false,
        message: 'Não reconheci este ambiente. Pode ser um lugar novo.',
      };
    }
    return { found: true, location };
  }

  // ── Tool 5 ─────────────────────────────────────────────────────
  private async alertCaregiver(type: AlertType, details: string) {
    await this.alertService.sendAlert(type, details);
    return { sent: true, type, details };
  }

  // ── Tool 6 ─────────────────────────────────────────────────────
  private async getAgenda(dayOfWeek: string) {
    const day = parseInt(dayOfWeek, 10);
    const validDay = isNaN(day) ? new Date().getDay() : Math.min(6, Math.max(0, day));
    const events = await this.db.agenda.findByDayOfWeek(validDay);

    const dayNames = ['domingo', 'segunda', 'terça', 'quarta', 'quinta', 'sexta', 'sábado'];
    return {
      dayOfWeek: validDay,
      dayName: dayNames[validDay],
      events,
      count: events.length,
    };
  }

  // ── Tool 7 ─────────────────────────────────────────────────────
  private async savePreference(category: string, key: string, value: string) {
    const pref = await this.db.preferences.set(category, key, value);
    return { saved: true, preference: pref };
  }

  // ── Tool 8 ─────────────────────────────────────────────────────
  private async getPreferences(category: string) {
    const prefs = await this.db.preferences.findByCategory(category);
    return { category, preferences: prefs, count: prefs.length };
  }

  // ── Tool 9 ─────────────────────────────────────────────────────
  private async getRoutine(timeOfDay: string) {
    const validTimes = ['morning', 'afternoon', 'evening', 'night'];
    const time = validTimes.includes(timeOfDay)
      ? (timeOfDay as 'morning' | 'afternoon' | 'evening' | 'night')
      : 'morning';

    const steps = await this.db.routines.findByTimeOfDay(time);
    return { timeOfDay: time, steps, count: steps.length };
  }
}
