import { create } from 'zustand';
import type { TTSLanguage, CaregiverContact, PendingAlert } from '../types';

interface PatientStore {
  patientId: string | null;
  patientName: string | null;
  language: TTSLanguage;
  elderlyMode: boolean;       // TTS mais lento, fontes maiores
  caregivers: CaregiverContact[];
  pendingAlerts: PendingAlert[];
  pendingAlertsCount: number;
  isOnboarded: boolean;

  // Ações
  setPatient: (id: string, name: string) => void;
  setLanguage: (lang: TTSLanguage) => void;
  setElderlyMode: (enabled: boolean) => void;
  setCaregivers: (caregivers: CaregiverContact[]) => void;
  addCaregiver: (caregiver: CaregiverContact) => void;
  removeCaregiver: (id: string) => void;
  setPendingAlerts: (alerts: PendingAlert[]) => void;
  addPendingAlert: (alert: PendingAlert) => void;
  clearPendingAlerts: () => void;
  setOnboarded: (value: boolean) => void;
  reset: () => void;
}

const initialState = {
  patientId: null,
  patientName: null,
  language: 'pt-BR' as TTSLanguage,
  elderlyMode: true,
  caregivers: [] as CaregiverContact[],
  pendingAlerts: [] as PendingAlert[],
  pendingAlertsCount: 0,
  isOnboarded: false,
};

export const usePatientStore = create<PatientStore>((set, get) => ({
  ...initialState,

  setPatient: (patientId, patientName) => set({ patientId, patientName }),

  setLanguage: (language) => set({ language }),

  setElderlyMode: (elderlyMode) => set({ elderlyMode }),

  setCaregivers: (caregivers) => set({ caregivers }),

  addCaregiver: (caregiver) => {
    const current = get().caregivers;
    const exists = current.findIndex((c) => c.id === caregiver.id);
    if (exists >= 0) {
      const updated = [...current];
      updated[exists] = caregiver;
      set({ caregivers: updated });
    } else {
      set({ caregivers: [...current, caregiver] });
    }
  },

  removeCaregiver: (id) => set((s) => ({
    caregivers: s.caregivers.filter((c) => c.id !== id),
  })),

  setPendingAlerts: (pendingAlerts) => set({
    pendingAlerts,
    pendingAlertsCount: pendingAlerts.length,
  }),

  addPendingAlert: (alert) => set((s) => ({
    pendingAlerts: [...s.pendingAlerts, alert],
    pendingAlertsCount: s.pendingAlertsCount + 1,
  })),

  clearPendingAlerts: () => set({ pendingAlerts: [], pendingAlertsCount: 0 }),

  setOnboarded: (isOnboarded) => set({ isOnboarded }),

  reset: () => set(initialState),
}));

// Selectors
export const selectPrimaryCaregiver = (s: PatientStore) =>
  s.caregivers.find((c) => c.isPrimary) ?? s.caregivers[0] ?? null;

export const selectHasCaregivers = (s: PatientStore) => s.caregivers.length > 0;
