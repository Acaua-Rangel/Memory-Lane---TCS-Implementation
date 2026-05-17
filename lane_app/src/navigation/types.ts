import type { NavigatorScreenParams } from '@react-navigation/native';

export type RootStackParamList = {
  ModeSelector: undefined;
  Patient: undefined;
  Caregiver: undefined;
};

export type CaregiverTabParamList = {
  PeopleTab: NavigatorScreenParams<PeopleStackParamList>;
  AgendaTab: NavigatorScreenParams<AgendaStackParamList>;
  RoutineTab: NavigatorScreenParams<RoutineStackParamList>;
  MedicationsTab: NavigatorScreenParams<MedicationsStackParamList>;
  Settings: undefined;
};

export type PeopleStackParamList = {
  PeopleList: undefined;
  PersonForm: { personId?: string };
  MemoriesList: { personId: string; personName: string };
  MemoryForm: { personId: string; memoryId?: string };
};

export type AgendaStackParamList = {
  AgendaList: undefined;
  AgendaForm: { agendaId?: string };
};

export type RoutineStackParamList = {
  RoutineList: undefined;
  RoutineForm: { routineId?: string };
};

export type MedicationsStackParamList = {
  MedicationsList: undefined;
  MedicationForm: { medicationId?: string };
};
