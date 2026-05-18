import React from 'react';
import { Text } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import type {
  CaregiverTabParamList,
  PeopleStackParamList,
  AgendaStackParamList,
  RoutineStackParamList,
  MedicationsStackParamList,
} from './types';
import { colors } from '../theme';

import { PeopleListScreen } from '../screens/caregiver/people/PeopleListScreen';
import { PersonFormScreen } from '../screens/caregiver/people/PersonFormScreen';
import { MemoriesListScreen } from '../screens/caregiver/memories/MemoriesListScreen';
import { MemoryFormScreen } from '../screens/caregiver/memories/MemoryFormScreen';
import { AgendaListScreen } from '../screens/caregiver/agenda/AgendaListScreen';
import { AgendaFormScreen } from '../screens/caregiver/agenda/AgendaFormScreen';
import { RoutineListScreen } from '../screens/caregiver/routine/RoutineListScreen';
import { RoutineFormScreen } from '../screens/caregiver/routine/RoutineFormScreen';
import { MedicationsListScreen } from '../screens/caregiver/medications/MedicationsListScreen';
import { MedicationFormScreen } from '../screens/caregiver/medications/MedicationFormScreen';
import { SettingsScreen } from '../screens/caregiver/SettingsScreen';

const Tab = createBottomTabNavigator<CaregiverTabParamList>();
const PeopleStack = createNativeStackNavigator<PeopleStackParamList>();
const AgendaStack = createNativeStackNavigator<AgendaStackParamList>();
const RoutineStack = createNativeStackNavigator<RoutineStackParamList>();
const MedicationsStack = createNativeStackNavigator<MedicationsStackParamList>();

function PeopleNavigator() {
  return (
    <PeopleStack.Navigator screenOptions={{ headerShown: false }}>
      <PeopleStack.Screen name="PeopleList" component={PeopleListScreen} />
      <PeopleStack.Screen name="PersonForm" component={PersonFormScreen} />
      <PeopleStack.Screen name="MemoriesList" component={MemoriesListScreen} />
      <PeopleStack.Screen name="MemoryForm" component={MemoryFormScreen} />
    </PeopleStack.Navigator>
  );
}

function AgendaNavigator() {
  return (
    <AgendaStack.Navigator screenOptions={{ headerShown: false }}>
      <AgendaStack.Screen name="AgendaList" component={AgendaListScreen} />
      <AgendaStack.Screen name="AgendaForm" component={AgendaFormScreen} />
    </AgendaStack.Navigator>
  );
}

function RoutineNavigator() {
  return (
    <RoutineStack.Navigator screenOptions={{ headerShown: false }}>
      <RoutineStack.Screen name="RoutineList" component={RoutineListScreen} />
      <RoutineStack.Screen name="RoutineForm" component={RoutineFormScreen} />
    </RoutineStack.Navigator>
  );
}

function MedicationsNavigator() {
  return (
    <MedicationsStack.Navigator screenOptions={{ headerShown: false }}>
      <MedicationsStack.Screen name="MedicationsList" component={MedicationsListScreen} />
      <MedicationsStack.Screen name="MedicationForm" component={MedicationFormScreen} />
    </MedicationsStack.Navigator>
  );
}

const TAB_ICON: Record<string, string> = {
  PeopleTab: '👥',
  AgendaTab: '📅',
  RoutineTab: '📋',
  MedicationsTab: '💊',
  Settings: '⚙️',
};

export function CaregiverNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }: { route: { name: string } }) => ({
        headerShown: false,
        tabBarIcon: () => (
          <Text style={{ fontSize: 22 }}>{TAB_ICON[route.name]}</Text>
        ),
        tabBarLabel: ({ focused }: { focused: boolean }) => {
          const labels: Record<string, string> = {
            PeopleTab: 'Pessoas',
            AgendaTab: 'Agenda',
            RoutineTab: 'Rotina',
            MedicationsTab: 'Remédios',
            Settings: 'Config',
          };
          return (
            <Text style={{
              fontSize: 10,
              color: focused ? colors.primary : colors.textMuted,
              fontWeight: focused ? '600' : '400',
            }}>
              {labels[route.name]}
            </Text>
          );
        },
        tabBarStyle: {
          backgroundColor: colors.surfaceLight,
          borderTopColor: colors.borderLight,
          height: 70,
          paddingBottom: 8,
          paddingTop: 4,
        },
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.textMuted,
      })}
    >
      <Tab.Screen name="PeopleTab" component={PeopleNavigator} />
      <Tab.Screen name="AgendaTab" component={AgendaNavigator} />
      <Tab.Screen name="RoutineTab" component={RoutineNavigator} />
      <Tab.Screen name="MedicationsTab" component={MedicationsNavigator} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}
