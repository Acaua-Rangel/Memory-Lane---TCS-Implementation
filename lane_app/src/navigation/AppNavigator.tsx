import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import type { RootStackParamList } from './types';
import { ModeSelector } from '../screens/ModeSelector';
import { PatientScreen } from '../screens/patient/PatientScreen';
import { CaregiverNavigator } from './CaregiverNavigator';

const Stack = createNativeStackNavigator<RootStackParamList>();

export function AppNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="ModeSelector"
        screenOptions={{ headerShown: false, animation: 'fade' }}
      >
        <Stack.Screen name="ModeSelector" component={ModeSelector} />
        <Stack.Screen name="Patient" component={PatientScreen} />
        <Stack.Screen name="Caregiver" component={CaregiverNavigator} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
