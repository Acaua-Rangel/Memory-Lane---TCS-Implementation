import React, { useState, useEffect } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, StatusBar, ScrollView,
  TouchableOpacity, Switch, Alert,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../../navigation/types';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useDatabase } from '../../hooks/useDatabase';
import { Input } from '../../components/ui/Input';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { colors, spacing, radius, typography } from '../../theme';
import { usePatientStore } from '../../store/patientStore';

export function SettingsScreen() {
  const { db } = useDatabase();
  const patientId = usePatientStore((s) => s.patientId);
  const patientName = usePatientStore((s) => s.patientName);
  const language = usePatientStore((s) => s.language);
  const elderlyMode = usePatientStore((s) => s.elderlyMode);
  const setPatient = usePatientStore((s) => s.setPatient);
  const setLanguage = usePatientStore((s) => s.setLanguage);
  const setElderlyMode = usePatientStore((s) => s.setElderlyMode);

  const [nameInput, setNameInput] = useState(patientName ?? '');
  const [saving, setSaving] = useState(false);

  const handleSaveName = async () => {
    if (!nameInput.trim()) return;
    setSaving(true);
    setPatient(patientId ?? '', nameInput.trim());
    await AsyncStorage.setItem('@lane:patient_name', nameInput.trim());
    setSaving(false);
    Alert.alert('Salvo!', `Nome do paciente atualizado para "${nameInput.trim()}"`);
  };

  const handleResetData = () => {
    Alert.alert(
      'Redefinir dados',
      'Isso irá apagar TODOS os dados: pessoas, memórias, agenda, rotina e medicamentos. Não pode ser desfeito.',
      [
        { text: 'Cancelar', style: 'cancel' },
        {
          text: 'Apagar tudo', style: 'destructive',
          onPress: async () => {
            if (!db) return;
            try {
              await AsyncStorage.clear();
              Alert.alert('Feito', 'Dados apagados. Reinicie o app.');
            } catch {
              Alert.alert('Erro', 'Não foi possível apagar os dados.');
            }
          },
        },
      ],
    );
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />
      <View style={styles.header}>
        <Text style={styles.title}>⚙️ Configurações</Text>
      </View>

      <ScrollView contentContainerStyle={styles.content}>

        {/* Perfil do paciente */}
        <Text style={styles.sectionTitle}>Perfil do Paciente</Text>
        <Card>
          <Input
            label="Nome do paciente"
            value={nameInput}
            onChangeText={setNameInput}
            placeholder="Ex: João da Silva"
            hint="A Lane usará este nome para se referir ao paciente"
          />
          <Button
            label={saving ? 'Salvando...' : 'Salvar nome'}
            onPress={handleSaveName}
            loading={saving}
            size="sm"
            style={{ marginTop: spacing.md, alignSelf: 'flex-end' }}
          />
        </Card>

        {/* Preferências de voz */}
        <Text style={styles.sectionTitle}>Voz e Idioma</Text>
        <Card style={styles.card}>
          <View style={styles.row}>
            <View style={styles.rowInfo}>
              <Text style={styles.rowLabel}>Modo Idoso (fala mais devagar)</Text>
              <Text style={styles.rowDesc}>Reduz a velocidade da fala para maior clareza</Text>
            </View>
            <Switch
              value={elderlyMode}
              onValueChange={setElderlyMode}
              trackColor={{ true: colors.primary }}
              thumbColor="#fff"
            />
          </View>

          <View style={[styles.row, { borderTopWidth: 1, borderTopColor: colors.borderLight, paddingTop: spacing.md }]}>
            <Text style={styles.rowLabel}>Idioma</Text>
            <View style={styles.langRow}>
              {(['pt-BR', 'en-US'] as const).map((lang) => (
                <TouchableOpacity
                  key={lang}
                  style={[styles.langChip, language === lang && styles.langChipActive]}
                  onPress={() => setLanguage(lang)}
                >
                  <Text style={[styles.langText, language === lang && styles.langTextActive]}>
                    {lang === 'pt-BR' ? '🇧🇷 PT' : '🇺🇸 EN'}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </Card>

        {/* Sobre */}
        <Text style={styles.sectionTitle}>Sobre o App</Text>
        <Card style={styles.card}>
          <View style={styles.aboutRow}>
            <Text style={styles.aboutIcon}>🧠</Text>
            <View>
              <Text style={styles.aboutTitle}>Memory Lane</Text>
              <Text style={styles.aboutDesc}>Versão 1.0.0 · Hackathon Google DeepMind Gemma 4</Text>
              <Text style={styles.aboutDesc}>Motor: Gemma 4 E2B + TCS (mock mode)</Text>
            </View>
          </View>
        </Card>

        {/* Zona de perigo */}
        <Text style={[styles.sectionTitle, { color: colors.danger }]}>Zona de Perigo</Text>
        <Card style={styles.dangerCard}>
          <Text style={styles.dangerDesc}>
            Apaga todos os dados do app: pessoas, memórias, agenda, rotina e medicamentos.
          </Text>
          <Button
            label="Apagar todos os dados"
            onPress={handleResetData}
            variant="danger"
            fullWidth
            style={{ marginTop: spacing.md }}
          />
        </Card>

        <View style={{ height: spacing.xl }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bgLight },
  header: {
    padding: spacing.lg, backgroundColor: colors.surfaceLight,
    borderBottomWidth: 1, borderBottomColor: colors.borderLight,
  },
  title: { ...typography.h2, color: colors.textDark },
  content: { padding: spacing.lg, gap: spacing.md },
  sectionTitle: { ...typography.small, fontWeight: '700', color: colors.textSecondary, textTransform: 'uppercase', letterSpacing: 0.5 },
  card: { gap: spacing.md },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  rowInfo: { flex: 1, marginRight: spacing.md },
  rowLabel: { ...typography.bodyBold, color: colors.textDark },
  rowDesc: { ...typography.small, color: colors.textSecondary },
  langRow: { flexDirection: 'row', gap: spacing.sm },
  langChip: {
    borderWidth: 1.5, borderColor: colors.borderLight, borderRadius: radius.full,
    paddingHorizontal: spacing.md, paddingVertical: 6, backgroundColor: colors.surfaceLight,
  },
  langChipActive: { borderColor: colors.primary, backgroundColor: colors.primary + '15' },
  langText: { ...typography.small, color: colors.textSecondary },
  langTextActive: { color: colors.primary, fontWeight: '600' },
  aboutRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  aboutIcon: { fontSize: 36 },
  aboutTitle: { ...typography.bodyBold, color: colors.textDark },
  aboutDesc: { ...typography.small, color: colors.textSecondary },
  dangerCard: { borderWidth: 1, borderColor: colors.danger + '40' },
  dangerDesc: { ...typography.small, color: colors.textSecondary, lineHeight: 20 },
});
