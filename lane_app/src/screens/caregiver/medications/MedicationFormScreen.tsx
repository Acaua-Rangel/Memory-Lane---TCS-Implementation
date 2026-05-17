import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, StatusBar, ScrollView,
  TouchableOpacity, Switch, Alert, ActivityIndicator,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { MedicationsStackParamList } from '../../../navigation/types';
import type { MedicationRecord } from '../../../types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Input } from '../../../components/ui/Input';
import { Button } from '../../../components/ui/Button';
import { colors, spacing, radius, typography, TIME_OF_DAY_LABELS } from '../../../theme';

type Props = NativeStackScreenProps<MedicationsStackParamList, 'MedicationForm'>;

export function MedicationFormScreen({ route, navigation }: Props) {
  const { medicationId } = route.params ?? {};
  const isEditing = !!medicationId;
  const { db, isReady } = useDatabase();

  const [name, setName] = useState('');
  const [dosage, setDosage] = useState('');
  const [timeOfDay, setTimeOfDay] = useState<MedicationRecord['timeOfDay']>('morning');
  const [description, setDescription] = useState('');
  const [instructions, setInstructions] = useState('');
  const [isActive, setIsActive] = useState(true);
  const [loading, setLoading] = useState(isEditing);
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!isReady || !isEditing) return;
    db!.medications.findById(medicationId!).then((m) => {
      if (!m) return;
      setName(m.name); setDosage(m.dosage); setTimeOfDay(m.timeOfDay);
      setDescription(m.description); setInstructions(m.instructions);
      setIsActive(m.isActive); setLoading(false);
    });
  }, [isReady, isEditing, medicationId]);

  const validate = () => {
    const e: Record<string, string> = {};
    if (!name.trim()) e.name = 'Nome é obrigatório';
    if (!dosage.trim()) e.dosage = 'Dosagem é obrigatória';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSave = async () => {
    if (!validate() || !db) return;
    setSaving(true);
    try {
      const data: MedicationRecord = {
        id: medicationId ?? '',
        name: name.trim(), dosage: dosage.trim(), timeOfDay,
        description: description.trim(), instructions: instructions.trim(), isActive,
      };
      if (isEditing) {
        await db.medications.upsert(data);
      } else {
        await db.medications.insert({
          name: data.name, dosage: data.dosage, timeOfDay: data.timeOfDay,
          description: data.description, instructions: data.instructions, isActive: data.isActive,
        });
      }
      navigation.goBack();
    } catch {
      Alert.alert('Erro', 'Não foi possível salvar. Tente novamente.');
    } finally {
      setSaving(false);
    }
  };

  const TIME_OPTIONS: MedicationRecord['timeOfDay'][] = ['morning', 'afternoon', 'evening', 'night'];
  const TIME_ICONS: Record<string, string> = { morning: '🌅', afternoon: '☀️', evening: '🌆', night: '🌙' };

  if (loading) return <SafeAreaView style={styles.safe}><ActivityIndicator style={{ flex: 1 }} color={colors.primary} /></SafeAreaView>;

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.back}>← Voltar</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{isEditing ? 'Editar Remédio' : 'Novo Remédio'}</Text>
        <View style={{ width: 60 }} />
      </View>

      <ScrollView contentContainerStyle={styles.form} keyboardShouldPersistTaps="handled">
        <Input label="Nome do medicamento *" value={name} onChangeText={setName}
          placeholder="Ex: Donepezila" error={errors.name} />

        <Input label="Dosagem *" value={dosage} onChangeText={setDosage}
          placeholder="Ex: 10mg - 1 comprimido" error={errors.dosage} />

        <View>
          <Text style={styles.fieldLabel}>Horário de tomar *</Text>
          <View style={styles.timeRow}>
            {TIME_OPTIONS.map((t) => (
              <TouchableOpacity
                key={t}
                style={[styles.timeChip, timeOfDay === t && styles.timeChipActive]}
                onPress={() => setTimeOfDay(t)}
              >
                <Text style={styles.timeIcon}>{TIME_ICONS[t]}</Text>
                <Text style={[styles.timeText, timeOfDay === t && styles.timeTextActive]}>
                  {TIME_OF_DAY_LABELS[t]}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <Input label="Descrição visual" value={description} onChangeText={setDescription}
          placeholder="Ex: Comprimido redondo branco pequeno"
          hint="Ajuda o paciente a identificar visualmente o remédio" />

        <Input label="Instruções de uso" value={instructions} onChangeText={setInstructions}
          placeholder="Ex: Tomar com meio copo de água, após o café"
          multiline numberOfLines={3} />

        <View style={styles.switchRow}>
          <View>
            <Text style={styles.switchLabel}>Medicamento ativo?</Text>
            <Text style={styles.switchDesc}>Desative temporariamente sem remover</Text>
          </View>
          <Switch value={isActive} onValueChange={setIsActive}
            trackColor={{ true: colors.success }} thumbColor="#fff" />
        </View>

        <Button label={saving ? 'Salvando...' : 'Salvar Remédio'} onPress={handleSave}
          loading={saving} fullWidth size="lg" style={{ marginTop: spacing.md }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bgLight },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    padding: spacing.lg, backgroundColor: colors.surfaceLight,
    borderBottomWidth: 1, borderBottomColor: colors.borderLight,
  },
  back: { ...typography.body, color: colors.primary },
  title: { ...typography.h3, color: colors.textDark },
  form: { padding: spacing.lg, gap: spacing.lg },
  fieldLabel: { ...typography.small, fontWeight: '600', color: colors.textDark, marginBottom: spacing.sm },
  timeRow: { flexDirection: 'row', gap: spacing.sm, flexWrap: 'wrap' },
  timeChip: {
    flex: 1, minWidth: 70, borderWidth: 1.5, borderColor: colors.borderLight,
    borderRadius: radius.md, padding: spacing.sm, alignItems: 'center',
    backgroundColor: colors.surfaceLight, gap: 4,
  },
  timeChipActive: { borderColor: colors.primary, backgroundColor: colors.primary + '15' },
  timeIcon: { fontSize: 20 },
  timeText: { ...typography.caption, color: colors.textSecondary },
  timeTextActive: { color: colors.primary, fontWeight: '600' },
  switchRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: colors.surfaceLight, padding: spacing.md,
    borderRadius: radius.md, borderWidth: 1, borderColor: colors.borderLight,
  },
  switchLabel: { ...typography.bodyBold, color: colors.textDark },
  switchDesc: { ...typography.small, color: colors.textSecondary },
});
