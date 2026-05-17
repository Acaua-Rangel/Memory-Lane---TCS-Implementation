import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, StatusBar, ScrollView,
  TouchableOpacity, Alert, ActivityIndicator,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RoutineStackParamList } from '../../../navigation/types';
import type { RoutineStep } from '../../../types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Input } from '../../../components/ui/Input';
import { Button } from '../../../components/ui/Button';
import { colors, spacing, radius, typography, TIME_OF_DAY_LABELS } from '../../../theme';

type Props = NativeStackScreenProps<RoutineStackParamList, 'RoutineForm'>;

export function RoutineFormScreen({ route, navigation }: Props) {
  const { routineId } = route.params ?? {};
  const isEditing = !!routineId;
  const { db, isReady } = useDatabase();

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [timeOfDay, setTimeOfDay] = useState<RoutineStep['timeOfDay']>('morning');
  const [order, setOrder] = useState('1');
  const [loading, setLoading] = useState(isEditing);
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!isReady || !isEditing) return;
    db!.routines.findAll().then((list) => {
      const step = list.find((s) => s.id === routineId);
      if (!step) return;
      setTitle(step.title); setDescription(step.description);
      setTimeOfDay(step.timeOfDay); setOrder(String(step.order));
      setLoading(false);
    });
  }, [isReady, isEditing, routineId]);

  const validate = () => {
    const e: Record<string, string> = {};
    if (!title.trim()) e.title = 'Título é obrigatório';
    if (!description.trim()) e.description = 'Descrição é obrigatória';
    if (!order || isNaN(Number(order))) e.order = 'Ordem deve ser um número';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSave = async () => {
    if (!validate() || !db) return;
    setSaving(true);
    try {
      const data = {
        title: title.trim(), description: description.trim(),
        timeOfDay, order: Number(order),
      };
      if (isEditing) {
        await db.routines.upsert({ id: routineId!, ...data });
      } else {
        await db.routines.insert(data);
      }
      navigation.goBack();
    } catch {
      Alert.alert('Erro', 'Não foi possível salvar. Tente novamente.');
    } finally {
      setSaving(false);
    }
  };

  const TIME_OPTIONS: RoutineStep['timeOfDay'][] = ['morning', 'afternoon', 'evening', 'night'];
  const TIME_ICONS: Record<string, string> = { morning: '🌅', afternoon: '☀️', evening: '🌆', night: '🌙' };

  if (loading) return <SafeAreaView style={styles.safe}><ActivityIndicator style={{ flex: 1 }} color={colors.primary} /></SafeAreaView>;

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.back}>← Voltar</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{isEditing ? 'Editar Passo' : 'Novo Passo'}</Text>
        <View style={{ width: 60 }} />
      </View>

      <ScrollView contentContainerStyle={styles.form} keyboardShouldPersistTaps="handled">
        <View>
          <Text style={styles.fieldLabel}>Período do dia *</Text>
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

        <Input label="Título do passo *" value={title} onChangeText={setTitle}
          placeholder="Ex: Tomar café da manhã" error={errors.title} />

        <Input label="Descrição *" value={description} onChangeText={setDescription}
          placeholder="Ex: Sentar à mesa, tomar o café que está na cozinha e tomar os remédios da manhã"
          multiline numberOfLines={4} error={errors.description}
          hint="A Lane vai ler esta descrição para o paciente" />

        <Input label="Ordem (posição na rotina) *" value={order} onChangeText={setOrder}
          placeholder="1" keyboardType="number-pad" error={errors.order}
          hint="1 = primeiro passo, 2 = segundo, etc." />

        <Button label={saving ? 'Salvando...' : 'Salvar Passo'} onPress={handleSave}
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
});
