import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, StatusBar, ScrollView,
  TouchableOpacity, Switch, Alert, ActivityIndicator,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { AgendaStackParamList } from '../../../navigation/types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Input } from '../../../components/ui/Input';
import { Button } from '../../../components/ui/Button';
import { colors, spacing, radius, typography, DAY_NAMES_FULL } from '../../../theme';

type Props = NativeStackScreenProps<AgendaStackParamList, 'AgendaForm'>;

export function AgendaFormScreen({ route, navigation }: Props) {
  const { agendaId } = route.params ?? {};
  const isEditing = !!agendaId;
  const { db, isReady } = useDatabase();

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [dayOfWeek, setDayOfWeek] = useState(new Date().getDay());
  const [time, setTime] = useState('');
  const [location, setLocation] = useState('');
  const [isRecurring, setIsRecurring] = useState(true);
  const [loading, setLoading] = useState(isEditing);
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!isReady || !isEditing) return;
    db!.agenda.findAll().then((list) => {
      const ev = list.find((e) => e.id === agendaId);
      if (!ev) return;
      setTitle(ev.title); setDescription(ev.description ?? '');
      setDayOfWeek(ev.dayOfWeek); setTime(ev.time ?? '');
      setLocation(ev.location ?? ''); setIsRecurring(ev.isRecurring);
      setLoading(false);
    });
  }, [isReady, isEditing, agendaId]);

  const validate = () => {
    const e: Record<string, string> = {};
    if (!title.trim()) e.title = 'Título é obrigatório';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSave = async () => {
    if (!validate() || !db) return;
    setSaving(true);
    try {
      const data = {
        title: title.trim(), description: description.trim(),
        dayOfWeek, time: time.trim() || undefined,
        location: location.trim() || undefined, isRecurring,
      };
      if (isEditing) {
        await db.agenda.upsert({ id: agendaId!, ...data });
      } else {
        await db.agenda.insert(data);
      }
      navigation.goBack();
    } catch {
      Alert.alert('Erro', 'Não foi possível salvar. Tente novamente.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <SafeAreaView style={styles.safe}><ActivityIndicator style={{ flex: 1 }} color={colors.primary} /></SafeAreaView>;

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.back}>← Voltar</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{isEditing ? 'Editar Evento' : 'Novo Evento'}</Text>
        <View style={{ width: 60 }} />
      </View>

      <ScrollView contentContainerStyle={styles.form} keyboardShouldPersistTaps="handled">
        <Input label="Título do evento *" value={title} onChangeText={setTitle}
          placeholder="Ex: Consulta com o médico" error={errors.title} />

        <View>
          <Text style={styles.fieldLabel}>Dia da semana *</Text>
          <View style={styles.daysRow}>
            {DAY_NAMES_FULL.map((name, i) => (
              <TouchableOpacity
                key={i} style={[styles.dayChip, dayOfWeek === i && styles.dayChipActive]}
                onPress={() => setDayOfWeek(i)}
              >
                <Text style={[styles.dayChipText, dayOfWeek === i && styles.dayChipTextActive]}>
                  {name.slice(0, 3)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <Input label="Horário (opcional)" value={time} onChangeText={setTime}
          placeholder="Ex: 14:30" hint="Formato HH:MM" keyboardType="numbers-and-punctuation" />

        <Input label="Local (opcional)" value={location} onChangeText={setLocation}
          placeholder="Ex: UBS Centro, Casa da Maria" />

        <Input label="Descrição (opcional)" value={description} onChangeText={setDescription}
          placeholder="Detalhes sobre o compromisso..." multiline numberOfLines={3} />

        <View style={styles.switchRow}>
          <View>
            <Text style={styles.switchLabel}>Evento semanal recorrente?</Text>
            <Text style={styles.switchDesc}>Repete toda semana no mesmo dia</Text>
          </View>
          <Switch value={isRecurring} onValueChange={setIsRecurring}
            trackColor={{ true: colors.primary }} thumbColor="#fff" />
        </View>

        <Button label={saving ? 'Salvando...' : 'Salvar Evento'} onPress={handleSave}
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
  daysRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  dayChip: {
    borderWidth: 1.5, borderColor: colors.borderLight, borderRadius: radius.md,
    paddingHorizontal: 12, paddingVertical: 8, backgroundColor: colors.surfaceLight,
  },
  dayChipActive: { borderColor: colors.primary, backgroundColor: colors.primary },
  dayChipText: { ...typography.small, color: colors.textSecondary },
  dayChipTextActive: { color: '#fff', fontWeight: '600' },
  switchRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    backgroundColor: colors.surfaceLight, padding: spacing.md,
    borderRadius: radius.md, borderWidth: 1, borderColor: colors.borderLight,
  },
  switchLabel: { ...typography.bodyBold, color: colors.textDark },
  switchDesc: { ...typography.small, color: colors.textSecondary },
});
