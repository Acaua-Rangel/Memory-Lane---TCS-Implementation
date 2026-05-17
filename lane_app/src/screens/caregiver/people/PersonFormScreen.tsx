import React, { useEffect, useState } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView, StatusBar, ScrollView,
  TouchableOpacity, Switch, ActivityIndicator, Alert,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { PeopleStackParamList } from '../../../navigation/types';
import { useDatabase } from '../../../hooks/useDatabase';
import { Input } from '../../../components/ui/Input';
import { Button } from '../../../components/ui/Button';
import { colors, spacing, radius, typography } from '../../../theme';

type Props = NativeStackScreenProps<PeopleStackParamList, 'PersonForm'>;

export function PersonFormScreen({ route, navigation }: Props) {
  const { personId } = route.params ?? {};
  const isEditing = !!personId;
  const { db, isReady } = useDatabase();

  const [name, setName] = useState('');
  const [relationship, setRelationship] = useState('');
  const [bio, setBio] = useState('');
  const [phone, setPhone] = useState('');
  const [isCaregiver, setIsCaregiver] = useState(false);
  const [loading, setLoading] = useState(isEditing);
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!isReady || !isEditing) return;
    db!.persons.findById(personId!).then((p) => {
      if (!p) return;
      setName(p.name);
      setRelationship(p.relationship);
      setBio(p.bio);
      setPhone(p.phoneNumber ?? '');
      setIsCaregiver(p.isCaregiver);
      setLoading(false);
    });
  }, [isReady, isEditing, personId]);

  const validate = () => {
    const e: Record<string, string> = {};
    if (!name.trim()) e.name = 'Nome é obrigatório';
    if (!relationship.trim()) e.relationship = 'Relacionamento é obrigatório';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSave = async () => {
    if (!validate() || !db) return;
    setSaving(true);
    try {
      if (isEditing) {
        await db.persons.update(personId!, {
          name: name.trim(),
          relationship: relationship.trim(),
          bio: bio.trim(),
          phoneNumber: phone.trim() || undefined,
          isCaregiver,
        });
      } else {
        await db.persons.insert({
          name: name.trim(),
          relationship: relationship.trim(),
          bio: bio.trim(),
          phoneNumber: phone.trim() || undefined,
          isCaregiver,
        });
      }
      navigation.goBack();
    } catch (err) {
      Alert.alert('Erro', 'Não foi possível salvar. Tente novamente.');
    } finally {
      setSaving(false);
    }
  };

  const QUICK_RELATIONSHIPS = ['Filho(a)', 'Cônjuge', 'Irmão/Irmã', 'Neto(a)', 'Médico(a)', 'Cuidador(a)', 'Amigo(a)'];

  if (loading) {
    return (
      <SafeAreaView style={styles.safe}>
        <ActivityIndicator style={{ flex: 1 }} color={colors.primary} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={colors.bgLight} />

      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.back}>← Voltar</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{isEditing ? 'Editar Pessoa' : 'Nova Pessoa'}</Text>
        <View style={{ width: 60 }} />
      </View>

      <ScrollView contentContainerStyle={styles.form} keyboardShouldPersistTaps="handled">
        <Input
          label="Nome completo *"
          value={name}
          onChangeText={setName}
          placeholder="Ex: Maria Silva"
          error={errors.name}
          autoCapitalize="words"
        />

        <View>
          <Text style={styles.fieldLabel}>Relacionamento *</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.quickRow}>
            {QUICK_RELATIONSHIPS.map((r) => (
              <TouchableOpacity
                key={r}
                style={[styles.quickChip, relationship === r && styles.quickChipActive]}
                onPress={() => setRelationship(r)}
              >
                <Text style={[styles.quickChipText, relationship === r && styles.quickChipTextActive]}>
                  {r}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
          <Input
            value={relationship}
            onChangeText={setRelationship}
            placeholder="Ou digite o relacionamento"
            error={errors.relationship}
          />
        </View>

        <Input
          label="Biografia / História de vida"
          value={bio}
          onChangeText={setBio}
          placeholder="Conte um pouco sobre esta pessoa — onde mora, o que faz, memórias importantes..."
          multiline
          numberOfLines={5}
          hint="Quanto mais detalhes, melhor a Lane poderá ajudar"
        />

        <Input
          label="Telefone"
          value={phone}
          onChangeText={setPhone}
          placeholder="Ex: +55 11 99999-0000"
          keyboardType="phone-pad"
        />

        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Text style={styles.switchLabel}>É cuidador(a)?</Text>
            <Text style={styles.switchDesc}>Cuidadores recebem alertas de emergência</Text>
          </View>
          <Switch
            value={isCaregiver}
            onValueChange={setIsCaregiver}
            trackColor={{ true: colors.primary }}
            thumbColor="#fff"
          />
        </View>

        <Button
          label={saving ? 'Salvando...' : 'Salvar'}
          onPress={handleSave}
          loading={saving}
          fullWidth
          size="lg"
          style={styles.saveBtn}
        />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bgLight },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.lg,
    backgroundColor: colors.surfaceLight,
    borderBottomWidth: 1,
    borderBottomColor: colors.borderLight,
  },
  back: { ...typography.body, color: colors.primary },
  title: { ...typography.h3, color: colors.textDark },

  form: { padding: spacing.lg, gap: spacing.lg },

  fieldLabel: { ...typography.small, color: colors.textDark, fontWeight: '600', marginBottom: spacing.sm },
  quickRow: { marginBottom: spacing.sm },
  quickChip: {
    borderWidth: 1.5,
    borderColor: colors.borderLight,
    borderRadius: radius.full,
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
    marginRight: spacing.sm,
    backgroundColor: colors.surfaceLight,
  },
  quickChipActive: { borderColor: colors.primary, backgroundColor: colors.primary + '15' },
  quickChipText: { ...typography.small, color: colors.textSecondary },
  quickChipTextActive: { color: colors.primary, fontWeight: '600' },

  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: colors.surfaceLight,
    padding: spacing.md,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.borderLight,
  },
  switchInfo: { flex: 1 },
  switchLabel: { ...typography.bodyBold, color: colors.textDark },
  switchDesc: { ...typography.small, color: colors.textSecondary },

  saveBtn: { marginTop: spacing.md },
});
