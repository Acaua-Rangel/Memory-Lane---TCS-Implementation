// Sistema de alertas offline-first para cuidadores.
// Armazena alertas pendentes no AsyncStorage e envia quando houver conexão.

import * as Notifications from 'expo-notifications';
import * as Network from 'expo-network';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { AlertType, CaregiverContact, PendingAlert } from '../types';
import { ALERTS } from '../config/constants';
import { uuid } from '../database/utils';

const PENDING_ALERTS_KEY = '@lane:pending_alerts';
const CAREGIVERS_KEY = '@lane:caregivers';

export class CaregiverAlertService {
  private caregivers: CaregiverContact[] = [];
  private isFlushingQueue = false;

  async initialize(): Promise<void> {
    await this.loadCaregivers();
    await this.configurePushNotifications();
  }

  private async configurePushNotifications(): Promise<void> {
    await Notifications.setNotificationHandler({
      handleNotification: async () => ({
        shouldShowAlert: true,
        shouldPlaySound: true,
        shouldSetBadge: true,
      }),
    });
  }

  // ── Cuidadores ─────────────────────────────────────────────────────────────

  async addCaregiver(contact: CaregiverContact): Promise<void> {
    const existing = this.caregivers.findIndex((c) => c.id === contact.id);
    if (existing >= 0) {
      this.caregivers[existing] = contact;
    } else {
      this.caregivers.push(contact);
    }
    await this.saveCaregivers();
  }

  async removeCaregiver(id: string): Promise<void> {
    this.caregivers = this.caregivers.filter((c) => c.id !== id);
    await this.saveCaregivers();
  }

  getCaregivers(): CaregiverContact[] {
    return [...this.caregivers];
  }

  getPrimaryCaregiver(): CaregiverContact | null {
    return this.caregivers.find((c) => c.isPrimary) ?? this.caregivers[0] ?? null;
  }

  // ── Envio de alertas ───────────────────────────────────────────────────────

  async sendAlert(type: AlertType, details: string): Promise<void> {
    const alert: PendingAlert = {
      id: uuid(),
      type,
      details,
      timestamp: Date.now(),
      retryCount: 0,
    };

    // Notificação local imediata (funciona offline)
    await this.sendLocalNotification(alert);

    // Tenta envio push remoto
    const networkState = await Network.getNetworkStateAsync();
    if (networkState.isConnected && networkState.isInternetReachable) {
      await this.sendPushNotification(alert);
    } else {
      // Enfileira para quando houver conexão
      await this.enqueueAlert(alert);
    }
  }

  // ── Notificação local (sempre funciona, mesmo offline) ────────────────────

  private async sendLocalNotification(alert: PendingAlert): Promise<void> {
    const titles: Record<AlertType, string> = {
      confused: '⚠️ Lane: Paciente confuso',
      wandering: '🚶 Lane: Paciente pode estar se perdendo',
      missed_medication: '💊 Lane: Remédio não tomado',
      fall_detected: '🆘 Lane: Possível queda detectada',
      emergency: '🆘 EMERGÊNCIA — Lane',
    };

    const caregiver = this.getPrimaryCaregiver();

    await Notifications.scheduleNotificationAsync({
      content: {
        title: titles[alert.type] ?? 'Lane: Alerta',
        body: `${caregiver ? `Para ${caregiver.name}: ` : ''}${alert.details}`,
        data: { alertId: alert.id, type: alert.type },
        sound: true,
      },
      trigger: null, // Imediato
    });
  }

  // ── Push remoto (requer backend / Firebase) ───────────────────────────────
  // Em produção: substitua por chamada à sua API ou Firebase Cloud Messaging

  private async sendPushNotification(alert: PendingAlert): Promise<boolean> {
    const tokens = this.caregivers
      .filter((c) => c.pushToken)
      .map((c) => c.pushToken as string);

    if (tokens.length === 0) return false;

    try {
      // Usa o serviço de notificações do Expo — substitua por Firebase em produção
      const messages = tokens.map((token) => ({
        to: token,
        sound: 'default',
        title: `Lane Alert: ${alert.type}`,
        body: alert.details,
        data: { alertId: alert.id, type: alert.type, timestamp: alert.timestamp },
        priority: 'high',
      }));

      const response = await fetch('https://exp.host/--/api/v2/push/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(messages),
      });

      return response.ok;
    } catch (err) {
      console.error('[CaregiverAlert] Erro ao enviar push:', err);
      return false;
    }
  }

  // ── Fila offline ──────────────────────────────────────────────────────────

  private async enqueueAlert(alert: PendingAlert): Promise<void> {
    const pending = await this.loadPendingAlerts();
    pending.push(alert);
    await AsyncStorage.setItem(PENDING_ALERTS_KEY, JSON.stringify(pending));
    console.log(`[CaregiverAlert] Alerta enfileirado (${pending.length} pendentes)`);
  }

  async getPendingAlerts(): Promise<PendingAlert[]> {
    return this.loadPendingAlerts();
  }

  async flushPendingAlerts(): Promise<void> {
    if (this.isFlushingQueue) return;
    this.isFlushingQueue = true;

    try {
      const pending = await this.loadPendingAlerts();
      if (pending.length === 0) return;

      const networkState = await Network.getNetworkStateAsync();
      if (!networkState.isConnected || !networkState.isInternetReachable) return;

      const remaining: PendingAlert[] = [];

      for (const alert of pending) {
        if (alert.retryCount >= ALERTS.MAX_RETRY_COUNT) continue; // Descarta após muitas tentativas

        const sent = await this.sendPushNotification(alert);
        if (!sent) {
          remaining.push({ ...alert, retryCount: alert.retryCount + 1 });
        }
      }

      await AsyncStorage.setItem(PENDING_ALERTS_KEY, JSON.stringify(remaining));
      console.log(`[CaregiverAlert] Flush: ${pending.length - remaining.length} enviados, ${remaining.length} pendentes`);
    } finally {
      this.isFlushingQueue = false;
    }
  }

  // ── Persistência ──────────────────────────────────────────────────────────

  private async loadPendingAlerts(): Promise<PendingAlert[]> {
    const raw = await AsyncStorage.getItem(PENDING_ALERTS_KEY);
    return raw ? (JSON.parse(raw) as PendingAlert[]) : [];
  }

  private async loadCaregivers(): Promise<void> {
    const raw = await AsyncStorage.getItem(CAREGIVERS_KEY);
    this.caregivers = raw ? (JSON.parse(raw) as CaregiverContact[]) : [];
  }

  private async saveCaregivers(): Promise<void> {
    await AsyncStorage.setItem(CAREGIVERS_KEY, JSON.stringify(this.caregivers));
  }

  async requestPushPermission(): Promise<string | null> {
    const { status } = await Notifications.requestPermissionsAsync();
    if (status !== 'granted') return null;
    const tokenData = await Notifications.getExpoPushTokenAsync();
    return tokenData.data;
  }
}
