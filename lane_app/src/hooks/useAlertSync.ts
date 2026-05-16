// Monitora conectividade e faz flush automático dos alertas pendentes.

import { useEffect, useRef, useState } from 'react';
import * as Network from 'expo-network';
import { AppState, type AppStateStatus } from 'react-native';
import { ALERTS } from '../config/constants';
import { usePatientStore } from '../store/patientStore';
import type { CaregiverAlertService } from '../alerts/CaregiverAlertService';

export function useAlertSync(alertService: CaregiverAlertService | null) {
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSyncAt, setLastSyncAt] = useState<number | null>(null);
  const syncTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { pendingAlertsCount, setPendingAlerts } = usePatientStore();

  const syncAlerts = async (): Promise<void> => {
    if (!alertService || isSyncing) return;

    const networkState = await Network.getNetworkStateAsync();
    if (!networkState.isConnected || !networkState.isInternetReachable) return;

    const pending = await alertService.getPendingAlerts();
    if (pending.length === 0) return;

    setIsSyncing(true);
    try {
      await alertService.flushPendingAlerts();
      const remaining = await alertService.getPendingAlerts();
      setPendingAlerts(remaining);
      setLastSyncAt(Date.now());
    } finally {
      setIsSyncing(false);
    }
  };

  // Sincroniza quando o app volta ao foreground
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state: AppStateStatus) => {
      if (state === 'active') {
        void syncAlerts();
      }
    });
    return () => subscription.remove();
  }, [alertService]);

  // Intervalo periódico de sincronização
  useEffect(() => {
    syncTimerRef.current = setInterval(() => {
      void syncAlerts();
    }, ALERTS.RETRY_INTERVAL_MS);

    return () => {
      if (syncTimerRef.current) clearInterval(syncTimerRef.current);
    };
  }, [alertService]);

  // Carrega alertas pendentes na montagem
  useEffect(() => {
    if (!alertService) return;
    alertService.getPendingAlerts().then(setPendingAlerts).catch(console.error);
  }, [alertService]);

  return {
    isSyncing,
    pendingCount: pendingAlertsCount,
    lastSyncAt,
    syncNow: syncAlerts,
  };
}
