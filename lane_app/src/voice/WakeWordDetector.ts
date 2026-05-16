// Detecção da wake word "Oi Lane" — sempre escutando com baixo consumo.
// Usa STT com reinício automático após cada detecção ou silêncio.

import { SpeechToText } from './SpeechToText';
import { VOICE } from '../config/constants';

type WakeWordCallback = () => void;

export class WakeWordDetector {
  private stt: SpeechToText;
  private active = false;
  private onDetected: WakeWordCallback | null = null;
  private restartTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    this.stt = new SpeechToText(VOICE.DEFAULT_LANGUAGE);
  }

  start(onDetected: WakeWordCallback): void {
    if (this.active) return;
    this.onDetected = onDetected;
    this.active = true;
    void this.listen();
  }

  stop(): void {
    this.active = false;
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    void this.stt.cancelListening();
  }

  private async listen(): Promise<void> {
    if (!this.active) return;

    void this.stt.startListening(
      (result) => {
        if (result.isFinal) {
          const detected = this.checkForWakeWord(result.transcript);
          if (detected) {
            console.log('[WakeWord] Detectado:', result.transcript);
            this.onDetected?.();
          }
          // Reinicia escuta após cada resultado final
          this.scheduleRestart();
        }
      },
      () => {
        // Erro ou timeout — reinicia automaticamente
        this.scheduleRestart();
      },
    );
  }

  private checkForWakeWord(text: string): boolean {
    const lower = text.toLowerCase().trim();
    return VOICE.WAKE_WORD_ALTERNATIVES.some((ww) => lower.includes(ww)) ||
      lower.startsWith(VOICE.WAKE_WORD);
  }

  private scheduleRestart(delayMs = 500): void {
    if (!this.active) return;
    if (this.restartTimer) clearTimeout(this.restartTimer);
    this.restartTimer = setTimeout(() => {
      void this.listen();
    }, delayMs);
  }

  destroy(): void {
    this.stop();
    this.stt.destroy();
  }
}
