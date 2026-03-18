
import { useState, useCallback } from 'react';

export interface ScamAlert {
  id: string;
  level: 'safe' | 'warning' | 'critical';
  reason: string;
  confidence: number;
  timestamp: Date;
  phoneNumber: string;
  contactName?: string;
  features?: {
    linguistic?: {
      authority: number;
      urgency: number;
      threat: number;
      bait: number;
      sensitivity: number;
      repetition: number;
      languageSwitching: number;
    };
    conversational?: {
      turnTaking: number;
      pauseLength: number;
      speechRate: number;
    };
    agnostic?: {
      backgroundNoise: number;
      energySpikes: number;
      pitchRaising: number;
    };
  };
}

interface UseScamAlertsReturn {
  alerts: ScamAlert[];
  isConnected: boolean;
  connectionError: string | null;
  clearAlert: (alertId: string) => void;
  clearAllAlerts: () => void;
  getActiveAlerts: () => ScamAlert[];
  rollingScores: Array<{ t: number; score: number; level: string; callId?: string }>;
  transcripts: string[];
  pushAlert: (alertData: Omit<ScamAlert, 'id' | 'timestamp'>) => void;
}

/**
 * Manages scam alerts locally. Alerts can be pushed in from the call analysis
 * result via `pushAlert`. The Socket.IO dependency has been removed because the
 * backend exposes per-job FastAPI WebSocket endpoints (/ws/status/{job_id}),
 * not a persistent Socket.IO server. Real-time WebSocket updates are handled
 * directly by the CallAnalysis component.
 */
export const useScamAlerts = (): UseScamAlertsReturn => {
  const [alerts, setAlerts] = useState<ScamAlert[]>([]);
  const [rollingScores, setRollingScores] = useState<Array<{ t: number; score: number; level: string; callId?: string }>>([]);
  const [transcripts, setTranscripts] = useState<string[]>([]);

  const pushAlert = useCallback((alertData: Omit<ScamAlert, 'id' | 'timestamp'>) => {
    const newAlert: ScamAlert = {
      ...alertData,
      id: Date.now().toString(),
      timestamp: new Date(),
    };
    setAlerts(prev => [newAlert, ...prev]);

    // Track rolling score
    const score = newAlert.confidence ?? 0;
    setRollingScores(prev => [
      ...prev.slice(-199),
      { t: Date.now(), score, level: newAlert.level },
    ]);

    // Auto-remove after 30 seconds
    setTimeout(() => {
      setAlerts(prev => prev.filter(a => a.id !== newAlert.id));
    }, 30000);
  }, []);

  const clearAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  }, []);

  const clearAllAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  const getActiveAlerts = useCallback(() => {
    return alerts.filter(alert => {
      const timeDiff = Date.now() - new Date(alert.timestamp).getTime();
      return timeDiff < 30000;
    });
  }, [alerts]);

  return {
    alerts,
    // No persistent socket connection — always report as not connected.
    // Individual job WebSocket connections are managed by CallAnalysis component.
    isConnected: false,
    connectionError: null,
    clearAlert,
    clearAllAlerts,
    getActiveAlerts,
    rollingScores,
    transcripts,
    pushAlert,
  };
};
