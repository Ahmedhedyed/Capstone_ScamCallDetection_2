
import { useState, useEffect, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';

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
}

export const useScamAlerts = (): UseScamAlertsReturn => {
  const [alerts, setAlerts] = useState<ScamAlert[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [rollingScores, setRollingScores] = useState<Array<{ t: number; score: number; level: string; callId?: string }>>([]);
  const [transcripts, setTranscripts] = useState<string[]>([]);

  useEffect(() => {
    // Initialize WebSocket connection
    const newSocket = io(import.meta.env.VITE_ALERT_SERVICE_URL || 'http://localhost:3001', {
      transports: ['websocket'],
      timeout: 5000,
      reconnection: true,
      reconnectionAttempts: 3,
      reconnectionDelay: 5000,
    });

    newSocket.on('connect', () => {
      console.log('Connected to alert service');
      setIsConnected(true);
      setConnectionError(null);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from alert service');
      setIsConnected(false);
    });

    newSocket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      setConnectionError(error.message);
      setIsConnected(false);
    });

    newSocket.on('scam_alert', (alertData: Omit<ScamAlert, 'id' | 'timestamp'>) => {
      const newAlert: ScamAlert = {
        ...alertData,
        id: Date.now().toString(),
        timestamp: new Date(),
      };
      
      setAlerts(prev => [newAlert, ...prev]);
      
      // Auto-remove alerts after 30 seconds
      setTimeout(() => {
        setAlerts(prev => prev.filter(alert => alert.id !== newAlert.id));
      }, 30000);
    });

    newSocket.on('call_analysis_update', (data: any) => {
      const score = typeof data?.score === 'number' ? data.score : 0;
      const level = data?.level || 'safe';
      setRollingScores(prev => [...prev.slice(-199), { t: Date.now(), score, level, callId: data?.call_id }]);
    });

    newSocket.on('transcript', (payload: { text: string }) => {
      if (payload?.text) setTranscripts(prev => [...prev.slice(-50), payload.text]);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const clearAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  }, []);

  const clearAllAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  const getActiveAlerts = useCallback(() => {
    return alerts.filter(alert => {
      const now = new Date();
      const alertTime = new Date(alert.timestamp);
      const timeDiff = now.getTime() - alertTime.getTime();
      return timeDiff < 30000; // Active for 30 seconds
    });
  }, [alerts]);

  return {
    alerts,
    isConnected,
    connectionError,
    clearAlert,
    clearAllAlerts,
    getActiveAlerts,
    rollingScores,
    transcripts,
  };
};


// import { useState, useEffect, useCallback } from 'react';
// import { io, Socket } from 'socket.io-client';

// export interface ScamAlert {
//   id: string;
//   level: 'safe' | 'warning' | 'critical';
//   reason: string;
//   confidence: number;
//   timestamp: Date;
//   phoneNumber: string;
//   contactName?: string;
//   features?: {
//     linguistic?: {
//       authority: number;
//       urgency: number;
//       threat: number;
//       bait: number;
//       sensitivity: number;
//       repetition: number;
//       languageSwitching: number;
//     };
//     conversational?: {
//       turnTaking: number;
//       pauseLength: number;
//       speechRate: number;
//     };
//     agnostic?: {
//       backgroundNoise: number;
//       energySpikes: number;
//       pitchRaising: number;
//     };
//   };
// }

// interface UseScamAlertsReturn {
//   alerts: ScamAlert[];
//   isConnected: boolean;
//   connectionError: string | null;
//   clearAlert: (alertId: string) => void;
//   clearAllAlerts: () => void;
//   getActiveAlerts: () => ScamAlert[];
//   rollingScores: Array<{ t: number; score: number; level: string; callId?: string }>;
//   transcripts: string[];
// }

// export const useScamAlerts = (): UseScamAlertsReturn => {
//   const [alerts, setAlerts] = useState<ScamAlert[]>([]);
//   const [isConnected, setIsConnected] = useState(false);
//   const [connectionError, setConnectionError] = useState<string | null>(null);
//   const [socket, setSocket] = useState<Socket | null>(null);
//   const [rollingScores, setRollingScores] = useState<Array<{ t: number; score: number; level: string; callId?: string }>>([]);
//   const [transcripts, setTranscripts] = useState<string[]>([]);

//   useEffect(() => {
//     // Initialize WebSocket connection
//     const newSocket = io(import.meta.env.VITE_ALERT_SERVICE_URL || 'http://localhost:3001', {
//       transports: ['websocket'],
//       timeout: 20000,
//     });

//     newSocket.on('connect', () => {
//       console.log('Connected to alert service');
//       setIsConnected(true);
//       setConnectionError(null);
//     });

//     newSocket.on('disconnect', () => {
//       console.log('Disconnected from alert service');
//       setIsConnected(false);
//     });

//     newSocket.on('connect_error', (error) => {
//       console.error('Connection error:', error);
//       setConnectionError(error.message);
//       setIsConnected(false);
//     });

//     newSocket.on('scam_alert', (alertData: Omit<ScamAlert, 'id' | 'timestamp'>) => {
//       const newAlert: ScamAlert = {
//         ...alertData,
//         id: Date.now().toString(),
//         timestamp: new Date(),
//       };
      
//       setAlerts(prev => [newAlert, ...prev]);
      
//       // Auto-remove alerts after 30 seconds
//       setTimeout(() => {
//         setAlerts(prev => prev.filter(alert => alert.id !== newAlert.id));
//       }, 30000);
//     });

//     newSocket.on('call_analysis_update', (data: any) => {
//       const score = typeof data?.score === 'number' ? data.score : 0;
//       const level = data?.level || 'safe';
//       setRollingScores(prev => [...prev.slice(-199), { t: Date.now(), score, level, callId: data?.call_id }]);
//     });

//     newSocket.on('transcript', (payload: { text: string }) => {
//       if (payload?.text) setTranscripts(prev => [...prev.slice(-50), payload.text]);
//     });

//     setSocket(newSocket);

//     return () => {
//       newSocket.close();
//     };
//   }, []);

//   const clearAlert = useCallback((alertId: string) => {
//     setAlerts(prev => prev.filter(alert => alert.id !== alertId));
//   }, []);

//   const clearAllAlerts = useCallback(() => {
//     setAlerts([]);
//   }, []);

//   const getActiveAlerts = useCallback(() => {
//     return alerts.filter(alert => {
//       const now = new Date();
//       const alertTime = new Date(alert.timestamp);
//       const timeDiff = now.getTime() - alertTime.getTime();
//       return timeDiff < 30000; // Active for 30 seconds
//     });
//   }, [alerts]);

//   return {
//     alerts,
//     isConnected,
//     connectionError,
//     clearAlert,
//     clearAllAlerts,
//     getActiveAlerts,
//     rollingScores,
//     transcripts,
//   };
// };
