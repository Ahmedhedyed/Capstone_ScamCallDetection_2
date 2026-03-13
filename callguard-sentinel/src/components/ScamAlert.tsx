import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, X, Shield, AlertCircle } from 'lucide-react';
import { ScamAlert as ScamAlertType } from '@/hooks/useScamAlerts';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface ScamAlertProps {
  alert: ScamAlertType;
  onDismiss: (alertId: string) => void;
}

const ScamAlert: React.FC<ScamAlertProps> = ({ alert, onDismiss }) => {
  const getAlertIcon = () => {
    switch (alert.level) {
      case 'critical':
        return <AlertTriangle className="h-6 w-6 text-alert-critical" />;
      case 'warning':
        return <AlertCircle className="h-6 w-6 text-alert-warning" />;
      case 'safe':
        return <Shield className="h-6 w-6 text-alert-safe" />;
      default:
        return <AlertTriangle className="h-6 w-6 text-alert-critical" />;
    }
  };

  const getAlertColor = () => {
    switch (alert.level) {
      case 'critical':
        return 'border-alert-critical bg-gradient-to-r from-red-900/20 to-red-800/10';
      case 'warning':
        return 'border-alert-warning bg-gradient-to-r from-yellow-900/20 to-yellow-800/10';
      case 'safe':
        return 'border-alert-safe bg-gradient-to-r from-green-900/20 to-green-800/10';
      default:
        return 'border-alert-critical bg-gradient-to-r from-red-900/20 to-red-800/10';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-alert-critical';
    if (confidence >= 0.6) return 'text-alert-warning';
    return 'text-alert-safe';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -50, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -50, scale: 0.9 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="fixed top-4 left-4 right-4 z-50 max-w-md mx-auto"
    >
      <Card className={`${getAlertColor()} border-2 shadow-elevated`}>
        <div className="p-4">
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3">
              {getAlertIcon()}
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  <h3 className="font-semibold text-foreground">
                    {alert.level === 'critical' ? 'SCAM DETECTED' : 
                     alert.level === 'warning' ? 'SUSPICIOUS CALL' : 'CALL VERIFIED'}
                  </h3>
                  <Badge 
                    variant="outline" 
                    className={`${getConfidenceColor(alert.confidence)} border-current`}
                  >
                    {Math.round(alert.confidence * 100)}%
                  </Badge>
                </div>
                
                <p className="text-sm text-muted-foreground mb-2">
                  {alert.contactName || alert.phoneNumber}
                </p>
                
                <p className="text-sm text-foreground mb-3">
                  {alert.reason}
                </p>

                {alert.features && (
                  <div className="space-y-2 text-xs">
                    {alert.features.linguistic && (
                      <div className="grid grid-cols-2 gap-2">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Authority:</span>
                          <span className={alert.features.linguistic.authority > 0.7 ? 'text-alert-critical' : 'text-foreground'}>
                            {Math.round(alert.features.linguistic.authority * 100)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Urgency:</span>
                          <span className={alert.features.linguistic.urgency > 0.7 ? 'text-alert-critical' : 'text-foreground'}>
                            {Math.round(alert.features.linguistic.urgency * 100)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Threat:</span>
                          <span className={alert.features.linguistic.threat > 0.7 ? 'text-alert-critical' : 'text-foreground'}>
                            {Math.round(alert.features.linguistic.threat * 100)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Repetition:</span>
                          <span className={alert.features.linguistic.repetition > 0.7 ? 'text-alert-critical' : 'text-foreground'}>
                            {Math.round(alert.features.linguistic.repetition * 100)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onDismiss(alert.id)}
              className="text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

interface ScamAlertContainerProps {
  alerts: ScamAlertType[];
  onDismiss: (alertId: string) => void;
}

export const ScamAlertContainer: React.FC<ScamAlertContainerProps> = ({ alerts, onDismiss }) => {
  return (
    <div className="fixed top-0 left-0 right-0 z-50 pointer-events-none">
      <div className="pointer-events-auto">
        <AnimatePresence>
          {alerts.map((alert) => (
            <ScamAlert
              key={alert.id}
              alert={alert}
              onDismiss={onDismiss}
            />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
};
