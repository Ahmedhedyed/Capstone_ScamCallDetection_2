import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { CheckCircle, XCircle, Clock, AlertTriangle, Shield, Brain, Zap, Target } from 'lucide-react';

interface AnalysisStatus {
  status: 'initializing' | 'transcribing' | 'analyzing' | 'verifying' | 'complete' | 'error';
  message?: string;
  step?: string;
  chunk_number?: number;
  total_chunks?: number;
  text?: string;
  data?: Record<string, unknown>;
  result?: AnalysisResult;
}

interface AnalysisResult {
  fraud_score: number;
  is_fraud: boolean;
  confidence: string;
  explanation?: string;
  features?: Record<string, number>;
}

interface CallAnalysisProps {
  jobId: string;
  analysisResult?: AnalysisResult; // Direct result from /analyze/fast/ endpoint
  onAnalysisComplete?: (result: AnalysisResult) => void;
  className?: string;
}

export const CallAnalysis: React.FC<CallAnalysisProps> = ({ 
  jobId, 
  analysisResult,
  onAnalysisComplete,
  className = ""
}) => {
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus | null>(null);

  useEffect(() => {
    if (jobId === 'direct-result' && analysisResult) {
      // Handle direct result from /analyze/full/ endpoint
      const status: AnalysisStatus = {
        status: 'complete',
        result: analysisResult
      };
      setAnalysisStatus(status);
      
      if (onAnalysisComplete) {
        onAnalysisComplete(analysisResult);
      }
    } else if (jobId && jobId !== 'direct-result') {
      // Handle WebSocket-based analysis (legacy support)
      const backendUrl = (import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000').replace(/^http/, 'ws');
      const ws = new WebSocket(`${backendUrl}/ws/status/${jobId}`);
      
      ws.onopen = () => {
        console.log('WebSocket connected for analysis:', jobId);
      };
      
      ws.onmessage = (event) => {
        const status: AnalysisStatus = JSON.parse(event.data);
        setAnalysisStatus(status);
        console.log('Analysis status:', status);
        
        // Call completion callback
        if (status.status === 'complete' && status.result && onAnalysisComplete) {
          onAnalysisComplete(status.result);
        }
        
        if (status.status === 'complete' || status.status === 'error') {
          ws.close();
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      return () => {
        if (ws) {
          ws.close();
        }
      };
    }
  }, [jobId, analysisResult, onAnalysisComplete]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'complete':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'initializing':
      case 'transcribing':
      case 'analyzing':
      case 'verifying':
        return <Clock className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'complete':
        return 'bg-green-100 text-green-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      case 'initializing':
      case 'transcribing':
      case 'analyzing':
      case 'verifying':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'medium':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default:
        return <CheckCircle className="h-4 w-4 text-green-500" />;
    }
  };

  if (!analysisStatus) {
    return (
      <Card className={`bg-gradient-card border-border ${className}`}>
        <CardContent className="p-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/20 rounded-full">
              <Clock className="h-5 w-5 text-blue-400 animate-spin" />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">Connecting to AI Analysis</p>
              <p className="text-xs text-muted-foreground">Initializing scam detection...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`bg-gradient-card border-border shadow-elevated ${className}`}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-3 text-xl">
          <div className="p-2 bg-purple-500/20 rounded-full">
            <Brain className="h-5 w-5 text-purple-400" />
          </div>
          AI Call Analysis
        </CardTitle>
        <CardDescription className="text-muted-foreground">
          Fast AI analysis powered by optimized machine learning
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Status Badge */}
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Status</span>
          <Badge className={getStatusColor(analysisStatus.status)}>
            {analysisStatus.status}
          </Badge>
        </div>

        {/* Progress Bar */}
        {(analysisStatus.status === 'transcribing' || analysisStatus.status === 'analyzing' || analysisStatus.status === 'verifying') && (
          <div className="space-y-2">
            {analysisStatus.status === 'transcribing' && analysisStatus.total_chunks && (
              <div className="flex justify-between text-sm">
                <span>Transcription Progress</span>
                <span>{analysisStatus.chunk_number || 0}/{analysisStatus.total_chunks}</span>
              </div>
            )}
            {analysisStatus.status === 'transcribing' && analysisStatus.total_chunks && (
              <Progress 
                value={((analysisStatus.chunk_number || 0) / analysisStatus.total_chunks) * 100} 
                className="w-full" 
              />
            )}
            <p className="text-sm text-gray-600">
              {analysisStatus.message || 'Processing...'}
            </p>
            {analysisStatus.text && (
              <div className="bg-gray-50 p-2 rounded text-sm italic">
                "{analysisStatus.text}"
              </div>
            )}
          </div>
        )}

        {/* Results */}
        {analysisStatus.status === 'complete' && analysisStatus.result && (
          <div className="space-y-6">
            {/* Main Analysis Result Card */}
            <Card className={`${analysisStatus.result.is_fraud ? 'bg-gradient-danger border-red-500/20' : 'bg-gradient-card border-green-500/20'} border-2 shadow-elevated`}>
              <CardHeader className="pb-4">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-full ${analysisStatus.result.is_fraud ? 'bg-red-500/20' : 'bg-green-500/20'}`}>
                    {analysisStatus.result.is_fraud ? (
                      <AlertTriangle className="h-6 w-6 text-red-400" />
                    ) : (
                      <Shield className="h-6 w-6 text-green-400" />
                    )}
                  </div>
                  <div>
                    <CardTitle className={`text-xl ${analysisStatus.result.is_fraud ? 'text-red-100' : 'text-green-100'}`}>
                      {analysisStatus.result.is_fraud ? 'SCAM DETECTED' : 'CALL SAFE'}
                    </CardTitle>
                    <CardDescription className={analysisStatus.result.is_fraud ? 'text-red-200' : 'text-green-200'}>
                      AI Analysis Complete
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Fraud Score with Visual Indicator */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-foreground">Fraud Risk Score</span>
                    <span className="text-2xl font-bold text-foreground">
                      {(analysisStatus.result.fraud_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div 
                      className={`h-3 rounded-full transition-all duration-500 ${
                        analysisStatus.result.fraud_score > 0.7 
                          ? 'bg-gradient-to-r from-red-500 to-red-600' 
                          : analysisStatus.result.fraud_score > 0.3 
                          ? 'bg-gradient-to-r from-yellow-500 to-orange-500' 
                          : 'bg-gradient-to-r from-green-500 to-green-600'
                      }`}
                      style={{ width: `${analysisStatus.result.fraud_score * 100}%` }}
                    />
                  </div>
                </div>

                {/* Confidence Level */}
                <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-blue-400" />
                    <span className="text-sm font-medium text-foreground">Confidence Level</span>
                  </div>
                  <Badge 
                    variant={analysisStatus.result.confidence === 'high' ? 'default' : 'secondary'}
                    className={analysisStatus.result.confidence === 'high' ? 'bg-blue-500 hover:bg-blue-600' : 'bg-gray-600'}
                  >
                    {analysisStatus.result.confidence?.toUpperCase() || 'MEDIUM'}
                  </Badge>
                </div>

                {/* Analysis Explanation */}
                {analysisStatus.result.explanation && (
                  <div className="p-4 bg-gray-800/30 rounded-lg border border-gray-700/50">
                    <div className="flex items-center gap-2 mb-2">
                      <Brain className="h-4 w-4 text-purple-400" />
                      <span className="text-sm font-medium text-foreground">AI Analysis</span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {analysisStatus.result.explanation}
                    </p>
                  </div>
                )}

                {/* Features Breakdown */}
                {analysisStatus.result.features && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 mb-3">
                      <Zap className="h-4 w-4 text-yellow-400" />
                      <span className="text-sm font-medium text-foreground">Detected Features</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(analysisStatus.result.features).map(([feature, score]) => (
                        <div key={feature} className="flex items-center justify-between p-2 bg-gray-800/30 rounded text-xs">
                          <span className="text-muted-foreground capitalize">
                            {feature.replace(/_/g, ' ')}
                          </span>
                          <span className="text-foreground font-medium">
                            {((score as number) * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {/* Error State */}
        {analysisStatus.status === 'error' && (
          <Card className="bg-gradient-danger border-red-500/20 border-2">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-red-500/20 rounded-full">
                  <XCircle className="h-5 w-5 text-red-400" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-red-100">Analysis Failed</h4>
                  <p className="text-xs text-red-200 mt-1">
                    {analysisStatus.message || 'Unknown error occurred'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </CardContent>
    </Card>
  );
};
