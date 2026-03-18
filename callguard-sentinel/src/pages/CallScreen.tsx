import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { PhoneOff, Mic, MicOff, Volume2, VolumeX, AlertTriangle, Shield, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useScamAlerts } from "@/hooks/useScamAlerts";
import { useContacts } from "@/hooks/useContacts";
import { ScamAlertContainer } from "@/components/ScamAlert";
import { motion, AnimatePresence } from "framer-motion";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { analyzeCallFast } from "@/lib/api";
import { Waveform } from "@/components/Waveform";
import { useMicLevels } from "@/hooks/useMicLevels";
import { CallAnalysis } from "@/components/CallAnalysis";

type ThreatLevel = "safe" | "warning" | "critical";

const CallScreen = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const phoneNumber = searchParams.get("number") || "";
  
  const [callDuration, setCallDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [isSpeakerOn, setIsSpeakerOn] = useState(false);
  const [detectorOn, setDetectorOn] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [threatLevel, setThreatLevel] = useState<ThreatLevel>("safe");
  const [scamReasons, setScamReasons] = useState<string[]>([]);
  const [callId, setCallId] = useState<string>("");
  const [isConnected, setIsConnected] = useState(false);
  const [callEnded, setCallEnded] = useState(false);
  const [analysisJobId, setAnalysisJobId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [durationTimer, setDurationTimer] = useState<NodeJS.Timeout | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  const [recordingPromise, setRecordingPromise] = useState<Promise<Blob> | null>(null);
  
  // Hooks
  const { alerts, clearAlert, isConnected: alertConnected, rollingScores, transcripts } = useScamAlerts();
  const waveformValuesFromModel = rollingScores.map(s => Math.min(1, Math.max(0.03, s.score)));
  const { levels: micLevels } = useMicLevels(true);
  const combinedWave = micLevels.length > 0 ? micLevels : waveformValuesFromModel;
  const { getContactByNumber } = useContacts();
  
  // Get contact info
  const currentContact = getContactByNumber(phoneNumber);
  const isKnownScam = currentContact?.isScam || false;

  useEffect(() => {
    const initializeCall = async () => {
      // Start call
      await startCall();
      setIsConnected(true);

      // Start recording automatically
      console.log('Attempting to start recording...');
      await startRecording();

      // Simulate call duration
      const timer = setInterval(() => {
        setCallDuration(prev => prev + 1);
      }, 1000);
      setDurationTimer(timer);

      // Check if this is a known scam number
      if (isKnownScam) {
        setThreatLevel("critical");
        setScamReasons(["Known scam number in database"]);
        toast.error("⚠️ CRITICAL: Known scam number detected!", {
          description: "This number has been flagged as a scam. Consider ending the call immediately.",
        });
      } else {
        // Simulate scam detection after 5 seconds
        setTimeout(() => {
          if (detectorOn) simulateScamDetection();
        }, 5000);
      }
    };

    initializeCall().catch(error => {
      console.error('Error initializing call:', error);
      toast.error("Failed to initialize call");
    });

    return () => {
      // Cleanup timer
      if (durationTimer) {
        clearInterval(durationTimer);
      }
      // Cleanup media recorder
      if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
      }
    };
  }, [isKnownScam]);

  // Cleanup effect for component unmount
  useEffect(() => {
    return () => {
      // Stop any active recording
      if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
      }
      // Clear duration timer
      if (durationTimer) {
        clearInterval(durationTimer);
      }
    };
  }, [mediaRecorder, isRecording, durationTimer]);

  const startCall = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const { data, error } = await supabase
        .from("calls")
        .insert({
          user_id: user.id,
          phone_number: phoneNumber,
          call_status: "active",
        })
        .select()
        .single();

      if (error) throw error;
      if (data) setCallId(data.id);
    } catch (error) {
      console.error("Error starting call:", error);
    }
  };

  const startRecording = async () => {
    if (isRecording) {
      console.log('Already recording, skipping...');
      return; // Already recording
    }
    
    try {
      console.log('Requesting microphone access...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: true, 
        video: false 
      });
      console.log('Microphone access granted, stream:', stream);
      
      // Try to use MP4 format, fallback to webm if not supported
      let mimeType = 'audio/mp4';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm;codecs=opus';
        console.log('MP4 not supported, using WebM');
      }
      console.log('Using MIME type:', mimeType);
      
      const recorder = new MediaRecorder(stream, {
        mimeType: mimeType
      });
      console.log('MediaRecorder created:', recorder);
      
      const chunks: Blob[] = [];
      
      // Create a promise that resolves when recording stops
      const recordingPromise = new Promise<Blob>((resolve) => {
        recorder.ondataavailable = (event) => {
          console.log('Data available event:', event.data.size, 'bytes');
          if (event.data.size > 0) {
            chunks.push(event.data);
          }
        };
        
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: mimeType });
          setRecordedBlob(blob);
          stream.getTracks().forEach(track => track.stop());
          console.log('Recording stopped, blob created:', blob.size, 'bytes');
          resolve(blob);
        };
      });
      
      console.log('Starting recorder...');
      recorder.start(1000); // Collect data every second
      setMediaRecorder(recorder);
      setRecordingPromise(recordingPromise);
      setIsRecording(true);
      console.log('Recording started successfully');
      toast.success("Recording started");
      
    } catch (error) {
      console.error('Error starting recording:', error);
      toast.error(`Error accessing microphone: ${(error as Error)?.message || 'Unknown error'}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setMediaRecorder(null);
      setIsRecording(false);
      console.log('Recording stop requested');
    }
  };

  const toggleRecording = async () => {
    if (!isRecording) {
      await startRecording();
    } else {
      stopRecording();
    }
  };

  const toggleDetector = () => {
    setDetectorOn((prev) => !prev);
    if (!detectorOn) {
      toast.success("Scam detector enabled");
    } else {
      toast("Scam detector paused");
    }
  };

  const blockCurrentNumber = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user || !phoneNumber) return;
      
      // For now, we'll mark the contact as a scam instead of using a blocked_numbers table
      // This will prevent future calls from this number
      await supabase
        .from("contacts")
        .upsert({
          user_id: user.id,
          phone_number: phoneNumber,
          name: `Blocked: ${phoneNumber}`,
          scam_risk_level: "critical",
          scam_score: 1.0,
          updated_at: new Date().toISOString()
        });
      
      toast.success("Number blocked and marked as scam");
    } catch (e) {
      // best-effort UI feedback
      toast.error("Failed to block number");
    }
  };

  const simulateScamDetection = () => {
    // Simulate random scam detection
    const random = Math.random();
    if (random > 0.7) {
      setThreatLevel("critical");
      setScamReasons(["Authority indicators detected", "Urgency patterns identified", "Request for sensitive information"]);
      toast.error("⚠️ CRITICAL SCAM DETECTED!", {
        description: "This call shows multiple scam indicators. Consider ending the call.",
      });
    } else if (random > 0.4) {
      setThreatLevel("warning");
      setScamReasons(["Suspicious language patterns", "Background noise anomalies"]);
      toast.warning("⚠️ Warning: Potential scam detected", {
        description: "Be cautious about sharing personal information.",
      });
    }
  };

  const startCallAnalysis = async (audioBlob?: Blob) => {
    try {
      setIsAnalyzing(true);
      
      const blobToAnalyze = audioBlob || recordedBlob;
      if (!blobToAnalyze) {
        toast.error("No recording available for analysis");
        setIsAnalyzing(false);
        return;
      }
      
      const fileExtension = blobToAnalyze.type.includes('mp4') ? 'mp4' : 'webm';
      const filename = `call_recording.${fileExtension}`;
      console.log('Sending audio for analysis:', blobToAnalyze.size, 'bytes, type:', blobToAnalyze.type);

      const result = await analyzeCallFast(blobToAnalyze, filename);
      console.log('Analysis result:', result);
      
      // The /analyze/fast/ endpoint returns the result directly, not a job_id
      setAnalysisJobId('direct-result');
      setAnalysisResult(result);
      toast.success("Call analysis completed (fast mode)");
      
      // Update threat level based on analysis result
      if (result.is_fraud) {
        setThreatLevel("critical");
        setScamReasons([result.explanation || "AI analysis detected fraud indicators"]);
      } else if (result.fraud_score > 0.3) {
        setThreatLevel("warning");
        setScamReasons(["AI analysis detected suspicious patterns"]);
      }
      
      setIsAnalyzing(false);
      
    } catch (error) {
      console.error('Error starting analysis:', error);
      toast.error("Failed to analyze call");
      setIsAnalyzing(false);
    }
  };

  const endCall = async () => {
    try {
      // Stop the duration timer immediately
      if (durationTimer) {
        clearInterval(durationTimer);
        setDurationTimer(null);
      }

      // Stop recording if active and wait for blob to be created
      let finalRecordedBlob = recordedBlob;
      if (mediaRecorder && isRecording && recordingPromise) {
        stopRecording();
        try {
          finalRecordedBlob = await recordingPromise;
          console.log('Got recorded blob from promise:', finalRecordedBlob.size, 'bytes');
        } catch (error) {
          console.error('Error waiting for recording to finish:', error);
        }
      }

      // Mark call as ended in UI immediately
      setCallEnded(true);
      setIsConnected(false);

      // Save to Supabase only if user is logged in and we have a callId
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (user && callId) {
          await supabase
            .from("calls")
            .update({
              duration: callDuration,
              ended_at: new Date().toISOString(),
              scam_detected: threatLevel !== "safe",
              scam_risk_level: threatLevel,
              scam_reasons: scamReasons,
              call_status: "completed",
            })
            .eq("id", callId);

          const { data: stats } = await supabase
            .from("scam_statistics")
            .select("*")
            .eq("user_id", user.id)
            .maybeSingle();

          if (stats) {
            await supabase
              .from("scam_statistics")
              .update({
                total_calls: stats.total_calls + 1,
                scam_calls_blocked: threatLevel === "critical" ? stats.scam_calls_blocked + 1 : stats.scam_calls_blocked,
                warning_calls: threatLevel === "warning" ? stats.warning_calls + 1 : stats.warning_calls,
                safe_calls: threatLevel === "safe" ? stats.safe_calls + 1 : stats.safe_calls,
              })
              .eq("user_id", user.id);
          } else {
            await supabase.from("scam_statistics").insert({
              user_id: user.id,
              total_calls: 1,
              scam_calls_blocked: threatLevel === "critical" ? 1 : 0,
              warning_calls: threatLevel === "warning" ? 1 : 0,
              safe_calls: threatLevel === "safe" ? 1 : 0,
            });
          }
        }
      } catch (dbError) {
        console.warn("Could not save call to database (user may not be logged in):", dbError);
      }

      // Start call analysis if we have recorded audio
      if (finalRecordedBlob) {
        console.log('Starting analysis with recorded audio:', finalRecordedBlob.size, 'bytes');
        await startCallAnalysis(finalRecordedBlob);
      } else {
        console.log('No audio recorded for analysis');
        toast.info("No audio recorded for analysis");
      }
      
    } catch (error) {
      console.error("Error ending call:", error);
      navigate("/dashboard");
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const getThreatColor = () => {
    switch (threatLevel) {
      case "critical":
        return "bg-alert-critical";
      case "warning":
        return "bg-alert-warning";
      default:
        return "bg-alert-safe";
    }
  };

  const getThreatText = () => {
    switch (threatLevel) {
      case "critical":
        return "CRITICAL SCAM DETECTED";
      case "warning":
        return "Warning: Suspicious Activity";
      default:
        return "Call is Safe";
    }
  };

  // If call has ended, show analysis results
  if (callEnded) {
    return (
      <div className="min-h-screen bg-gradient-dark flex flex-col items-center justify-center p-6">
        <div className="w-full max-w-2xl space-y-6">
          {/* Call Summary */}
          <Card className={`${threatLevel === "critical" ? "bg-gradient-danger border-red-500/20" : threatLevel === "warning" ? "bg-gradient-warning border-yellow-500/20" : "bg-gradient-card border-green-500/20"} border-2 shadow-elevated p-6`}>
            <div className="text-center">
              <div className="flex items-center justify-center gap-3 mb-4">
                <div className={`p-3 rounded-full ${threatLevel === "critical" ? "bg-red-500/20" : threatLevel === "warning" ? "bg-yellow-500/20" : "bg-green-500/20"}`}>
                  {threatLevel === "critical" ? (
                    <AlertTriangle className="h-8 w-8 text-red-400" />
                  ) : threatLevel === "warning" ? (
                    <AlertTriangle className="h-8 w-8 text-yellow-400" />
                  ) : (
                    <Shield className="h-8 w-8 text-green-400" />
                  )}
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">Call Ended</h2>
                  <p className="text-lg text-muted-foreground">{phoneNumber}</p>
                </div>
              </div>
              <div className="flex items-center justify-center gap-4">
                <Badge variant="outline" className="text-sm bg-gray-800/50 border-gray-600 text-foreground">
                  <Clock className="h-3 w-3 mr-1" />
                  Duration: {formatTime(callDuration)}
                </Badge>
                <Badge 
                  variant={threatLevel === "critical" ? "destructive" : threatLevel === "warning" ? "secondary" : "default"}
                  className={`text-sm font-medium ${
                    threatLevel === "critical" 
                      ? "bg-red-500 hover:bg-red-600" 
                      : threatLevel === "warning" 
                      ? "bg-yellow-500 hover:bg-yellow-600" 
                      : "bg-green-500 hover:bg-green-600"
                  }`}
                >
                  {threatLevel === "critical" ? "🚨 CRITICAL" : threatLevel === "warning" ? "⚠️ WARNING" : "✅ SAFE"}
                </Badge>
              </div>
            </div>
          </Card>

          {/* Analysis Results */}
          {analysisJobId ? (
            <CallAnalysis 
              jobId={analysisJobId}
              analysisResult={analysisResult}
              onAnalysisComplete={(result) => {
                console.log('Call analysis completed:', result);
                setIsAnalyzing(false);
              }}
            />
          ) : isAnalyzing ? (
            <Card className="bg-gradient-card border-border p-6">
              <div className="text-center">
                <Clock className="h-8 w-8 text-blue-500 animate-spin mx-auto mb-4" />
                <p className="text-foreground">Starting fast analysis...</p>
                <p className="text-sm text-muted-foreground mt-2">Optimized for speed</p>
              </div>
            </Card>
          ) : null}

          {/* Action Buttons */}
          <div className="flex gap-4 justify-center">
            <Button 
              onClick={() => navigate("/dashboard")}
              className="bg-primary hover:bg-primary/90"
            >
              Back to Dashboard
            </Button>
            <Button 
              onClick={() => navigate("/dialer")}
              variant="outline"
              className="border-border"
            >
              Make Another Call
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen ${threatLevel === "critical" ? "bg-gradient-danger animate-pulse" : "bg-gradient-dark"} flex flex-col items-center justify-center p-6`}>
      <div className="w-full max-w-md space-y-8">
        {/* Threat Level Indicator */}
        <Card className={`${getThreatColor()} border-none p-6 shadow-elevated`}>
          <div className="text-center text-white">
            <p className="text-2xl font-bold">{getThreatText()}</p>
            <p className="text-sm mt-2 opacity-90">{formatTime(callDuration)}</p>
          </div>
        </Card>

        {/* Phone Number */}
        <div className="text-center">
          <p className="text-4xl font-bold text-foreground mb-2">{phoneNumber}</p>
          <p className="text-muted-foreground">Analyzing call in real-time...</p>
        </div>

        {/* Waveform & Live Transcript */}
        <Card className="bg-gradient-card border-border p-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-semibold text-foreground">Live Audio</p>
            {isRecording ? (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-xs text-red-500 font-medium">RECORDING</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                <span className="text-xs text-gray-500 font-medium">NOT RECORDING</span>
              </div>
            )}
          </div>
          <Waveform values={combinedWave} height={56} className="w-full" />
          
          {/* Debug Info */}
          <div className="mt-2 text-xs text-muted-foreground">
            <div>Duration: {callDuration}s | Recording: {isRecording ? 'Yes' : 'No'} | MediaRecorder: {mediaRecorder ? 'Active' : 'None'}</div>
            {recordedBlob && <div>Audio Blob: {recordedBlob.size} bytes</div>}
          </div>
        </Card>

        {/* Scam Indicators */}
        {scamReasons.length > 0 && (
          <Card className="bg-gradient-card border-border p-4">
            <p className="text-sm font-semibold text-foreground mb-3">Detected Threats:</p>
            <div className="space-y-2">
              {scamReasons.map((reason, index) => (
                <Badge key={index} variant="destructive" className="mr-2">
                  {reason}
                </Badge>
              ))}
            </div>
          </Card>
        )}

        {/* Live Transcript */}
        {transcripts.length > 0 && (
          <Card className="bg-gradient-card border-border p-4">
            <p className="text-sm font-semibold text-foreground mb-3">Live Transcript</p>
            <div className="max-h-40 overflow-y-auto space-y-1 text-sm">
              {transcripts.slice(-10).map((t, i) => (
                <div key={i} className="text-muted-foreground">{t}</div>
              ))}
            </div>
          </Card>
        )}

        {/* Call Controls */}
        <div className="flex justify-center gap-6">
          <Button
            onClick={() => setIsMuted(!isMuted)}
            variant="outline"
            size="icon"
            className="h-14 w-14 rounded-full border-border"
          >
            {isMuted ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
          </Button>
          
          <Button
            onClick={endCall}
            size="icon"
            className="h-16 w-16 rounded-full bg-destructive hover:bg-destructive/90 shadow-glow-red"
          >
            <PhoneOff className="h-6 w-6" />
          </Button>

          <Button
            onClick={() => setIsSpeakerOn(!isSpeakerOn)}
            variant="outline"
            size="icon"
            className="h-14 w-14 rounded-full border-border"
          >
            {isSpeakerOn ? <Volume2 className="h-6 w-6" /> : <VolumeX className="h-6 w-6" />}
          </Button>
        </div>

        {/* Advanced Controls */}
        <div className="flex justify-center gap-3 mt-6">
          <Button onClick={toggleDetector} variant="outline" className="border-border">
            {detectorOn ? "Detector: ON" : "Detector: OFF"}
          </Button>
          <Button onClick={toggleRecording} className={isRecording ? "bg-alert-critical hover:bg-alert-critical/90" : "bg-primary hover:bg-primary/90"}>
            {isRecording ? "Stop Recording" : "Start Recording"}
          </Button>
          <Button onClick={blockCurrentNumber} variant="outline" className="border-border">
            Block Number
          </Button>
        </div>
      </div>
    </div>
  );
};

export default CallScreen;

// import { useState, useEffect } from "react";
// import { useNavigate, useSearchParams } from "react-router-dom";
// import { PhoneOff, Mic, MicOff, Volume2, VolumeX, AlertTriangle, Shield, Clock } from "lucide-react";
// import { Button } from "@/components/ui/button";
// import { Card } from "@/components/ui/card";
// import { Badge } from "@/components/ui/badge";
// import { useScamAlerts } from "@/hooks/useScamAlerts";
// import { useContacts } from "@/hooks/useContacts";
// import { ScamAlertContainer } from "@/components/ScamAlert";
// import { motion, AnimatePresence } from "framer-motion";
// import { supabase } from "@/integrations/supabase/client";
// import { toast } from "sonner";
// import { Waveform } from "@/components/Waveform";
// import { useMicLevels } from "@/hooks/useMicLevels";
// import { CallAnalysis } from "@/components/CallAnalysis";

// type ThreatLevel = "safe" | "warning" | "critical";

// const CallScreen = () => {
//   const navigate = useNavigate();
//   const [searchParams] = useSearchParams();
//   const phoneNumber = searchParams.get("number") || "";
  
//   const [callDuration, setCallDuration] = useState(0);
//   const [isMuted, setIsMuted] = useState(false);
//   const [isSpeakerOn, setIsSpeakerOn] = useState(false);
//   const [detectorOn, setDetectorOn] = useState(true);
//   const [isRecording, setIsRecording] = useState(false);
//   const [threatLevel, setThreatLevel] = useState<ThreatLevel>("safe");
//   const [scamReasons, setScamReasons] = useState<string[]>([]);
//   const [callId, setCallId] = useState<string>("");
//   const [isConnected, setIsConnected] = useState(false);
//   const [callEnded, setCallEnded] = useState(false);
//   const [analysisJobId, setAnalysisJobId] = useState<string | null>(null);
//   const [isAnalyzing, setIsAnalyzing] = useState(false);
//   const [durationTimer, setDurationTimer] = useState<NodeJS.Timeout | null>(null);
//   const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
//   const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
//   const [analysisResult, setAnalysisResult] = useState<any>(null);
//   const [recordingPromise, setRecordingPromise] = useState<Promise<Blob> | null>(null);
  
//   // Hooks
//   const { alerts, clearAlert, isConnected: alertConnected, rollingScores, transcripts } = useScamAlerts();
//   const waveformValuesFromModel = rollingScores.map(s => Math.min(1, Math.max(0.03, s.score)));
//   const { levels: micLevels } = useMicLevels(true);
//   const combinedWave = micLevels.length > 0 ? micLevels : waveformValuesFromModel;
//   const { getContactByNumber } = useContacts();
  
//   // Get contact info
//   const currentContact = getContactByNumber(phoneNumber);
//   const isKnownScam = currentContact?.isScam || false;

//   useEffect(() => {
//     const initializeCall = async () => {
//       // Start call
//       await startCall();
//       setIsConnected(true);

//       // Start recording automatically
//       console.log('Attempting to start recording...');
//       await startRecording();

//       // Simulate call duration
//       const timer = setInterval(() => {
//         setCallDuration(prev => prev + 1);
//       }, 1000);
//       setDurationTimer(timer);

//       // Check if this is a known scam number
//       if (isKnownScam) {
//         setThreatLevel("critical");
//         setScamReasons(["Known scam number in database"]);
//         toast.error("⚠️ CRITICAL: Known scam number detected!", {
//           description: "This number has been flagged as a scam. Consider ending the call immediately.",
//         });
//       } else {
//         // Simulate scam detection after 5 seconds
//         setTimeout(() => {
//           if (detectorOn) simulateScamDetection();
//         }, 5000);
//       }
//     };

//     initializeCall().catch(error => {
//       console.error('Error initializing call:', error);
//       toast.error("Failed to initialize call");
//     });

//     return () => {
//       // Cleanup timer
//       if (durationTimer) {
//         clearInterval(durationTimer);
//       }
//       // Cleanup media recorder
//       if (mediaRecorder && isRecording) {
//         mediaRecorder.stop();
//       }
//     };
//   }, [isKnownScam]);

//   // Cleanup effect for component unmount
//   useEffect(() => {
//     return () => {
//       // Stop any active recording
//       if (mediaRecorder && isRecording) {
//         mediaRecorder.stop();
//       }
//       // Clear duration timer
//       if (durationTimer) {
//         clearInterval(durationTimer);
//       }
//     };
//   }, [mediaRecorder, isRecording, durationTimer]);

//   const startCall = async () => {
//     try {
//       const { data: { user } } = await supabase.auth.getUser();
//       if (!user) return;

//       const { data, error } = await supabase
//         .from("calls")
//         .insert({
//           user_id: user.id,
//           phone_number: phoneNumber,
//           call_status: "active",
//         })
//         .select()
//         .single();

//       if (error) throw error;
//       if (data) setCallId(data.id);
//     } catch (error) {
//       console.error("Error starting call:", error);
//     }
//   };

//   const startRecording = async () => {
//     if (isRecording) {
//       console.log('Already recording, skipping...');
//       return; // Already recording
//     }
    
//     try {
//       console.log('Requesting microphone access...');
//       const stream = await navigator.mediaDevices.getUserMedia({ 
//         audio: true, 
//         video: false 
//       });
//       console.log('Microphone access granted, stream:', stream);
      
//       // Try to use MP4 format, fallback to webm if not supported
//       let mimeType = 'audio/mp4';
//       if (!MediaRecorder.isTypeSupported(mimeType)) {
//         mimeType = 'audio/webm;codecs=opus';
//         console.log('MP4 not supported, using WebM');
//       }
//       console.log('Using MIME type:', mimeType);
      
//       const recorder = new MediaRecorder(stream, {
//         mimeType: mimeType
//       });
//       console.log('MediaRecorder created:', recorder);
      
//       const chunks: Blob[] = [];
      
//       // Create a promise that resolves when recording stops
//       const recordingPromise = new Promise<Blob>((resolve) => {
//         recorder.ondataavailable = (event) => {
//           console.log('Data available event:', event.data.size, 'bytes');
//           if (event.data.size > 0) {
//             chunks.push(event.data);
//           }
//         };
        
//         recorder.onstop = () => {
//           const blob = new Blob(chunks, { type: mimeType });
//           setRecordedBlob(blob);
//           stream.getTracks().forEach(track => track.stop());
//           console.log('Recording stopped, blob created:', blob.size, 'bytes');
//           resolve(blob);
//         };
//       });
      
//       console.log('Starting recorder...');
//       recorder.start(1000); // Collect data every second
//       setMediaRecorder(recorder);
//       setRecordingPromise(recordingPromise);
//       setIsRecording(true);
//       console.log('Recording started successfully');
//       toast.success("Recording started");
      
//     } catch (error) {
//       console.error('Error starting recording:', error);
//       toast.error(`Error accessing microphone: ${(error as Error)?.message || 'Unknown error'}`);
//     }
//   };

//   const stopRecording = () => {
//     if (mediaRecorder && isRecording) {
//       mediaRecorder.stop();
//       setMediaRecorder(null);
//       setIsRecording(false);
//       console.log('Recording stop requested');
//     }
//   };

//   const toggleRecording = async () => {
//     if (!isRecording) {
//       await startRecording();
//     } else {
//       stopRecording();
//     }
//   };

//   const toggleDetector = () => {
//     setDetectorOn((prev) => !prev);
//     if (!detectorOn) {
//       toast.success("Scam detector enabled");
//     } else {
//       toast("Scam detector paused");
//     }
//   };

//   const blockCurrentNumber = async () => {
//     try {
//       const { data: { user } } = await supabase.auth.getUser();
//       if (!user || !phoneNumber) return;
      
//       // For now, we'll mark the contact as a scam instead of using a blocked_numbers table
//       // This will prevent future calls from this number
//       await supabase
//         .from("contacts")
//         .upsert({
//           user_id: user.id,
//           phone_number: phoneNumber,
//           name: `Blocked: ${phoneNumber}`,
//           scam_risk_level: "critical",
//           scam_score: 1.0,
//           updated_at: new Date().toISOString()
//         });
      
//       toast.success("Number blocked and marked as scam");
//     } catch (e) {
//       // best-effort UI feedback
//       toast.error("Failed to block number");
//     }
//   };

//   const simulateScamDetection = () => {
//     // Simulate random scam detection
//     const random = Math.random();
//     if (random > 0.7) {
//       setThreatLevel("critical");
//       setScamReasons(["Authority indicators detected", "Urgency patterns identified", "Request for sensitive information"]);
//       toast.error("⚠️ CRITICAL SCAM DETECTED!", {
//         description: "This call shows multiple scam indicators. Consider ending the call.",
//       });
//     } else if (random > 0.4) {
//       setThreatLevel("warning");
//       setScamReasons(["Suspicious language patterns", "Background noise anomalies"]);
//       toast.warning("⚠️ Warning: Potential scam detected", {
//         description: "Be cautious about sharing personal information.",
//       });
//     }
//   };

//   const startCallAnalysis = async (audioBlob?: Blob) => {
//     try {
//       setIsAnalyzing(true);
      
//       const blobToAnalyze = audioBlob || recordedBlob;
//       if (!blobToAnalyze) {
//         toast.error("No recording available for analysis");
//         setIsAnalyzing(false);
//         return;
//       }
      
//       // Determine file extension based on MIME type
//       const fileExtension = blobToAnalyze.type.includes('mp4') ? 'mp4' : 'webm';
//       const formData = new FormData();
//       formData.append('file', blobToAnalyze, `call_recording.${fileExtension}`);
      
//       console.log('Sending audio for analysis:', blobToAnalyze.size, 'bytes, type:', blobToAnalyze.type);
      
//       const response = await fetch('http://localhost:8001/analyze/fast/', {
//         method: 'POST',
//         body: formData,
//       });
      
//       if (!response.ok) {
//         throw new Error('Analysis request failed');
//       }
      
//       const result = await response.json();
//       console.log('Analysis result:', result);
      
//       // The /analyze/fast/ endpoint returns the result directly, not a job_id
//       setAnalysisJobId('direct-result');
//       setAnalysisResult(result);
//       toast.success("Call analysis completed (fast mode)");
      
//       // Update threat level based on analysis result
//       if (result.is_fraud) {
//         setThreatLevel("critical");
//         setScamReasons([result.explanation || "AI analysis detected fraud indicators"]);
//       } else if (result.fraud_score > 0.3) {
//         setThreatLevel("warning");
//         setScamReasons(["AI analysis detected suspicious patterns"]);
//       }
      
//       setIsAnalyzing(false);
      
//     } catch (error) {
//       console.error('Error starting analysis:', error);
//       toast.error("Failed to analyze call");
//       setIsAnalyzing(false);
//     }
//   };

//   const endCall = async () => {
//     try {
//       const { data: { user } } = await supabase.auth.getUser();
//       if (!user || !callId) return;

//       // Stop the duration timer immediately
//       if (durationTimer) {
//         clearInterval(durationTimer);
//         setDurationTimer(null);
//       }

//       // Stop recording if active and wait for blob to be created
//       let finalRecordedBlob = recordedBlob;
//       if (mediaRecorder && isRecording && recordingPromise) {
//         stopRecording();
        
//         // Wait for the recording to finish and blob to be created
//         try {
//           finalRecordedBlob = await recordingPromise;
//           console.log('Got recorded blob from promise:', finalRecordedBlob.size, 'bytes');
//         } catch (error) {
//           console.error('Error waiting for recording to finish:', error);
//         }
//       }

//       // Mark call as ended
//       setCallEnded(true);
//       setIsConnected(false);

//       await supabase
//         .from("calls")
//         .update({
//           duration: callDuration,
//           ended_at: new Date().toISOString(),
//           scam_detected: threatLevel !== "safe",
//           scam_risk_level: threatLevel,
//           scam_reasons: scamReasons,
//           call_status: "completed",
//         })
//         .eq("id", callId);

//       // Update statistics
//       const { data: stats } = await supabase
//         .from("scam_statistics")
//         .select("*")
//         .eq("user_id", user.id)
//         .single();

//       if (stats) {
//         await supabase
//           .from("scam_statistics")
//           .update({
//             total_calls: stats.total_calls + 1,
//             scam_calls_blocked: threatLevel === "critical" ? stats.scam_calls_blocked + 1 : stats.scam_calls_blocked,
//             warning_calls: threatLevel === "warning" ? stats.warning_calls + 1 : stats.warning_calls,
//             safe_calls: threatLevel === "safe" ? stats.safe_calls + 1 : stats.safe_calls,
//           })
//           .eq("user_id", user.id);
//       } else {
//         await supabase.from("scam_statistics").insert({
//           user_id: user.id,
//           total_calls: 1,
//           scam_calls_blocked: threatLevel === "critical" ? 1 : 0,
//           warning_calls: threatLevel === "warning" ? 1 : 0,
//           safe_calls: threatLevel === "safe" ? 1 : 0,
//         });
//       }

//       // Start call analysis if we have recorded audio
//       if (finalRecordedBlob) {
//         console.log('Starting analysis with recorded audio:', finalRecordedBlob.size, 'bytes');
//         await startCallAnalysis(finalRecordedBlob);
//       } else {
//         console.log('No audio recorded for analysis');
//         toast.info("No audio recorded for analysis");
//       }
      
//     } catch (error) {
//       console.error("Error ending call:", error);
//       navigate("/dashboard");
//     }
//   };

//   const formatTime = (seconds: number) => {
//     const mins = Math.floor(seconds / 60);
//     const secs = seconds % 60;
//     return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
//   };

//   const getThreatColor = () => {
//     switch (threatLevel) {
//       case "critical":
//         return "bg-alert-critical";
//       case "warning":
//         return "bg-alert-warning";
//       default:
//         return "bg-alert-safe";
//     }
//   };

//   const getThreatText = () => {
//     switch (threatLevel) {
//       case "critical":
//         return "CRITICAL SCAM DETECTED";
//       case "warning":
//         return "Warning: Suspicious Activity";
//       default:
//         return "Call is Safe";
//     }
//   };

//   // If call has ended, show analysis results
//   if (callEnded) {
//     return (
//       <div className="min-h-screen bg-gradient-dark flex flex-col items-center justify-center p-6">
//         <div className="w-full max-w-2xl space-y-6">
//           {/* Call Summary */}
//           <Card className={`${threatLevel === "critical" ? "bg-gradient-danger border-red-500/20" : threatLevel === "warning" ? "bg-gradient-warning border-yellow-500/20" : "bg-gradient-card border-green-500/20"} border-2 shadow-elevated p-6`}>
//             <div className="text-center">
//               <div className="flex items-center justify-center gap-3 mb-4">
//                 <div className={`p-3 rounded-full ${threatLevel === "critical" ? "bg-red-500/20" : threatLevel === "warning" ? "bg-yellow-500/20" : "bg-green-500/20"}`}>
//                   {threatLevel === "critical" ? (
//                     <AlertTriangle className="h-8 w-8 text-red-400" />
//                   ) : threatLevel === "warning" ? (
//                     <AlertTriangle className="h-8 w-8 text-yellow-400" />
//                   ) : (
//                     <Shield className="h-8 w-8 text-green-400" />
//                   )}
//                 </div>
//                 <div>
//                   <h2 className="text-3xl font-bold text-foreground">Call Ended</h2>
//                   <p className="text-lg text-muted-foreground">{phoneNumber}</p>
//                 </div>
//               </div>
//               <div className="flex items-center justify-center gap-4">
//                 <Badge variant="outline" className="text-sm bg-gray-800/50 border-gray-600 text-foreground">
//                   <Clock className="h-3 w-3 mr-1" />
//                   Duration: {formatTime(callDuration)}
//                 </Badge>
//                 <Badge 
//                   variant={threatLevel === "critical" ? "destructive" : threatLevel === "warning" ? "secondary" : "default"}
//                   className={`text-sm font-medium ${
//                     threatLevel === "critical" 
//                       ? "bg-red-500 hover:bg-red-600" 
//                       : threatLevel === "warning" 
//                       ? "bg-yellow-500 hover:bg-yellow-600" 
//                       : "bg-green-500 hover:bg-green-600"
//                   }`}
//                 >
//                   {threatLevel === "critical" ? "🚨 CRITICAL" : threatLevel === "warning" ? "⚠️ WARNING" : "✅ SAFE"}
//                 </Badge>
//               </div>
//             </div>
//           </Card>

//           {/* Analysis Results */}
//           {analysisJobId ? (
//             <CallAnalysis 
//               jobId={analysisJobId}
//               analysisResult={analysisResult}
//               onAnalysisComplete={(result) => {
//                 console.log('Call analysis completed:', result);
//                 setIsAnalyzing(false);
//               }}
//             />
//           ) : isAnalyzing ? (
//             <Card className="bg-gradient-card border-border p-6">
//               <div className="text-center">
//                 <Clock className="h-8 w-8 text-blue-500 animate-spin mx-auto mb-4" />
//                 <p className="text-foreground">Starting fast analysis...</p>
//                 <p className="text-sm text-muted-foreground mt-2">Optimized for speed</p>
//               </div>
//             </Card>
//           ) : null}

//           {/* Action Buttons */}
//           <div className="flex gap-4 justify-center">
//             <Button 
//               onClick={() => navigate("/dashboard")}
//               className="bg-primary hover:bg-primary/90"
//             >
//               Back to Dashboard
//             </Button>
//             <Button 
//               onClick={() => navigate("/dialer")}
//               variant="outline"
//               className="border-border"
//             >
//               Make Another Call
//             </Button>
//           </div>
//         </div>
//       </div>
//     );
//   }

//   return (
//     <div className={`min-h-screen ${threatLevel === "critical" ? "bg-gradient-danger animate-pulse" : "bg-gradient-dark"} flex flex-col items-center justify-center p-6`}>
//       <div className="w-full max-w-md space-y-8">
//         {/* Threat Level Indicator */}
//         <Card className={`${getThreatColor()} border-none p-6 shadow-elevated`}>
//           <div className="text-center text-white">
//             <p className="text-2xl font-bold">{getThreatText()}</p>
//             <p className="text-sm mt-2 opacity-90">{formatTime(callDuration)}</p>
//           </div>
//         </Card>

//         {/* Phone Number */}
//         <div className="text-center">
//           <p className="text-4xl font-bold text-foreground mb-2">{phoneNumber}</p>
//           <p className="text-muted-foreground">Analyzing call in real-time...</p>
//         </div>

//         {/* Waveform & Live Transcript */}
//         <Card className="bg-gradient-card border-border p-4">
//           <div className="flex items-center justify-between mb-2">
//             <p className="text-sm font-semibold text-foreground">Live Audio</p>
//             {isRecording ? (
//               <div className="flex items-center gap-2">
//                 <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
//                 <span className="text-xs text-red-500 font-medium">RECORDING</span>
//               </div>
//             ) : (
//               <div className="flex items-center gap-2">
//                 <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
//                 <span className="text-xs text-gray-500 font-medium">NOT RECORDING</span>
//               </div>
//             )}
//           </div>
//           <Waveform values={combinedWave} height={56} className="w-full" />
          
//           {/* Debug Info */}
//           <div className="mt-2 text-xs text-muted-foreground">
//             <div>Duration: {callDuration}s | Recording: {isRecording ? 'Yes' : 'No'} | MediaRecorder: {mediaRecorder ? 'Active' : 'None'}</div>
//             {recordedBlob && <div>Audio Blob: {recordedBlob.size} bytes</div>}
//           </div>
//         </Card>

//         {/* Scam Indicators */}
//         {scamReasons.length > 0 && (
//           <Card className="bg-gradient-card border-border p-4">
//             <p className="text-sm font-semibold text-foreground mb-3">Detected Threats:</p>
//             <div className="space-y-2">
//               {scamReasons.map((reason, index) => (
//                 <Badge key={index} variant="destructive" className="mr-2">
//                   {reason}
//                 </Badge>
//               ))}
//             </div>
//           </Card>
//         )}

//         {/* Live Transcript */}
//         {transcripts.length > 0 && (
//           <Card className="bg-gradient-card border-border p-4">
//             <p className="text-sm font-semibold text-foreground mb-3">Live Transcript</p>
//             <div className="max-h-40 overflow-y-auto space-y-1 text-sm">
//               {transcripts.slice(-10).map((t, i) => (
//                 <div key={i} className="text-muted-foreground">{t}</div>
//               ))}
//             </div>
//           </Card>
//         )}

//         {/* Call Controls */}
//         <div className="flex justify-center gap-6">
//           <Button
//             onClick={() => setIsMuted(!isMuted)}
//             variant="outline"
//             size="icon"
//             className="h-14 w-14 rounded-full border-border"
//           >
//             {isMuted ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
//           </Button>
          
//           <Button
//             onClick={endCall}
//             size="icon"
//             className="h-16 w-16 rounded-full bg-destructive hover:bg-destructive/90 shadow-glow-red"
//           >
//             <PhoneOff className="h-6 w-6" />
//           </Button>

//           <Button
//             onClick={() => setIsSpeakerOn(!isSpeakerOn)}
//             variant="outline"
//             size="icon"
//             className="h-14 w-14 rounded-full border-border"
//           >
//             {isSpeakerOn ? <Volume2 className="h-6 w-6" /> : <VolumeX className="h-6 w-6" />}
//           </Button>
//         </div>

//         {/* Advanced Controls */}
//         <div className="flex justify-center gap-3 mt-6">
//           <Button onClick={toggleDetector} variant="outline" className="border-border">
//             {detectorOn ? "Detector: ON" : "Detector: OFF"}
//           </Button>
//           <Button onClick={toggleRecording} className={isRecording ? "bg-alert-critical hover:bg-alert-critical/90" : "bg-primary hover:bg-primary/90"}>
//             {isRecording ? "Stop Recording" : "Start Recording"}
//           </Button>
//           <Button onClick={blockCurrentNumber} variant="outline" className="border-border">
//             Block Number
//           </Button>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default CallScreen;
