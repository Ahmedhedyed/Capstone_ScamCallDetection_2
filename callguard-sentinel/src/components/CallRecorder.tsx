import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Mic, MicOff, Play, Square, Upload } from 'lucide-react';
import { CallAnalysis } from './CallAnalysis';

export const CallRecorder: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [analysisJobId, setAnalysisJobId] = useState<string | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [analysisStatus, setAnalysisStatus] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Format time for display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: true, 
        video: false 
      });
      
      streamRef.current = stream;
      
      // Use webm/opus (widely supported), fallback to ogg, then mp4
      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = MediaRecorder.isTypeSupported('audio/ogg;codecs=vorbis')
          ? 'audio/ogg;codecs=vorbis'
          : 'audio/mp4';
      }
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: mimeType
      });
      
      mediaRecorderRef.current = mediaRecorder;
      const chunks: Blob[] = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType });
        setRecordedBlob(blob);
        setRecordedChunks(chunks);
      };
      
      mediaRecorder.start(1000); // Collect data every second
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error accessing microphone. Please check permissions.');
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsPaused(false);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    }
  };

  // Pause/Resume recording
  const togglePause = () => {
    if (mediaRecorderRef.current) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
        setIsPaused(false);
        timerRef.current = setInterval(() => {
          setRecordingTime(prev => prev + 1);
        }, 1000);
      } else {
        mediaRecorderRef.current.pause();
        setIsPaused(true);
        if (timerRef.current) {
          clearInterval(timerRef.current);
        }
      }
    }
  };

  // Upload and analyze recording
  const uploadAndAnalyze = async () => {
    if (!recordedBlob) return;
    
    setIsUploading(true);
    
    try {
      // Determine file extension based on MIME type
      const fileExtension = recordedBlob.type.includes('mp4') ? 'mp4' : 'webm';
      const formData = new FormData();
      formData.append('file', recordedBlob, `recording.${fileExtension}`);
      
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'}/analyze/`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      const result = await response.json();
      const jobId = result.job_id;
      
      // Set the analysis job ID to trigger the CallAnalysis component
      setAnalysisJobId(jobId);
      setIsUploading(false);
      
    } catch (error) {
      console.error('Upload error:', error);
      alert('Error uploading recording. Please try again.');
      setIsUploading(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mic className="h-5 w-5" />
            Call Recorder
          </CardTitle>
          <CardDescription>
            Record a call and analyze it for scam detection
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Recording Controls */}
          <div className="flex items-center justify-center gap-4">
            {!isRecording && !recordedBlob && (
              <Button onClick={startRecording} className="flex items-center gap-2">
                <Mic className="h-4 w-4" />
                Start Recording
              </Button>
            )}
            
            {isRecording && (
              <>
                <Button 
                  onClick={togglePause} 
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  {isPaused ? <Play className="h-4 w-4" /> : <MicOff className="h-4 w-4" />}
                  {isPaused ? 'Resume' : 'Pause'}
                </Button>
                <Button onClick={stopRecording} variant="destructive" className="flex items-center gap-2">
                  <Square className="h-4 w-4" />
                  Stop Recording
                </Button>
              </>
            )}
          </div>

          {/* Recording Status */}
          {isRecording && (
            <div className="text-center">
              <div className="text-2xl font-mono mb-2">
                {formatTime(recordingTime)}
              </div>
              <Badge variant="destructive" className="animate-pulse">
                {isPaused ? 'Paused' : 'Recording...'}
              </Badge>
            </div>
          )}

          {/* Upload Section */}
          {recordedBlob && !analysisStatus && (
            <div className="text-center space-y-4">
              <p className="text-sm text-gray-600">
                Recording complete ({formatTime(recordingTime)})
              </p>
              <Button 
                onClick={uploadAndAnalyze} 
                disabled={isUploading}
                className="flex items-center gap-2"
              >
                <Upload className="h-4 w-4" />
                {isUploading ? 'Uploading...' : 'Analyze Recording'}
              </Button>
            </div>
          )}

          {/* Analysis Status */}
          {analysisJobId && (
            <CallAnalysis 
              jobId={analysisJobId}
              onAnalysisComplete={(result) => {
                console.log('Analysis completed:', result);
              }}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
};
