import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { ArrowLeft, Phone, Clock, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatDistanceToNow } from "date-fns";

interface Call {
  id: string;
  phone_number: string;
  duration: number;
  scam_detected: boolean;
  scam_risk_level: string;
  scam_reasons: string[];
  started_at: string;
}

const CallHistory = () => {
  const navigate = useNavigate();
  const [calls, setCalls] = useState<Call[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        navigate("/auth");
        return;
      }

      const { data, error } = await supabase
        .from("calls")
        .select("*")
        .eq("user_id", user.id)
        .order("started_at", { ascending: false })
        .limit(50);

      if (error) throw error;
      setCalls(data || []);
    } catch (error) {
      console.error("Error loading history:", error);
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getRiskBadge = (level: string) => {
    switch (level) {
      case "critical":
        return <Badge variant="destructive">Critical Scam</Badge>;
      case "warning":
        return <Badge className="bg-alert-warning text-white">Warning</Badge>;
      default:
        return <Badge className="bg-alert-safe text-white">Safe</Badge>;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-dark">
      {/* Header */}
      <div className="p-6 flex items-center border-b border-border">
        <Button variant="ghost" onClick={() => navigate("/dashboard")} className="text-foreground">
          <ArrowLeft className="h-5 w-5" />
        </Button>
        <h1 className="text-xl font-semibold text-foreground ml-4">Call History</h1>
      </div>

      <div className="p-6 space-y-4">
        {loading ? (
          <div className="text-center text-muted-foreground">Loading history...</div>
        ) : calls.length === 0 ? (
          <Card className="bg-gradient-card border-border p-8 text-center">
            <p className="text-muted-foreground">No call history yet</p>
            <p className="text-sm text-muted-foreground mt-2">Your recent calls will appear here</p>
          </Card>
        ) : (
          calls.map((call) => (
            <Card key={call.id} className="bg-gradient-card border-border p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-full ${call.scam_detected ? "bg-destructive/20" : "bg-alert-safe/20"}`}>
                    <Phone className={`h-5 w-5 ${call.scam_detected ? "text-destructive" : "text-alert-safe"}`} />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-foreground">{call.phone_number}</h3>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span>{formatDistanceToNow(new Date(call.started_at), { addSuffix: true })}</span>
                      <span>•</span>
                      <span>{formatDuration(call.duration)}</span>
                    </div>
                  </div>
                </div>
                {getRiskBadge(call.scam_risk_level)}
              </div>

              {call.scam_reasons && call.scam_reasons.length > 0 && (
                <div className="mt-3 pt-3 border-t border-border">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="h-4 w-4 text-alert-warning mt-0.5" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-foreground mb-1">Detected Threats:</p>
                      <div className="flex flex-wrap gap-2">
                        {call.scam_reasons.map((reason, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {reason}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </Card>
          ))
        )}
      </div>
    </div>
  );
};

export default CallHistory;
