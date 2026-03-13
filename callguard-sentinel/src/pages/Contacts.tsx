import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { ArrowLeft, Phone, Shield, AlertTriangle, CheckCircle, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useContactPicker } from "@/hooks/useContactPicker";
import { toast } from "sonner";

interface Contact {
  id: string;
  name: string;
  phone_number: string;
  scam_risk_level: string;
  scam_score: number;
}

const Contacts = () => {
  const navigate = useNavigate();
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [loading, setLoading] = useState(true);
  const { supported, pick, loading: picking } = useContactPicker();

  useEffect(() => {
    loadContacts();
  }, []);

  const loadContacts = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        navigate("/auth");
        return;
      }

      const { data, error } = await supabase
        .from("contacts")
        .select("*")
        .eq("user_id", user.id)
        .order("name");

      if (error) throw error;
      setContacts(data || []);
    } catch (error) {
      console.error("Error loading contacts:", error);
    } finally {
      setLoading(false);
    }
  };

  const importFromDevice = async () => {
    const picked = await pick();
    if (!picked) {
      toast("Contact Picker not available. Use CSV import instead.");
      return;
    }
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;
      const rows = picked
        .filter(p => p.tel)
        .map(p => ({
          user_id: user.id,
          name: p.name || "Unknown",
          phone_number: p.tel!,
          scam_risk_level: "safe",
          scam_score: 0,
        }));
      if (rows.length === 0) return;
      await supabase.from("contacts").upsert(rows, { onConflict: "user_id,phone_number" });
      toast.success(`Imported ${rows.length} contacts`);
      await loadContacts();
    } catch (e) {
      toast.error("Failed to import contacts");
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level) {
      case "high":
        return <Shield className="h-5 w-5 text-alert-critical" />;
      case "medium":
        return <AlertTriangle className="h-5 w-5 text-alert-warning" />;
      default:
        return <CheckCircle className="h-5 w-5 text-alert-safe" />;
    }
  };

  const getRiskBadge = (level: string) => {
    switch (level) {
      case "high":
        return <Badge variant="destructive">High Risk</Badge>;
      case "medium":
        return <Badge className="bg-alert-warning text-white">Medium Risk</Badge>;
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
        <h1 className="text-xl font-semibold text-foreground ml-4">Contacts</h1>
      </div>

      <div className="p-6 space-y-4">
        <div className="flex justify-end">
          <Button onClick={importFromDevice} disabled={picking} variant="outline" className="border-border">
            <Upload className="h-4 w-4 mr-2" /> {supported ? "Import from device" : "Import (unsupported)"}
          </Button>
        </div>
        {loading ? (
          <div className="text-center text-muted-foreground">Loading contacts...</div>
        ) : contacts.length === 0 ? (
          <Card className="bg-gradient-card border-border p-8 text-center">
            <p className="text-muted-foreground">No contacts yet</p>
            <p className="text-sm text-muted-foreground mt-2">Add contacts to see scam risk indicators</p>
          </Card>
        ) : (
          contacts.map((contact) => (
            <Card key={contact.id} className="bg-gradient-card border-border p-4 hover:shadow-glow-red transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  {getRiskIcon(contact.scam_risk_level)}
                  <div>
                    <h3 className="text-lg font-semibold text-foreground">{contact.name}</h3>
                    <p className="text-sm text-muted-foreground">{contact.phone_number}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {getRiskBadge(contact.scam_risk_level)}
                  <Button onClick={() => navigate(`/call?number=${contact.phone_number}`)} className="bg-primary hover:bg-primary/90">
                    <Phone className="h-4 w-4 mr-2" /> Call
                  </Button>
                </div>
              </div>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};

export default Contacts;
