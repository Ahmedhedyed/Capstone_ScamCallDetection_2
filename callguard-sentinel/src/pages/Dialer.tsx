import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Phone, Delete, ArrowLeft, Search, User, AlertTriangle, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useContacts } from "@/hooks/useContacts";
import { useScamAlerts } from "@/hooks/useScamAlerts";
import { ScamAlertContainer } from "@/components/ScamAlert";
import { motion, AnimatePresence } from "framer-motion";
import { supabase } from "@/integrations/supabase/client";
import { checkBackendHealth } from "@/lib/api";
import { toast } from "sonner";

const Dialer = () => {
  const navigate = useNavigate();
  const [phoneNumber, setPhoneNumber] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [showSearch, setShowSearch] = useState(false);
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);
  
  const { 
    contacts, 
    searchContacts, 
    getContactByNumber, 
    syncContacts, 
    isSyncing 
  } = useContacts();
  
  const {
    alerts,
    clearAlert,
  } = useScamAlerts();

  const [filteredContacts, setFilteredContacts] = useState(contacts);

  useEffect(() => {
    if (searchQuery.trim()) {
      setFilteredContacts(searchContacts(searchQuery));
    } else {
      setFilteredContacts(contacts);
    }
  }, [searchQuery, contacts, searchContacts]);

  // Poll backend health every 5 s until connected, then every 30 s to detect drops
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      const ok = await checkBackendHealth();
      if (!cancelled) setBackendConnected(ok);
      if (!cancelled) {
        setTimeout(poll, ok ? 30_000 : 5_000);
      }
    };
    poll();
    return () => { cancelled = true; };
  }, []);

  const handleDigitClick = (digit: string) => {
    setPhoneNumber(prev => prev + digit);
    setSearchQuery(prev => prev + digit);
  };

  const handleDelete = () => {
    setPhoneNumber(prev => prev.slice(0, -1));
    setSearchQuery(prev => prev.slice(0, -1));
  };

  const handleCall = async () => {
    if (phoneNumber.length === 0) return;
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      navigate("/auth");
      return;
    }
    // Always do a fresh check right before the call
    const isOnline = await checkBackendHealth();
    setBackendConnected(isOnline);
    if (isOnline) {
      try {
        await initiateCall({ userId: user.id, phoneNumber });
      } catch {
        toast("Could not reach backend – starting demo call");
      }
    } else {
      toast("Backend offline – starting demo call");
    }
    navigate(`/call?number=${phoneNumber}`);
  };

  const handleContactSelect = (contactNumber: string) => {
    setPhoneNumber(contactNumber);
    setSearchQuery(contactNumber);
    setShowSearch(false);
  };

  const currentContact = getContactByNumber(phoneNumber);
  const isScamNumber = currentContact?.isScam || false;
  const scamScore = currentContact?.scamScore || 0;

  const digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"];

  return (
    <div className="min-h-screen bg-gradient-dark flex flex-col">
      {/* Scam Alerts */}
      <ScamAlertContainer alerts={alerts} onDismiss={clearAlert} />
      
      {/* Header */}
      <div className="p-6 flex items-center justify-between">
        <div className="flex items-center">
          <Button variant="ghost" onClick={() => navigate("/dashboard")} className="text-foreground">
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <h1 className="text-xl font-semibold text-foreground ml-4">Dialer</h1>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            onClick={() => setShowSearch(!showSearch)}
            className="text-foreground"
          >
            <Search className="h-5 w-5" />
          </Button>
          <Button
            variant="ghost"
            onClick={() => syncContacts()}
            disabled={isSyncing}
            className="text-foreground"
          >
            <User className="h-5 w-5" />
          </Button>
          <div className={`w-2 h-2 rounded-full ${backendConnected ? 'bg-alert-safe' : 'bg-alert-critical'}`} />
          {backendConnected === false && (
            <span className="text-xs text-muted-foreground ml-1">Backend offline</span>
          )}
        </div>
      </div>

      {/* Search Bar */}
      <AnimatePresence>
        {showSearch && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="px-6 mb-4"
          >
            <Input
              type="text"
              placeholder="Search contacts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-secondary border-border text-foreground"
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Contact Search Results */}
      <AnimatePresence>
        {showSearch && searchQuery && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="px-6 mb-4 max-h-60 overflow-y-auto"
          >
            <Card className="bg-gradient-card border-border">
              <div className="p-4 space-y-2">
                {filteredContacts.length > 0 ? (
                  filteredContacts.map((contact) => (
                    <div
                      key={contact.id}
                      onClick={() => handleContactSelect(contact.phoneNumber)}
                      className="flex items-center justify-between p-3 rounded-lg hover:bg-secondary/50 cursor-pointer transition-colors"
                    >
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                          <User className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <p className="font-medium text-foreground">{contact.name}</p>
                          <p className="text-sm text-muted-foreground">{contact.phoneNumber}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        {contact.isScam && (
                          <Badge variant="destructive" className="text-xs">
                            <AlertTriangle className="h-3 w-3 mr-1" />
                            Scam
                          </Badge>
                        )}
                        {!contact.isScam && (
                          <Badge variant="outline" className="text-xs text-alert-safe border-alert-safe">
                            <Shield className="h-3 w-3 mr-1" />
                            Safe
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-center text-muted-foreground py-4">No contacts found</p>
                )}
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex-1 flex flex-col items-center justify-center px-6 pb-12">
        {/* Phone Number Display */}
        <Card className={`bg-gradient-card border-border p-8 mb-8 w-full max-w-md ${
          isScamNumber ? 'border-alert-critical shadow-glow-red' : ''
        }`}>
          <div className="text-center">
            <input
              type="text"
              value={phoneNumber}
              readOnly
              placeholder="Enter number"
              className="bg-transparent text-3xl font-semibold text-foreground text-center w-full outline-none"
            />
            
            {/* Contact Info */}
            {currentContact && (
              <div className="mt-4 space-y-2">
                <p className="text-lg font-medium text-foreground">{currentContact.name}</p>
                <div className="flex items-center justify-center space-x-2">
                  {isScamNumber ? (
                    <Badge variant="destructive" className="text-sm">
                      <AlertTriangle className="h-4 w-4 mr-1" />
                      SCAM DETECTED ({Math.round(scamScore * 100)}%)
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="text-sm text-alert-safe border-alert-safe">
                      <Shield className="h-4 w-4 mr-1" />
                      VERIFIED SAFE
                    </Badge>
                  )}
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Dial Pad */}
        <div className="grid grid-cols-3 gap-4 w-full max-w-sm mb-8">
          {digits.map((digit) => (
            <Button
              key={digit}
              onClick={() => handleDigitClick(digit)}
              className="h-16 text-2xl font-semibold bg-secondary hover:bg-secondary/80 text-foreground"
            >
              {digit}
            </Button>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 w-full max-w-sm">
          <Button
            onClick={handleDelete}
            variant="outline"
            className="flex-1 h-14 border-border text-foreground"
          >
            <Delete className="h-5 w-5" />
          </Button>
          <Button
            onClick={handleCall}
            disabled={phoneNumber.length === 0}
            className={`flex-1 h-14 ${
              isScamNumber 
                ? 'bg-alert-critical hover:bg-alert-critical/90 text-white shadow-glow-red' 
                : 'bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow-red'
            }`}
          >
            <Phone className="h-5 w-5 mr-2" />
            {isScamNumber ? 'Call (RISKY)' : 'Call'}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Dialer;
