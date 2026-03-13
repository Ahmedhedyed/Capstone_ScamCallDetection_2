
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Shield, Phone, AlertTriangle, TrendingUp, Users, Clock } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  Legend,
} from "recharts";

const Dashboard = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState({
    total_calls: 0,
    scam_calls_blocked: 0,
    warning_calls: 0,
    safe_calls: 0,
  });
  const [loading, setLoading] = useState(true);
  const [recentCalls, setRecentCalls] = useState<any[]>([]);
  const [riskBreakdown, setRiskBreakdown] = useState<{ name: string; value: number }[]>([]);
  const [topRiskyContacts, setTopRiskyContacts] = useState<any[]>([]);

  useEffect(() => {
    loadStats();
    loadCharts();
    loadTopContacts();
  }, []);

  const loadStats = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        navigate("/auth");
        return;
      }

      const { data, error } = await supabase
        .from("scam_statistics")
        .select("*")
        .eq("user_id", user.id)
        .maybeSingle();

      if (error) {
        console.error("Error loading stats:", error);
      } else if (data) {
        setStats(data);
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadCharts = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      // Recent calls (last 10)
      const { data: calls } = await supabase
        .from("calls")
        .select("id, started_at, duration, scam_detected, scam_risk_level")
        .eq("user_id", user.id)
        .order("started_at", { ascending: false })
        .limit(10);

      setRecentCalls((calls || []).reverse().map((c) => ({
        time: new Date(c.started_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        duration: c.duration || 0,
        scam: c.scam_detected ? 1 : 0,
      })));

      // Risk breakdown pie
      const { data: breakdown } = await supabase
        .from("calls")
        .select("scam_risk_level, count:scam_risk_level")
        .eq("user_id", user.id);

      const counts: Record<string, number> = { safe: 0, warning: 0, critical: 0 };
      (breakdown || []).forEach((row: any) => {
        const level = row.scam_risk_level || "safe";
        counts[level] = (counts[level] || 0) + 1;
      });
      setRiskBreakdown([
        { name: "Safe", value: counts.safe || 0 },
        { name: "Warning", value: counts.warning || 0 },
        { name: "Critical", value: counts.critical || 0 },
      ]);
    } catch (e) {
      // ignore chart errors in UI
    }
  };

  const loadTopContacts = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const { data } = await supabase
        .from("contacts")
        .select("id, name, phone_number, scam_risk_level, scam_score")
        .eq("user_id", user.id)
        .order("scam_score", { ascending: false })
        .limit(5);

      setTopRiskyContacts(data || []);
    } catch (e) {
      // ignore
    }
  };

  const statCards = [
    {
      title: "Total Calls",
      value: stats.total_calls,
      icon: Phone,
      color: "text-foreground",
    },
    {
      title: "Scams Blocked",
      value: stats.scam_calls_blocked,
      icon: Shield,
      color: "text-alert-critical",
    },
    {
      title: "Warnings",
      value: stats.warning_calls,
      icon: AlertTriangle,
      color: "text-alert-warning",
    },
    {
      title: "Safe Calls",
      value: stats.safe_calls,
      icon: TrendingUp,
      color: "text-alert-safe",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-dark p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">CallGuard-Sentinel</h1>
            <p className="text-muted-foreground">Real-time call protection</p>
          </div>
          <Button onClick={() => navigate("/dialer")} className="bg-primary hover:bg-primary/90">
            <Phone className="mr-2 h-4 w-4" />
            Make Call
          </Button>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {statCards.map((stat) => (
            <Card key={stat.title} className="bg-gradient-card border-border p-6 shadow-elevated">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">{stat.title}</p>
                  <p className="text-3xl font-bold text-foreground mt-2">{stat.value}</p>
                </div>
                <stat.icon className={`h-10 w-10 ${stat.color}`} />
              </div>
            </Card>
          ))}
        </div>

        {/* Analytics */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card className="bg-gradient-card border-border p-4 lg:col-span-2">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold text-foreground">Recent Call Durations</h3>
              </div>
              <span className="text-xs text-muted-foreground">Last 10 calls</span>
            </div>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={recentCalls} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorDur" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6}/>
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                  <XAxis dataKey="time" stroke="#aaa"/>
                  <YAxis stroke="#aaa"/>
                  <Tooltip contentStyle={{ background: "#111", border: "1px solid #333" }} />
                  <Area type="monotone" dataKey="duration" stroke="#ef4444" fillOpacity={1} fill="url(#colorDur)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card className="bg-gradient-card border-border p-4">
            <div className="flex items-center gap-2 mb-3">
              <Shield className="h-5 w-5 text-alert-critical" />
              <h3 className="text-lg font-semibold text-foreground">Risk Breakdown</h3>
            </div>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={riskBreakdown} dataKey="value" nameKey="name" outerRadius={80} label>
                    {riskBreakdown.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={
                        entry.name === "Critical" ? "#ef4444" : entry.name === "Warning" ? "#f59e0b" : "#22c55e"
                      } />
                    ))}
                  </Pie>
                  <Legend />
                  <Tooltip contentStyle={{ background: "#111", border: "1px solid #333" }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        {/* Top Risky Contacts and Quick Links */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card className="bg-gradient-card border-border p-4 lg:col-span-2">
            <div className="flex items-center gap-2 mb-3">
              <Users className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold text-foreground">Top Risky Contacts</h3>
            </div>
            <div className="space-y-2">
              {topRiskyContacts.length === 0 ? (
                <p className="text-sm text-muted-foreground">No contacts yet</p>
              ) : (
                topRiskyContacts.map((c) => (
                  <div key={c.id} className="flex items-center justify-between p-3 rounded-lg hover:bg-secondary/50">
                    <div>
                      <p className="text-foreground font-medium">{c.name}</p>
                      <p className="text-muted-foreground text-sm">{c.phone_number}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      {c.scam_risk_level === "high" ? (
                        <Badge variant="destructive">High</Badge>
                      ) : c.scam_risk_level === "medium" ? (
                        <Badge className="bg-alert-warning text-white">Medium</Badge>
                      ) : (
                        <Badge className="bg-alert-safe text-white">Safe</Badge>
                      )}
                      <Button size="sm" onClick={() => navigate(`/call?number=${c.phone_number}`)} className="bg-primary hover:bg-primary/90">
                        <Phone className="h-4 w-4 mr-1" /> Call
                      </Button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>

          {/* Quick Actions */}
          <div className="space-y-4">
            <Card className="bg-gradient-card border-border p-4 cursor-pointer hover:shadow-glow-red transition-shadow" onClick={() => navigate("/contacts")}>
              <h3 className="text-lg font-semibold text-foreground mb-1">Contacts</h3>
              <p className="text-muted-foreground text-sm">View and manage your contacts</p>
            </Card>
            <Card className="bg-gradient-card border-border p-4 cursor-pointer hover:shadow-glow-red transition-shadow" onClick={() => navigate("/history")}>
              <h3 className="text-lg font-semibold text-foreground mb-1">Call History</h3>
              <p className="text-muted-foreground text-sm">Review past calls and detections</p>
            </Card>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Dashboard;


// import { useEffect, useState } from "react";
// import { useNavigate } from "react-router-dom";
// import { supabase } from "@/integrations/supabase/client";
// import { Shield, Phone, AlertTriangle, TrendingUp, Users, Clock } from "lucide-react";
// import { Card } from "@/components/ui/card";
// import { Button } from "@/components/ui/button";
// import { Badge } from "@/components/ui/badge";
// import {
//   LineChart,
//   Line,
//   XAxis,
//   YAxis,
//   Tooltip,
//   ResponsiveContainer,
//   CartesianGrid,
//   PieChart,
//   Pie,
//   Cell,
//   AreaChart,
//   Area,
//   Legend,
// } from "recharts";

// const Dashboard = () => {
//   const navigate = useNavigate();
//   const [stats, setStats] = useState({
//     total_calls: 0,
//     scam_calls_blocked: 0,
//     warning_calls: 0,
//     safe_calls: 0,
//   });
//   const [loading, setLoading] = useState(true);
//   const [recentCalls, setRecentCalls] = useState<any[]>([]);
//   const [riskBreakdown, setRiskBreakdown] = useState<{ name: string; value: number }[]>([]);
//   const [topRiskyContacts, setTopRiskyContacts] = useState<any[]>([]);

//   useEffect(() => {
//     loadStats();
//     loadCharts();
//     loadTopContacts();
//   }, []);

//   const loadStats = async () => {
//     try {
//       const { data: { user } } = await supabase.auth.getUser();
//       if (!user) {
//         navigate("/auth");
//         return;
//       }

//       const { data, error } = await supabase
//         .from("scam_statistics")
//         .select("*")
//         .eq("user_id", user.id)
//         .single();

//       if (error && error.code !== "PGRST116") {
//         console.error("Error loading stats:", error);
//       } else if (data) {
//         setStats(data);
//       }
//     } catch (error) {
//       console.error("Error:", error);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const loadCharts = async () => {
//     try {
//       const { data: { user } } = await supabase.auth.getUser();
//       if (!user) return;

//       // Recent calls (last 10)
//       const { data: calls } = await supabase
//         .from("calls")
//         .select("id, started_at, duration, scam_detected, scam_risk_level")
//         .eq("user_id", user.id)
//         .order("started_at", { ascending: false })
//         .limit(10);

//       setRecentCalls((calls || []).reverse().map((c) => ({
//         time: new Date(c.started_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
//         duration: c.duration || 0,
//         scam: c.scam_detected ? 1 : 0,
//       })));

//       // Risk breakdown pie
//       const { data: breakdown } = await supabase
//         .from("calls")
//         .select("scam_risk_level, count:scam_risk_level")
//         .eq("user_id", user.id);

//       const counts: Record<string, number> = { safe: 0, warning: 0, critical: 0 };
//       (breakdown || []).forEach((row: any) => {
//         const level = row.scam_risk_level || "safe";
//         counts[level] = (counts[level] || 0) + 1;
//       });
//       setRiskBreakdown([
//         { name: "Safe", value: counts.safe || 0 },
//         { name: "Warning", value: counts.warning || 0 },
//         { name: "Critical", value: counts.critical || 0 },
//       ]);
//     } catch (e) {
//       // ignore chart errors in UI
//     }
//   };

//   const loadTopContacts = async () => {
//     try {
//       const { data: { user } } = await supabase.auth.getUser();
//       if (!user) return;

//       const { data } = await supabase
//         .from("contacts")
//         .select("id, name, phone_number, scam_risk_level, scam_score")
//         .eq("user_id", user.id)
//         .order("scam_score", { ascending: false })
//         .limit(5);

//       setTopRiskyContacts(data || []);
//     } catch (e) {
//       // ignore
//     }
//   };

//   const statCards = [
//     {
//       title: "Total Calls",
//       value: stats.total_calls,
//       icon: Phone,
//       color: "text-foreground",
//     },
//     {
//       title: "Scams Blocked",
//       value: stats.scam_calls_blocked,
//       icon: Shield,
//       color: "text-alert-critical",
//     },
//     {
//       title: "Warnings",
//       value: stats.warning_calls,
//       icon: AlertTriangle,
//       color: "text-alert-warning",
//     },
//     {
//       title: "Safe Calls",
//       value: stats.safe_calls,
//       icon: TrendingUp,
//       color: "text-alert-safe",
//     },
//   ];

//   return (
//     <div className="min-h-screen bg-gradient-dark p-6">
//       <div className="max-w-6xl mx-auto space-y-6">
//         {/* Header */}
//         <div className="flex items-center justify-between">
//           <div>
//             <h1 className="text-3xl font-bold text-foreground">CallGuard-Sentinel</h1>
//             <p className="text-muted-foreground">Real-time call protection</p>
//           </div>
//           <Button onClick={() => navigate("/dialer")} className="bg-primary hover:bg-primary/90">
//             <Phone className="mr-2 h-4 w-4" />
//             Make Call
//           </Button>
//         </div>

//         {/* Stats Grid */}
//         <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
//           {statCards.map((stat) => (
//             <Card key={stat.title} className="bg-gradient-card border-border p-6 shadow-elevated">
//               <div className="flex items-center justify-between">
//                 <div>
//                   <p className="text-sm text-muted-foreground">{stat.title}</p>
//                   <p className="text-3xl font-bold text-foreground mt-2">{stat.value}</p>
//                 </div>
//                 <stat.icon className={`h-10 w-10 ${stat.color}`} />
//               </div>
//             </Card>
//           ))}
//         </div>

//         {/* Analytics */}
//         <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
//           <Card className="bg-gradient-card border-border p-4 lg:col-span-2">
//             <div className="flex items-center justify-between mb-3">
//               <div className="flex items-center gap-2">
//                 <TrendingUp className="h-5 w-5 text-primary" />
//                 <h3 className="text-lg font-semibold text-foreground">Recent Call Durations</h3>
//               </div>
//               <span className="text-xs text-muted-foreground">Last 10 calls</span>
//             </div>
//             <div className="h-56">
//               <ResponsiveContainer width="100%" height="100%">
//                 <AreaChart data={recentCalls} margin={{ left: 8, right: 8, top: 10, bottom: 0 }}>
//                   <defs>
//                     <linearGradient id="colorDur" x1="0" y1="0" x2="0" y2="1">
//                       <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6}/>
//                       <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
//                     </linearGradient>
//                   </defs>
//                   <CartesianGrid strokeDasharray="3 3" stroke="#222" />
//                   <XAxis dataKey="time" stroke="#aaa"/>
//                   <YAxis stroke="#aaa"/>
//                   <Tooltip contentStyle={{ background: "#111", border: "1px solid #333" }} />
//                   <Area type="monotone" dataKey="duration" stroke="#ef4444" fillOpacity={1} fill="url(#colorDur)" />
//                 </AreaChart>
//               </ResponsiveContainer>
//             </div>
//           </Card>

//           <Card className="bg-gradient-card border-border p-4">
//             <div className="flex items-center gap-2 mb-3">
//               <Shield className="h-5 w-5 text-alert-critical" />
//               <h3 className="text-lg font-semibold text-foreground">Risk Breakdown</h3>
//             </div>
//             <div className="h-56">
//               <ResponsiveContainer width="100%" height="100%">
//                 <PieChart>
//                   <Pie data={riskBreakdown} dataKey="value" nameKey="name" outerRadius={80} label>
//                     {riskBreakdown.map((entry, index) => (
//                       <Cell key={`cell-${index}`} fill={
//                         entry.name === "Critical" ? "#ef4444" : entry.name === "Warning" ? "#f59e0b" : "#22c55e"
//                       } />
//                     ))}
//                   </Pie>
//                   <Legend />
//                   <Tooltip contentStyle={{ background: "#111", border: "1px solid #333" }} />
//                 </PieChart>
//               </ResponsiveContainer>
//             </div>
//           </Card>
//         </div>

//         {/* Top Risky Contacts and Quick Links */}
//         <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
//           <Card className="bg-gradient-card border-border p-4 lg:col-span-2">
//             <div className="flex items-center gap-2 mb-3">
//               <Users className="h-5 w-5 text-primary" />
//               <h3 className="text-lg font-semibold text-foreground">Top Risky Contacts</h3>
//             </div>
//             <div className="space-y-2">
//               {topRiskyContacts.length === 0 ? (
//                 <p className="text-sm text-muted-foreground">No contacts yet</p>
//               ) : (
//                 topRiskyContacts.map((c) => (
//                   <div key={c.id} className="flex items-center justify-between p-3 rounded-lg hover:bg-secondary/50">
//                     <div>
//                       <p className="text-foreground font-medium">{c.name}</p>
//                       <p className="text-muted-foreground text-sm">{c.phone_number}</p>
//                     </div>
//                     <div className="flex items-center gap-2">
//                       {c.scam_risk_level === "high" ? (
//                         <Badge variant="destructive">High</Badge>
//                       ) : c.scam_risk_level === "medium" ? (
//                         <Badge className="bg-alert-warning text-white">Medium</Badge>
//                       ) : (
//                         <Badge className="bg-alert-safe text-white">Safe</Badge>
//                       )}
//                       <Button size="sm" onClick={() => navigate(`/call?number=${c.phone_number}`)} className="bg-primary hover:bg-primary/90">
//                         <Phone className="h-4 w-4 mr-1" /> Call
//                       </Button>
//                     </div>
//                   </div>
//                 ))
//               )}
//             </div>
//           </Card>

//           {/* Quick Actions */}
//           <div className="space-y-4">
//             <Card className="bg-gradient-card border-border p-4 cursor-pointer hover:shadow-glow-red transition-shadow" onClick={() => navigate("/contacts")}>
//               <h3 className="text-lg font-semibold text-foreground mb-1">Contacts</h3>
//               <p className="text-muted-foreground text-sm">View and manage your contacts</p>
//             </Card>
//             <Card className="bg-gradient-card border-border p-4 cursor-pointer hover:shadow-glow-red transition-shadow" onClick={() => navigate("/history")}>
//               <h3 className="text-lg font-semibold text-foreground mb-1">Call History</h3>
//               <p className="text-muted-foreground text-sm">Review past calls and detections</p>
//             </Card>
//           </div>
//         </div>

//         {/* Quick Actions */}
//         <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//           <Card className="bg-gradient-card border-border p-6 cursor-pointer hover:shadow-glow-red transition-shadow" onClick={() => navigate("/contacts")}>
//             <h3 className="text-xl font-semibold text-foreground mb-2">Contacts</h3>
//             <p className="text-muted-foreground">View and manage your contacts with scam risk indicators</p>
//           </Card>
//           <Card className="bg-gradient-card border-border p-6 cursor-pointer hover:shadow-glow-red transition-shadow" onClick={() => navigate("/history")}>
//             <h3 className="text-xl font-semibold text-foreground mb-2">Call History</h3>
//             <p className="text-muted-foreground">Review past calls and scam detections</p>
//           </Card>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Dashboard;
