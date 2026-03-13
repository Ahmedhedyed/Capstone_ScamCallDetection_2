import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Dialer from "./pages/Dialer";
import CallScreen from "./pages/CallScreen";
import Contacts from "./pages/Contacts";
import CallHistory from "./pages/CallHistory";
import Auth from "./pages/Auth";
import NotFound from "./pages/NotFound";
import { CallRecorder } from "./components/CallRecorder";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Auth />} />
          <Route path="/auth" element={<Auth />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/dialer" element={<Dialer />} />
          <Route path="/call" element={<CallScreen />} />
          <Route path="/contacts" element={<Contacts />} />
          <Route path="/history" element={<CallHistory />} />
          <Route path="/recorder" element={<CallRecorder />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;

