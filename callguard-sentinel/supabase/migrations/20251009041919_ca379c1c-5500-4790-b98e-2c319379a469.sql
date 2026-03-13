-- Create contacts table
CREATE TABLE public.contacts (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  name TEXT NOT NULL,
  phone_number TEXT NOT NULL,
  email TEXT,
  scam_risk_level TEXT DEFAULT 'unknown',
  scam_score DECIMAL(3, 2) DEFAULT 0.00,
  last_called_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create calls table
CREATE TABLE public.calls (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  contact_id UUID REFERENCES public.contacts(id) ON DELETE SET NULL,
  phone_number TEXT NOT NULL,
  duration INTEGER DEFAULT 0,
  scam_detected BOOLEAN DEFAULT false,
  scam_risk_level TEXT DEFAULT 'safe',
  scam_score DECIMAL(3, 2) DEFAULT 0.00,
  scam_reasons TEXT[],
  transcription TEXT,
  audio_features JSONB,
  linguistic_features JSONB,
  conversational_features JSONB,
  call_status TEXT DEFAULT 'completed',
  started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  ended_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create scam_reports table
CREATE TABLE public.scam_reports (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  call_id UUID REFERENCES public.calls(id) ON DELETE CASCADE NOT NULL,
  user_id UUID NOT NULL,
  phone_number TEXT NOT NULL,
  scam_type TEXT,
  confidence_score DECIMAL(3, 2) NOT NULL,
  features_detected TEXT[],
  transcription_snippet TEXT,
  reported_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create scam_statistics table
CREATE TABLE public.scam_statistics (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL,
  total_calls INTEGER DEFAULT 0,
  scam_calls_blocked INTEGER DEFAULT 0,
  warning_calls INTEGER DEFAULT 0,
  safe_calls INTEGER DEFAULT 0,
  last_scam_detected_at TIMESTAMP WITH TIME ZONE,
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.contacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scam_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scam_statistics ENABLE ROW LEVEL SECURITY;

-- Contacts policies
CREATE POLICY "Users can view their own contacts"
ON public.contacts FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own contacts"
ON public.contacts FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own contacts"
ON public.contacts FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own contacts"
ON public.contacts FOR DELETE
USING (auth.uid() = user_id);

-- Calls policies
CREATE POLICY "Users can view their own calls"
ON public.calls FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own calls"
ON public.calls FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own calls"
ON public.calls FOR UPDATE
USING (auth.uid() = user_id);

-- Scam reports policies
CREATE POLICY "Users can view their own scam reports"
ON public.scam_reports FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own scam reports"
ON public.scam_reports FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- Statistics policies
CREATE POLICY "Users can view their own statistics"
ON public.scam_statistics FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own statistics"
ON public.scam_statistics FOR UPDATE
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own statistics"
ON public.scam_statistics FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SET search_path = public;

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_contacts_updated_at
BEFORE UPDATE ON public.contacts
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_scam_statistics_updated_at
BEFORE UPDATE ON public.scam_statistics
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();