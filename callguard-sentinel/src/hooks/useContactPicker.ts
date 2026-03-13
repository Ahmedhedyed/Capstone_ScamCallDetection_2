import { useCallback, useState } from "react";

export interface PickedContact {
  name?: string;
  tel?: string;
  email?: string;
}

export function useContactPicker() {
  const [supported, setSupported] = useState<boolean>("contacts" in navigator && "ContactsManager" in window);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pick = useCallback(async (): Promise<PickedContact[] | null> => {
    setError(null);
    if (!("contacts" in navigator) || !("select" in (navigator as any).contacts)) {
      setSupported(false);
      return null;
    }
    try {
      setLoading(true);
      const contacts = await (navigator as any).contacts.select(["name", "tel", "email"], { multiple: true });
      return (contacts || []).map((c: any) => ({
        name: Array.isArray(c.name) ? c.name[0] : c.name,
        tel: Array.isArray(c.tel) ? c.tel[0] : c.tel,
        email: Array.isArray(c.email) ? c.email[0] : c.email,
      }));
    } catch (e: any) {
      setError(e?.message || "Failed to pick contacts");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { supported, loading, error, pick };
}


