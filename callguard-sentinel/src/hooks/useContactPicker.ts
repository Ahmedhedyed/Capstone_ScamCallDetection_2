import { useCallback, useState } from "react";

export interface PickedContact {
  name?: string;
  tel?: string;
  email?: string;
}

// Minimal typing for the experimental Contact Picker API (not yet in TS lib.dom.d.ts)
interface ContactProperty {
  name?: string[];
  tel?: string[];
  email?: string[];
}

interface ContactsManager {
  select(properties: string[], options?: { multiple?: boolean }): Promise<ContactProperty[]>;
}

interface NavigatorWithContacts extends Navigator {
  contacts: ContactsManager;
}

export function useContactPicker() {
  const [supported, setSupported] = useState<boolean>("contacts" in navigator && "ContactsManager" in window);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pick = useCallback(async (): Promise<PickedContact[] | null> => {
    setError(null);
    const nav = navigator as NavigatorWithContacts;
    if (!("contacts" in navigator) || !("select" in nav.contacts)) {
      setSupported(false);
      return null;
    }
    try {
      setLoading(true);
      const contacts = await nav.contacts.select(["name", "tel", "email"], { multiple: true });
      return (contacts || []).map((c) => ({
        name: Array.isArray(c.name) ? c.name[0] : c.name,
        tel: Array.isArray(c.tel) ? c.tel[0] : c.tel,
        email: Array.isArray(c.email) ? c.email[0] : c.email,
      }));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to pick contacts");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { supported, loading, error, pick };
}


