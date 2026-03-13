import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export interface Contact {
  id: string;
  name: string;
  phoneNumber: string;
  email?: string;
  isScam: boolean;
  scamScore?: number;
  lastCallDate?: Date;
  callCount: number;
  avatar?: string;
  notes?: string;
}

export interface CallRecord {
  id: string;
  contactId?: string;
  phoneNumber: string;
  contactName?: string;
  timestamp: Date;
  duration: number;
  type: 'incoming' | 'outgoing' | 'missed';
  isScam: boolean;
  scamScore?: number;
  scamReason?: string;
  recordingUrl?: string;
}

// Mock API functions - replace with actual API calls
const fetchContacts = async (): Promise<Contact[]> => {
  // Simulate API call
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // Mock data
  return [
    {
      id: '1',
      name: 'John Doe',
      phoneNumber: '+1234567890',
      email: 'john@example.com',
      isScam: false,
      callCount: 5,
      lastCallDate: new Date('2024-01-15'),
    },
    {
      id: '2',
      name: 'Jane Smith',
      phoneNumber: '+1987654321',
      isScam: false,
      callCount: 2,
      lastCallDate: new Date('2024-01-14'),
    },
    {
      id: '3',
      name: 'Unknown Caller',
      phoneNumber: '+1555000123',
      isScam: true,
      scamScore: 0.95,
      callCount: 1,
      lastCallDate: new Date('2024-01-13'),
    },
  ];
};

const fetchCallHistory = async (): Promise<CallRecord[]> => {
  await new Promise(resolve => setTimeout(resolve, 500));
  
  return [
    {
      id: '1',
      contactId: '1',
      phoneNumber: '+1234567890',
      contactName: 'John Doe',
      timestamp: new Date('2024-01-15T10:30:00'),
      duration: 180,
      type: 'outgoing',
      isScam: false,
    },
    {
      id: '2',
      phoneNumber: '+1555000123',
      timestamp: new Date('2024-01-13T14:20:00'),
      duration: 0,
      type: 'missed',
      isScam: true,
      scamScore: 0.95,
      scamReason: 'Bank details requested',
    },
  ];
};

const syncContacts = async (): Promise<Contact[]> => {
  // Simulate contact synchronization
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // In a real app, this would sync with device contacts
  return fetchContacts();
};

export const useContacts = () => {
  const queryClient = useQueryClient();
  
  const {
    data: contacts = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ['contacts'],
    queryFn: fetchContacts,
  });

  const {
    data: callHistory = [],
    isLoading: isLoadingHistory,
  } = useQuery({
    queryKey: ['callHistory'],
    queryFn: fetchCallHistory,
  });

  const syncMutation = useMutation({
    mutationFn: syncContacts,
    onSuccess: (newContacts) => {
      queryClient.setQueryData(['contacts'], newContacts);
    },
  });

  const searchContacts = (query: string) => {
    if (!query.trim()) return contacts;
    
    return contacts.filter(contact =>
      contact.name.toLowerCase().includes(query.toLowerCase()) ||
      contact.phoneNumber.includes(query)
    );
  };

  const getContactByNumber = (phoneNumber: string) => {
    return contacts.find(contact => contact.phoneNumber === phoneNumber);
  };

  const addContact = (contact: Omit<Contact, 'id'>) => {
    const newContact: Contact = {
      ...contact,
      id: Date.now().toString(),
    };
    
    queryClient.setQueryData(['contacts'], (old: Contact[] = []) => [...old, newContact]);
  };

  const updateContact = (id: string, updates: Partial<Contact>) => {
    queryClient.setQueryData(['contacts'], (old: Contact[] = []) =>
      old.map(contact => contact.id === id ? { ...contact, ...updates } : contact)
    );
  };

  const deleteContact = (id: string) => {
    queryClient.setQueryData(['contacts'], (old: Contact[] = []) =>
      old.filter(contact => contact.id !== id)
    );
  };

  return {
    contacts,
    callHistory,
    isLoading,
    isLoadingHistory,
    error,
    syncContacts: syncMutation.mutate,
    isSyncing: syncMutation.isPending,
    searchContacts,
    getContactByNumber,
    addContact,
    updateContact,
    deleteContact,
  };
};
