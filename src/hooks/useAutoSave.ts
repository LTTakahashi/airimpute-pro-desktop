import { useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { useStore } from '@/store';

interface AutoSaveOptions {
  interval?: number; // in seconds
  enabled?: boolean;
  onSave?: () => void;
  onError?: (error: Error) => void;
}

const useAutoSave = (options: AutoSaveOptions = {}) => {
  const {
    interval = 300, // 5 minutes by default
    enabled = true,
    onSave,
    onError
  } = options;

  const { currentDataset, imputationResults } = useStore();
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastSaveRef = useRef<Date | null>(null);

  useEffect(() => {
    if (!enabled || !currentDataset) {
      return () => {}; // Return empty cleanup function
    }

    const performAutoSave = async () => {
      try {
        await invoke('auto_save_project', {
          datasetId: currentDataset.id,
          imputationId: imputationResults?.id
        });
        
        lastSaveRef.current = new Date();
        onSave?.();
      } catch (error) {
        console.error('Auto-save failed:', error);
        onError?.(error as Error);
      }
    };

    // Set up the interval
    const startAutoSave = () => {
      timeoutRef.current = setInterval(performAutoSave, interval * 1000);
    };

    startAutoSave();

    // Also save when the window is about to close
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (currentDataset && lastSaveRef.current) {
        const timeSinceLastSave = Date.now() - lastSaveRef.current.getTime();
        // If it's been more than 30 seconds since last save, prompt the user
        if (timeSinceLastSave > 30000) {
          e.preventDefault();
          e.returnValue = '';
          performAutoSave();
        }
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      if (timeoutRef.current) {
        clearInterval(timeoutRef.current);
      }
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [enabled, interval, currentDataset, imputationResults, onSave, onError]);

  return {
    lastSave: lastSaveRef.current,
    manualSave: async () => {
      if (currentDataset) {
        try {
          await invoke('save_project', {
            datasetId: currentDataset.id,
            imputationId: imputationResults?.id
          });
          lastSaveRef.current = new Date();
          onSave?.();
        } catch (error) {
          onError?.(error as Error);
          throw error;
        }
      }
    }
  };
};

export default useAutoSave;