import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { open } from '@tauri-apps/api/dialog';
import { invoke } from '@tauri-apps/api/tauri';

interface ShortcutHandler {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  handler: () => void;
  description: string;
}

const useKeyboardShortcuts = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const shortcuts: ShortcutHandler[] = [
      {
        key: 'o',
        ctrl: true,
        handler: async () => {
          const filePath = await open({
            multiple: false,
            filters: [{
              name: 'Data Files',
              extensions: ['csv', 'xlsx', 'xls', 'json', 'parquet', 'hdf5', 'nc']
            }]
          });
          if (filePath) {
            navigate('/data-import');
          }
        },
        description: 'Open file'
      },
      {
        key: 's',
        ctrl: true,
        handler: async () => {
          await invoke('save_project');
        },
        description: 'Save project'
      },
      {
        key: 'i',
        ctrl: true,
        shift: true,
        handler: () => navigate('/data-import'),
        description: 'Quick import'
      },
      {
        key: 'r',
        ctrl: true,
        shift: true,
        handler: () => navigate('/imputation'),
        description: 'Run imputation'
      },
      {
        key: 'e',
        ctrl: true,
        handler: () => navigate('/export'),
        description: 'Export data'
      },
      {
        key: 'F1',
        handler: () => navigate('/help'),
        description: 'Open help'
      },
      {
        key: ',',
        ctrl: true,
        handler: () => navigate('/settings'),
        description: 'Open settings'
      }
    ];

    const handleKeyDown = (event: KeyboardEvent) => {
      for (const shortcut of shortcuts) {
        const ctrlMatch = shortcut.ctrl ? event.ctrlKey || event.metaKey : !event.ctrlKey && !event.metaKey;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;
        const altMatch = shortcut.alt ? event.altKey : !event.altKey;
        
        if (
          event.key === shortcut.key &&
          ctrlMatch &&
          shiftMatch &&
          altMatch
        ) {
          event.preventDefault();
          shortcut.handler();
          break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [navigate]);
};

export default useKeyboardShortcuts;