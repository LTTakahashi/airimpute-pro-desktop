import React, { useEffect, useState } from 'react';
import { Cpu, HardDrive, Activity, Wifi, WifiOff } from 'lucide-react';
import { invoke } from '@tauri-apps/api/tauri';
import { useStore } from '@/store';

interface SystemStatus {
  cpuUsage: number;
  memoryUsage: number;
  memoryTotal: number;
  pythonStatus: 'connected' | 'disconnected' | 'error';
}

const StatusBar: React.FC = () => {
  const { currentDataset, imputationResults } = useStore();
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    cpuUsage: 0,
    memoryUsage: 0,
    memoryTotal: 0,
    pythonStatus: 'disconnected'
  });
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    // Update system status periodically
    const updateSystemStatus = async () => {
      try {
        const status = await invoke<SystemStatus>('get_system_status');
        setSystemStatus(status);
      } catch (err) {
        console.error('Failed to get system status:', err);
      }
    };

    updateSystemStatus();
    const interval = setInterval(updateSystemStatus, 5000);

    // Online status listeners
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      clearInterval(interval);
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <footer className="bg-gray-100 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
      <div className="px-4 py-2 flex items-center justify-between text-xs text-gray-600 dark:text-gray-400">
        {/* Left side - Dataset info */}
        <div className="flex items-center space-x-4">
          {currentDataset ? (
            <>
              <span>
                Dataset: <strong>{currentDataset.info.name}</strong>
              </span>
              <span className="text-gray-400">|</span>
              <span>
                {currentDataset.info.rows.toLocaleString()} rows × {currentDataset.info.columns} cols
              </span>
              {imputationResults && (
                <>
                  <span className="text-gray-400">|</span>
                  <span>
                    Imputed: <strong>{imputationResults.method}</strong>
                  </span>
                </>
              )}
            </>
          ) : (
            <span>No dataset loaded</span>
          )}
        </div>

        {/* Right side - System status */}
        <div className="flex items-center space-x-4">
          {/* CPU Usage */}
          <div className="flex items-center space-x-1">
            <Cpu className="w-3 h-3" />
            <span>{systemStatus.cpuUsage.toFixed(0)}%</span>
          </div>

          {/* Memory Usage */}
          <div className="flex items-center space-x-1">
            <HardDrive className="w-3 h-3" />
            <span>
              {(systemStatus.memoryUsage / 1024).toFixed(1)}GB / 
              {(systemStatus.memoryTotal / 1024).toFixed(1)}GB
            </span>
          </div>

          {/* Python Status */}
          <div className="flex items-center space-x-1">
            <Activity 
              className={`w-3 h-3 ${
                systemStatus.pythonStatus === 'connected' 
                  ? 'text-green-500' 
                  : systemStatus.pythonStatus === 'error'
                  ? 'text-red-500'
                  : 'text-gray-400'
              }`}
            />
            <span>Python</span>
          </div>

          {/* Online Status */}
          <div className="flex items-center space-x-1">
            {isOnline ? (
              <>
                <Wifi className="w-3 h-3 text-green-500" />
                <span>Online</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3 h-3 text-red-500" />
                <span>Offline</span>
              </>
            )}
          </div>
        </div>
      </div>
    </footer>
  );
};

export default StatusBar;