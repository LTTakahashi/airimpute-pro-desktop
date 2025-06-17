import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Save, RotateCcw, Monitor, Cpu, Database } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { ScientificCard } from '@/components/layout/ScientificCard';
import { NumericInput } from '@/components/forms/NumericInput';
import { invoke } from '@tauri-apps/api/tauri';
import { useStore } from '@/store';

interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  autoSave: boolean;
  autoSaveInterval: number;
  notifications: boolean;
  language: string;
  expertMode: boolean;
}

interface ComputationSettings {
  maxThreads: number;
  maxMemoryMB: number;
  chunkSize: number;
  gpuAcceleration: boolean;
  cachingEnabled: boolean;
  compressionLevel: number;
}

interface SystemInfo {
  cpuCores: number;
  totalMemoryMB: number;
  availableMemoryMB: number;
  pythonVersion: string;
  rustVersion: string;
  gpuAvailable: boolean;
}

const Settings: React.FC = () => {
  const { theme, setTheme } = useStore();
  const [activeTab, setActiveTab] = useState<'general' | 'computation' | 'system'>('general');
  const [preferences, setPreferences] = useState<UserPreferences>({
    theme: theme as any,
    autoSave: true,
    autoSaveInterval: 300,
    notifications: true,
    language: 'en',
    expertMode: false
  });
  const [computationSettings, setComputationSettings] = useState<ComputationSettings>({
    maxThreads: 4,
    maxMemoryMB: 4096,
    chunkSize: 10000,
    gpuAcceleration: false,
    cachingEnabled: true,
    compressionLevel: 5
  });
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    loadSettings();
    loadSystemInfo();
  }, []);

  const loadSettings = async () => {
    try {
      const prefs = await invoke<UserPreferences>('get_user_preferences');
      setPreferences(prefs);
      
      const compSettings = await invoke<ComputationSettings>('get_computation_settings');
      setComputationSettings(compSettings);
    } catch (err) {
      console.error('Failed to load settings:', err);
    }
  };

  const loadSystemInfo = async () => {
    try {
      const info = await invoke<SystemInfo>('get_system_info');
      setSystemInfo(info);
    } catch (err) {
      console.error('Failed to load system info:', err);
    }
  };

  const handleSaveSettings = async () => {
    setSaving(true);
    setSaved(false);

    try {
      await invoke('update_user_preferences', { preferences });
      await invoke('update_computation_settings', { settings: computationSettings });
      
      // Apply theme change
      setTheme(preferences.theme);
      
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error('Failed to save settings:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleResetToDefaults = async () => {
    try {
      await invoke('reset_to_defaults');
      await loadSettings();
    } catch (err) {
      console.error('Failed to reset settings:', err);
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">Settings</h1>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'general', label: 'General', icon: SettingsIcon },
            { id: 'computation', label: 'Computation', icon: Cpu },
            { id: 'system', label: 'System Info', icon: Monitor }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`
                py-2 px-1 border-b-2 font-medium text-sm flex items-center
                ${activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}
              `}
            >
              <tab.icon className="w-4 h-4 mr-2" />
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* General Settings */}
      {activeTab === 'general' && (
        <div className="space-y-6">
          <ScientificCard
            title="Appearance"
            description="Customize the application appearance"
          >
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Theme</label>
                <select
                  value={preferences.theme}
                  onChange={(e) => setPreferences({ ...preferences, theme: e.target.value as any })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="system">System</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Language</label>
                <select
                  value={preferences.language}
                  onChange={(e) => setPreferences({ ...preferences, language: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="en">English</option>
                  <option value="es">Español</option>
                  <option value="pt">Português</option>
                  <option value="zh">中文</option>
                </select>
              </div>
            </div>
          </ScientificCard>

          <ScientificCard
            title="Behavior"
            description="Configure application behavior"
          >
            <div className="space-y-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={preferences.autoSave}
                  onChange={(e) => setPreferences({ ...preferences, autoSave: e.target.checked })}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <span className="ml-2">Enable auto-save</span>
              </label>

              {preferences.autoSave && (
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Auto-save interval (seconds)
                  </label>
                  <NumericInput
                    label="Auto-save interval (seconds)"
                    value={preferences.autoSaveInterval}
                    onChange={(value) => setPreferences({ ...preferences, autoSaveInterval: value })}
                    min={60}
                    max={3600}
                    step={60}
                  />
                </div>
              )}

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={preferences.notifications}
                  onChange={(e) => setPreferences({ ...preferences, notifications: e.target.checked })}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <span className="ml-2">Enable notifications</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={preferences.expertMode}
                  onChange={(e) => setPreferences({ ...preferences, expertMode: e.target.checked })}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <span className="ml-2">Expert mode (show advanced options)</span>
              </label>
            </div>
          </ScientificCard>
        </div>
      )}

      {/* Computation Settings */}
      {activeTab === 'computation' && (
        <div className="space-y-6">
          <ScientificCard
            title="Performance"
            description="Configure computational resources"
          >
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Maximum threads
                </label>
                <NumericInput
                  label="Maximum threads"
                  value={computationSettings.maxThreads}
                  onChange={(value) => setComputationSettings({ ...computationSettings, maxThreads: value })}
                  min={1}
                  max={systemInfo?.cpuCores || 16}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Available CPU cores: {systemInfo?.cpuCores || 'Unknown'}
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Maximum memory (MB)
                </label>
                <NumericInput
                  label="Maximum memory (MB)"
                  value={computationSettings.maxMemoryMB}
                  onChange={(value) => setComputationSettings({ ...computationSettings, maxMemoryMB: value })}
                  min={512}
                  max={systemInfo?.totalMemoryMB || 16384}
                  step={512}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Available memory: {systemInfo ? `${(systemInfo.availableMemoryMB / 1024).toFixed(1)} GB` : 'Unknown'}
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Data chunk size
                </label>
                <NumericInput
                  label="Data chunk size"
                  value={computationSettings.chunkSize}
                  onChange={(value) => setComputationSettings({ ...computationSettings, chunkSize: value })}
                  min={1000}
                  max={100000}
                  step={1000}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Number of rows to process at once
                </p>
              </div>
            </div>
          </ScientificCard>

          <ScientificCard
            title="Advanced Options"
            description="Fine-tune performance settings"
          >
            <div className="space-y-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={computationSettings.gpuAcceleration}
                  onChange={(e) => setComputationSettings({ ...computationSettings, gpuAcceleration: e.target.checked })}
                  disabled={!systemInfo?.gpuAvailable}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500 disabled:opacity-50"
                />
                <span className={`ml-2 ${!systemInfo?.gpuAvailable ? 'text-gray-400' : ''}`}>
                  Enable GPU acceleration
                </span>
              </label>
              {!systemInfo?.gpuAvailable && (
                <p className="text-xs text-gray-500 ml-6">No compatible GPU detected</p>
              )}

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={computationSettings.cachingEnabled}
                  onChange={(e) => setComputationSettings({ ...computationSettings, cachingEnabled: e.target.checked })}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <span className="ml-2">Enable result caching</span>
              </label>

              <div>
                <label className="block text-sm font-medium mb-2">
                  Compression level (0-9)
                </label>
                <NumericInput
                  label="Compression level (0-9)"
                  value={computationSettings.compressionLevel}
                  onChange={(value) => setComputationSettings({ ...computationSettings, compressionLevel: value })}
                  constraints={{ min: 0, max: 9 }}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Higher values reduce file size but increase processing time
                </p>
              </div>
            </div>
          </ScientificCard>
        </div>
      )}

      {/* System Info */}
      {activeTab === 'system' && systemInfo && (
        <div className="space-y-6">
          <ScientificCard
            title="System Information"
            description="Current system configuration"
          >
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b">
                <span className="text-gray-600">CPU Cores</span>
                <span className="font-medium">{systemInfo.cpuCores}</span>
              </div>
              <div className="flex justify-between py-2 border-b">
                <span className="text-gray-600">Total Memory</span>
                <span className="font-medium">{(systemInfo.totalMemoryMB / 1024).toFixed(1)} GB</span>
              </div>
              <div className="flex justify-between py-2 border-b">
                <span className="text-gray-600">Available Memory</span>
                <span className="font-medium">{(systemInfo.availableMemoryMB / 1024).toFixed(1)} GB</span>
              </div>
              <div className="flex justify-between py-2 border-b">
                <span className="text-gray-600">Python Version</span>
                <span className="font-medium">{systemInfo.pythonVersion}</span>
              </div>
              <div className="flex justify-between py-2 border-b">
                <span className="text-gray-600">Rust Version</span>
                <span className="font-medium">{systemInfo.rustVersion}</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-gray-600">GPU Available</span>
                <span className="font-medium">{systemInfo.gpuAvailable ? 'Yes' : 'No'}</span>
              </div>
            </div>
          </ScientificCard>

          <ScientificCard
            title="Cache Management"
            description="Manage application cache"
          >
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Clear cached data to free up disk space and resolve potential issues.
              </p>
              <Button
                variant="outline"
                onClick={() => invoke('clear_cache')}
              >
                <Database className="w-4 h-4 mr-2" />
                Clear Cache
              </Button>
            </div>
          </ScientificCard>
        </div>
      )}

      {/* Action Buttons */}
      <div className="mt-8 flex justify-between">
        <Button
          variant="outline"
          onClick={handleResetToDefaults}
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          Reset to Defaults
        </Button>
        <div className="flex items-center space-x-3">
          {saved && (
            <span className="text-green-600 text-sm">Settings saved!</span>
          )}
          <Button
            onClick={handleSaveSettings}
            disabled={saving}
          >
            <Save className="w-4 h-4 mr-2" />
            {saving ? 'Saving...' : 'Save Settings'}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Settings;