import { useState } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { Button } from '../ui/Button';
import { Card } from '../ui/Card';

export function DebugPanel() {
  const [pingResponse, setPingResponse] = useState<string>('Not tested');
  const [pythonResponse, setPythonResponse] = useState<string>('Not tested');
  const [numpyResponse, setNumpyResponse] = useState<string>('Not tested');
  const [loading, setLoading] = useState<string | null>(null);

  const handlePing = async () => {
    setLoading('ping');
    try {
      const response = await invoke<string>('ping');
      setPingResponse(`✅ Success: ${response}`);
    } catch (error) {
      setPingResponse(`❌ Error: ${error}`);
    } finally {
      setLoading(null);
    }
  };

  const handlePythonCheck = async () => {
    setLoading('python');
    try {
      const response = await invoke<string>('check_python_bridge');
      setPythonResponse(`✅ Success: ${response}`);
    } catch (error) {
      setPythonResponse(`❌ Error: ${error}`);
    } finally {
      setLoading(null);
    }
  };

  const handleNumpyCheck = async () => {
    setLoading('numpy');
    try {
      const response = await invoke<string>('test_numpy');
      setNumpyResponse(`✅ Success: ${response}`);
    } catch (error) {
      setNumpyResponse(`❌ Error: ${error}`);
    } finally {
      setLoading(null);
    }
  };

  return (
    <Card className="p-6 m-4">
      <h2 className="text-2xl font-bold mb-4">Debug Panel</h2>
      <div className="space-y-4">
        <div>
          <Button 
            onClick={handlePing} 
            disabled={loading === 'ping'}
            className="mb-2"
          >
            {loading === 'ping' ? 'Testing...' : 'Test Tauri Bridge (Ping)'}
          </Button>
          <p className="text-sm font-mono bg-gray-100 p-2 rounded">
            {pingResponse}
          </p>
        </div>

        <div>
          <Button 
            onClick={handlePythonCheck} 
            disabled={loading === 'python'}
            variant="secondary"
            className="mb-2"
          >
            {loading === 'python' ? 'Testing...' : 'Test Python Bridge'}
          </Button>
          <p className="text-sm font-mono bg-gray-100 p-2 rounded">
            {pythonResponse}
          </p>
        </div>

        <div>
          <Button 
            onClick={handleNumpyCheck} 
            disabled={loading === 'numpy'}
            variant="secondary"
            className="mb-2"
          >
            {loading === 'numpy' ? 'Testing...' : 'Test Numpy Import'}
          </Button>
          <p className="text-sm font-mono bg-gray-100 p-2 rounded">
            {numpyResponse}
          </p>
        </div>
      </div>
    </Card>
  );
}