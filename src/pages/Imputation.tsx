import React, { useState, useEffect } from 'react';
import { Play, Settings, Info, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Progress } from '@/components/ui/Progress';
import { ScientificCard } from '@/components/layout/ScientificCard';
import { NumericInput } from '@/components/forms/NumericInput';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/store';

interface ImputationMethod {
  id: string;
  name: string;
  description: string;
  category: 'statistical' | 'machine_learning' | 'deep_learning' | 'hybrid';
  parameters: MethodParameter[];
}

interface MethodParameter {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'select';
  default: any;
  description: string;
  options?: string[];
  min?: number;
  max?: number;
}

interface ImputationProgress {
  method: string;
  progress: number;
  current_step: string;
  estimated_time_remaining: number;
}

const Imputation: React.FC = () => {
  const navigate = useNavigate();
  const { currentDataset } = useStore();
  const [methods, setMethods] = useState<ImputationMethod[]>([]);
  const [selectedMethod, setSelectedMethod] = useState<string>('');
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ImputationProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!currentDataset) {
      navigate('/data-import');
      return () => {}; // Return empty cleanup function
    }

    loadMethods();
    
    // Listen for progress updates
    const unlisten = listen<ImputationProgress>('imputation-progress', (event) => {
      setProgress(event.payload);
    });

    return () => {
      unlisten.then(fn => fn());
    };
  }, [currentDataset, navigate]);

  const loadMethods = async () => {
    try {
      const availableMethods = await invoke<ImputationMethod[]>('get_available_methods');
      setMethods(availableMethods);
      if (availableMethods.length > 0) {
        setSelectedMethod(availableMethods[0].id);
        // Initialize parameters with defaults
        const defaultParams: Record<string, any> = {};
        availableMethods[0].parameters.forEach(param => {
          defaultParams[param.name] = param.default;
        });
        setParameters(defaultParams);
      }
    } catch (err) {
      setError('Failed to load imputation methods');
    }
  };

  const handleMethodChange = (methodId: string) => {
    setSelectedMethod(methodId);
    const method = methods.find(m => m.id === methodId);
    if (method) {
      const defaultParams: Record<string, any> = {};
      method.parameters.forEach(param => {
        defaultParams[param.name] = param.default;
      });
      setParameters(defaultParams);
    }
  };

  const handleParameterChange = (name: string, value: any) => {
    setParameters(prev => ({ ...prev, [name]: value }));
  };

  const handleRunImputation = async () => {
    if (!currentDataset || !selectedMethod) return;

    setRunning(true);
    setError(null);
    setProgress(null);

    try {
      await invoke('run_imputation', {
        datasetId: currentDataset.id,
        method: selectedMethod,
        parameters
      });

      // Navigate to results/analysis page
      navigate('/analysis');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Imputation failed');
    } finally {
      setRunning(false);
    }
  };

  const selectedMethodDetails = methods.find(m => m.id === selectedMethod);

  if (!currentDataset) {
    return null;
  }

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <h1 className="text-3xl font-bold mb-6">Imputation</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Dataset Info */}
        <div className="lg:col-span-1">
          <ScientificCard
            title="Dataset"
            description="Current dataset information"
          >
            <div className="space-y-3">
              <div>
                <p className="text-sm text-gray-500">Name</p>
                <p className="font-medium">{currentDataset.info.name}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Shape</p>
                <p className="font-medium">
                  {currentDataset.info.rows.toLocaleString()} × {currentDataset.info.columns}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Missing Values</p>
                <p className="font-medium">
                  {currentDataset.info.missing_count.toLocaleString()} ({currentDataset.info.missing_percentage.toFixed(1)}%)
                </p>
              </div>
            </div>
          </ScientificCard>
        </div>

        {/* Method Selection and Configuration */}
        <div className="lg:col-span-2">
          <ScientificCard
            title="Imputation Method"
            description="Select and configure the imputation algorithm"
          >
            {/* Method Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-2">
                Select Method
              </label>
              <select
                value={selectedMethod}
                onChange={(e) => handleMethodChange(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={running}
              >
                {methods.map(method => (
                  <option key={method.id} value={method.id}>
                    {method.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Method Description */}
            {selectedMethodDetails && (
              <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                <div className="flex items-start">
                  <Info className="w-5 h-5 text-blue-500 mr-2 mt-0.5" />
                  <div>
                    <p className="font-medium text-blue-900 mb-1">
                      {selectedMethodDetails.name}
                    </p>
                    <p className="text-sm text-blue-700">
                      {selectedMethodDetails.description}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Method Parameters */}
            {selectedMethodDetails && selectedMethodDetails.parameters.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-medium mb-4 flex items-center">
                  <Settings className="w-5 h-5 mr-2" />
                  Parameters
                </h3>
                <div className="space-y-4">
                  {selectedMethodDetails.parameters.map(param => (
                    <div key={param.name}>
                      <label className="block text-sm font-medium mb-1">
                        {param.name}
                      </label>
                      <p className="text-xs text-gray-500 mb-2">
                        {param.description}
                      </p>
                      {param.type === 'number' ? (
                        <NumericInput
                          label={param.name}
                          value={parameters[param.name] || param.default}
                          onChange={(value) => handleParameterChange(param.name, value)}
                          min={param.min}
                          max={param.max}
                          disabled={running}
                        />
                      ) : param.type === 'select' ? (
                        <select
                          value={parameters[param.name] || param.default}
                          onChange={(e) => handleParameterChange(param.name, e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          disabled={running}
                        >
                          {param.options?.map(option => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      ) : param.type === 'boolean' ? (
                        <input
                          type="checkbox"
                          checked={parameters[param.name] || param.default}
                          onChange={(e) => handleParameterChange(param.name, e.target.checked)}
                          className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                          disabled={running}
                        />
                      ) : (
                        <input
                          type="text"
                          value={parameters[param.name] || param.default}
                          onChange={(e) => handleParameterChange(param.name, e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                          disabled={running}
                        />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Progress */}
            {running && progress && (
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">{progress.current_step}</span>
                  <span className="text-sm text-gray-500">
                    {progress.progress}% - {Math.ceil(progress.estimated_time_remaining)}s remaining
                  </span>
                </div>
                <Progress value={progress.progress} />
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex justify-end space-x-3">
              <Button
                variant="outline"
                onClick={() => navigate('/data-import')}
                disabled={running}
              >
                Change Dataset
              </Button>
              <Button
                onClick={handleRunImputation}
                disabled={running || !selectedMethod}
              >
                <Play className="w-4 h-4 mr-2" />
                {running ? 'Running...' : 'Run Imputation'}
              </Button>
            </div>
          </ScientificCard>
        </div>
      </div>
    </div>
  );
};

export default Imputation;