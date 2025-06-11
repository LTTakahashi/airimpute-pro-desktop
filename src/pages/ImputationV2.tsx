import React, { useState, useEffect } from 'react';
import { Play, Settings, Info, AlertCircle, XCircle, CheckCircle, Clock } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Progress } from '@/components/ui/Progress';
import { ScientificCard } from '@/components/layout/ScientificCard';
import { NumericInput } from '@/components/forms/NumericInput';
import { Alert } from '@/components/ui/Alert';
import { invoke } from '@tauri-apps/api/tauri';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/store';

interface ImputationMethod {
  id: string;
  name: string;
  description: string;
  category: string;
  complexity: string;
  suitable_for: string[];
  parameters: Record<string, any>;
  requires_gpu: boolean;
}

interface ValidationResult {
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  summary: {
    n_rows: number;
    n_cols: number;
    numeric_columns: string[];
    missing_values: Record<string, number>;
    missing_percentage: Record<string, number>;
    memory_mb: number;
  };
}

interface ImputationJob {
  job_id: string;
  dataset_id: string;
  method: string;
  status: string;
  progress: number;
  eta_seconds?: number;
  created_at: string;
  message?: string;
}

interface ProgressEvent {
  job_id: string;
  progress: number;
  message: string;
  elapsed_seconds: number;
  eta_seconds?: number;
}

interface ErrorEvent {
  job_id: string;
  error: string;
  code: string;
  suggestion?: string;
}

const ImputationV2: React.FC = () => {
  const navigate = useNavigate();
  const { currentDataset } = useStore();
  
  // State
  const [methods, setMethods] = useState<ImputationMethod[]>([]);
  const [selectedMethod, setSelectedMethod] = useState<string>('');
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [currentJob, setCurrentJob] = useState<ImputationJob | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState<ErrorEvent | null>(null);
  const [loading, setLoading] = useState(false);
  const [estimatedTime, setEstimatedTime] = useState<string>('');

  // Event listeners
  useEffect(() => {
    if (!currentDataset) {
      navigate('/data-import');
      return () => {}; // Return empty cleanup function
    }

    let unlisteners: UnlistenFn[] = [];

    const setupListeners = async () => {
      // Progress updates
      unlisteners.push(await listen<ProgressEvent>('imputation:progress', (event) => {
        setProgress(event.payload);
      }));

      // Job started
      unlisteners.push(await listen<{ job_id: string; method: string }>('imputation:started', (event) => {
        console.log('Imputation started:', event.payload);
      }));

      // Job completed
      unlisteners.push(await listen<{ job_id: string; metadata: any }>('imputation:completed', (event) => {
        setCurrentJob(prev => prev ? { ...prev, status: 'completed', progress: 100 } : null);
        setProgress(null);
        // Navigate to results
        setTimeout(() => navigate('/analysis'), 1000);
      }));

      // Job failed
      unlisteners.push(await listen<{ job_id: string; error: string }>('imputation:failed', (event) => {
        setCurrentJob(prev => prev ? { ...prev, status: 'failed' } : null);
        setProgress(null);
      }));

      // Error with details
      unlisteners.push(await listen<ErrorEvent>('imputation:error', (event) => {
        setError(event.payload);
      }));
    };

    // Load initial data
    loadMethods();
    validateData();
    setupListeners();

    // Cleanup
    return () => {
      unlisteners.forEach(fn => fn());
    };
  }, [currentDataset, navigate]);

  const loadMethods = async () => {
    try {
      const availableMethods = await invoke<ImputationMethod[]>('get_imputation_methods');
      setMethods(availableMethods);
      
      if (availableMethods.length > 0) {
        // Select first simple method by default
        const defaultMethod = availableMethods.find(m => m.category === 'Simple') || availableMethods[0];
        setSelectedMethod(defaultMethod.id);
        setParameters(extractDefaultParameters(defaultMethod.parameters));
      }
    } catch (err) {
      console.error('Failed to load methods:', err);
      setError({
        job_id: '',
        error: 'Failed to load imputation methods',
        code: 'LOAD_ERROR',
      });
    }
  };

  const validateData = async () => {
    if (!currentDataset) return;

    try {
      const result = await invoke<ValidationResult>('validate_imputation_data', {
        datasetId: currentDataset.id,
      });
      setValidation(result);
    } catch (err) {
      console.error('Validation failed:', err);
    }
  };

  const extractDefaultParameters = (paramDefs: Record<string, any>): Record<string, any> => {
    const defaults: Record<string, any> = {};
    Object.entries(paramDefs).forEach(([key, def]) => {
      if (def && typeof def === 'object' && 'default' in def) {
        defaults[key] = def.default;
      }
    });
    return defaults;
  };

  const handleMethodChange = (methodId: string) => {
    setSelectedMethod(methodId);
    const method = methods.find(m => m.id === methodId);
    if (method) {
      setParameters(extractDefaultParameters(method.parameters));
      estimateTime(methodId);
    }
  };

  const handleParameterChange = (name: string, value: any) => {
    setParameters(prev => ({ ...prev, [name]: value }));
  };

  const estimateTime = async (methodId?: string) => {
    if (!currentDataset) return;

    try {
      const result = await invoke<{
        estimated_ms: number;
        estimated_readable: string;
        confidence: number;
      }>('estimate_imputation_time', {
        datasetId: currentDataset.id,
        method: methodId || selectedMethod,
      });
      
      setEstimatedTime(result.estimated_readable);
    } catch (err) {
      console.error('Failed to estimate time:', err);
    }
  };

  const handleRunImputation = async () => {
    if (!currentDataset || !selectedMethod) return;

    // Check validation
    if (validation && !validation.is_valid) {
      const criticalErrors = validation.errors.filter(e => 
        !e.includes('warning') && !e.includes('suggestion')
      );
      if (criticalErrors.length > 0) {
        setError({
          job_id: '',
          error: 'Data validation failed. Please fix errors before proceeding.',
          code: 'VALIDATION_ERROR',
        });
        return;
      }
    }

    setLoading(true);
    setError(null);
    setProgress(null);

    try {
      const job = await invoke<ImputationJob>('run_imputation_v2', {
        datasetId: currentDataset.id,
        method: selectedMethod,
        parameters,
      });
      
      setCurrentJob(job);
    } catch (err) {
      setError({
        job_id: '',
        error: err instanceof Error ? err.message : 'Failed to start imputation',
        code: 'START_ERROR',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCancelImputation = async () => {
    if (!currentJob) return;

    try {
      await invoke('cancel_imputation', {
        jobId: currentJob.job_id,
      });
      setCurrentJob(null);
      setProgress(null);
    } catch (err) {
      console.error('Failed to cancel:', err);
    }
  };

  const selectedMethodDetails = methods.find(m => m.id === selectedMethod);
  const isRunning = !!(currentJob && (currentJob.status === 'pending' || currentJob.status === 'running'));

  if (!currentDataset) {
    return null;
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <h1 className="text-3xl font-bold mb-6">Data Imputation</h1>

      {/* Validation Alerts */}
      {validation && !validation.is_valid && (
        <Alert variant="error" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <div>
            <h4 className="font-semibold">Data Validation Issues</h4>
            <ul className="list-disc list-inside mt-2">
              {validation.errors.map((error, idx) => (
                <li key={idx} className="text-sm">{error}</li>
              ))}
            </ul>
          </div>
        </Alert>
      )}

      {validation && validation.warnings.length > 0 && (
        <Alert variant="warning" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <div>
            <h4 className="font-semibold">Warnings</h4>
            <ul className="list-disc list-inside mt-2">
              {validation.warnings.map((warning, idx) => (
                <li key={idx} className="text-sm">{warning}</li>
              ))}
            </ul>
          </div>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Dataset Info */}
        <div className="lg:col-span-1">
          <ScientificCard
            title="Dataset Overview"
            description="Current dataset information and statistics"
          >
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-500">Dataset</p>
                <p className="font-medium">{currentDataset.info.name}</p>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Rows</p>
                  <p className="font-medium">{currentDataset.info.rows.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Columns</p>
                  <p className="font-medium">{currentDataset.info.columns}</p>
                </div>
              </div>

              <div>
                <p className="text-sm text-gray-500">Missing Values</p>
                <p className="font-medium">
                  {currentDataset.info.missing_count.toLocaleString()} 
                  <span className="text-sm text-gray-500 ml-1">
                    ({currentDataset.info.missing_percentage.toFixed(1)}%)
                  </span>
                </p>
              </div>

              {validation && validation.summary.memory_mb && (
                <div>
                  <p className="text-sm text-gray-500">Memory Usage</p>
                  <p className="font-medium">{validation.summary.memory_mb.toFixed(1)} MB</p>
                </div>
              )}

              {estimatedTime && (
                <div>
                  <p className="text-sm text-gray-500">Estimated Time</p>
                  <p className="font-medium flex items-center">
                    <Clock className="w-4 h-4 mr-1" />
                    {estimatedTime}
                  </p>
                </div>
              )}
            </div>
          </ScientificCard>

          {/* Missing Value Details */}
          {validation && validation.summary.missing_percentage && (
            <ScientificCard
              title="Missing Values by Column"
              description="Percentage of missing values"
              className="mt-6"
            >
              <div className="space-y-2">
                {Object.entries(validation.summary.missing_percentage)
                  .filter(([_, pct]) => pct > 0)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 5)
                  .map(([col, pct]) => (
                    <div key={col} className="flex justify-between items-center">
                      <span className="text-sm truncate mr-2">{col}</span>
                      <div className="flex items-center">
                        <div className="w-20 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className="bg-orange-500 h-2 rounded-full"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-500 w-12 text-right">
                          {pct.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>
            </ScientificCard>
          )}
        </div>

        {/* Method Selection and Configuration */}
        <div className="lg:col-span-2">
          <ScientificCard
            title="Imputation Configuration"
            description="Select and configure the imputation method"
          >
            {/* Method Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-2">
                Imputation Method
              </label>
              <select
                value={selectedMethod}
                onChange={(e) => handleMethodChange(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isRunning}
              >
                <option value="">Select a method</option>
                {methods.map(method => (
                  <option key={method.id} value={method.id}>
                    {method.name} ({method.category})
                  </option>
                ))}
              </select>
            </div>

            {/* Method Details */}
            {selectedMethodDetails && (
              <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                <div className="flex items-start">
                  <Info className="w-5 h-5 text-blue-500 mr-2 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-medium text-blue-900 mb-1">
                      {selectedMethodDetails.name}
                    </p>
                    <p className="text-sm text-blue-700 mb-2">
                      {selectedMethodDetails.description}
                    </p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                        Complexity: {selectedMethodDetails.complexity}
                      </span>
                      {selectedMethodDetails.requires_gpu && (
                        <span className="text-xs px-2 py-1 bg-purple-100 text-purple-700 rounded">
                          GPU Recommended
                        </span>
                      )}
                      {selectedMethodDetails.suitable_for.map((use, idx) => (
                        <span key={idx} className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded">
                          {use}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Method Parameters */}
            {selectedMethodDetails && Object.keys(selectedMethodDetails.parameters).length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-medium mb-4 flex items-center">
                  <Settings className="w-5 h-5 mr-2" />
                  Parameters
                </h3>
                <div className="space-y-4">
                  {Object.entries(selectedMethodDetails.parameters).map(([key, paramDef]) => {
                    if (!paramDef || typeof paramDef !== 'object') return null;
                    
                    return (
                      <div key={key}>
                        <label className="block text-sm font-medium mb-1">
                          {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </label>
                        {paramDef.description && (
                          <p className="text-xs text-gray-500 mb-2">
                            {paramDef.description}
                          </p>
                        )}
                        
                        {paramDef.type === 'int' || paramDef.type === 'number' ? (
                          <NumericInput
                            label={paramDef.label}
                            value={parameters[key] ?? paramDef.default}
                            onChange={(value) => handleParameterChange(key, value)}
                            min={paramDef.min}
                            max={paramDef.max}
                            disabled={isRunning}
                          />
                        ) : paramDef.options ? (
                          <select
                            value={parameters[key] ?? paramDef.default}
                            onChange={(e) => handleParameterChange(key, e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            disabled={isRunning}
                          >
                            {paramDef.options.map((option: string) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="text"
                            value={parameters[key] ?? paramDef.default ?? ''}
                            onChange={(e) => handleParameterChange(key, e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                            disabled={isRunning}
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Progress */}
            {isRunning && progress && (
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">{progress.message}</span>
                  <span className="text-sm text-gray-500">
                    {progress.progress.toFixed(0)}%
                    {progress.eta_seconds && ` - ${Math.ceil(progress.eta_seconds)}s remaining`}
                  </span>
                </div>
                <Progress value={progress.progress} className="h-2" />
                <p className="text-xs text-gray-500 mt-1">
                  Elapsed: {Math.floor(progress.elapsed_seconds)}s
                </p>
              </div>
            )}

            {/* Success Message */}
            {currentJob && currentJob.status === 'completed' && (
              <Alert variant="success" className="mb-6">
                <CheckCircle className="h-4 w-4" />
                <span>Imputation completed successfully! Redirecting to results...</span>
              </Alert>
            )}

            {/* Error Message */}
            {error && (
              <Alert variant="error" className="mb-6">
                <XCircle className="h-4 w-4" />
                <div>
                  <p className="font-semibold">{error.error}</p>
                  {error.suggestion && (
                    <p className="text-sm mt-1">{error.suggestion}</p>
                  )}
                </div>
              </Alert>
            )}

            {/* Action Buttons */}
            <div className="flex justify-between">
              <Button
                variant="outline"
                onClick={() => navigate('/data-import')}
                disabled={isRunning}
              >
                Change Dataset
              </Button>
              
              <div className="flex space-x-3">
                {isRunning ? (
                  <Button
                    variant="outline"
                    onClick={handleCancelImputation}
                  >
                    <XCircle className="w-4 h-4 mr-2" />
                    Cancel
                  </Button>
                ) : (
                  <Button
                    onClick={handleRunImputation}
                    disabled={!selectedMethod || loading || (validation && !validation.is_valid)}
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Run Imputation
                  </Button>
                )}
              </div>
            </div>
          </ScientificCard>
        </div>
      </div>
    </div>
  );
};

export default ImputationV2;