import { invoke } from '@tauri-apps/api/tauri';

export interface ImputationMethod {
  id: string;
  name: string;
  description: string;
  category: 'statistical' | 'machine_learning' | 'deep_learning' | 'hybrid';
  parameters: any[];
}

export interface ImputationJobRequest {
  datasetId: string;
  method: string;
  parameters: Record<string, any>;
}

export interface ImputationJobResponse {
  jobId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress?: number;
  result?: any;
  error?: string;
}

export interface ImputationProgress {
  jobId: string;
  progress: number;
  currentStep: string;
  estimatedTimeRemaining?: number;
}

class ImputationService {
  async getAvailableMethods(): Promise<ImputationMethod[]> {
    try {
      return await invoke<ImputationMethod[]>('get_available_methods');
    } catch (error) {
      console.error('Failed to get imputation methods:', error);
      throw error;
    }
  }

  async runImputation(request: ImputationJobRequest): Promise<ImputationJobResponse> {
    try {
      return await invoke<ImputationJobResponse>('run_imputation', { ...request });
    } catch (error) {
      console.error('Failed to run imputation:', error);
      throw error;
    }
  }

  async runBatchImputation(
    datasetId: string,
    methods: string[],
    parameters: Record<string, any>
  ): Promise<ImputationJobResponse[]> {
    try {
      return await invoke<ImputationJobResponse[]>('run_batch_imputation', {
        datasetId,
        methods,
        parameters
      });
    } catch (error) {
      console.error('Failed to run batch imputation:', error);
      throw error;
    }
  }

  async getJobStatus(jobId: string): Promise<ImputationJobResponse> {
    try {
      return await invoke<ImputationJobResponse>('get_imputation_status', { jobId });
    } catch (error) {
      console.error('Failed to get job status:', error);
      throw error;
    }
  }

  async cancelImputation(jobId: string): Promise<void> {
    try {
      await invoke('cancel_imputation', { jobId });
    } catch (error) {
      console.error('Failed to cancel imputation:', error);
      throw error;
    }
  }

  async validateResults(datasetId: string, imputationId: string): Promise<any> {
    try {
      return await invoke('validate_imputation_results', {
        datasetId,
        imputationId
      });
    } catch (error) {
      console.error('Failed to validate results:', error);
      throw error;
    }
  }

  async compareMethods(
    datasetId: string,
    methodIds: string[],
    metrics: string[]
  ): Promise<any> {
    try {
      return await invoke('compare_methods', {
        datasetId,
        methodIds,
        metrics
      });
    } catch (error) {
      console.error('Failed to compare methods:', error);
      throw error;
    }
  }

  async estimateProcessingTime(
    datasetId: string,
    method: string,
    parameters: Record<string, any>
  ): Promise<number> {
    try {
      return await invoke<number>('estimate_processing_time', {
        datasetId,
        method,
        parameters
      });
    } catch (error) {
      console.error('Failed to estimate processing time:', error);
      throw error;
    }
  }

  async getMethodDocumentation(methodId: string): Promise<string> {
    try {
      return await invoke<string>('get_method_documentation', { methodId });
    } catch (error) {
      console.error('Failed to get method documentation:', error);
      throw error;
    }
  }
}

export default new ImputationService();