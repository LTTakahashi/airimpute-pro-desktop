import { invoke } from '@tauri-apps/api/tauri';
import { open, save } from '@tauri-apps/api/dialog';

export interface DatasetLoadOptions {
  delimiter?: string;
  parse_dates?: boolean;
  date_column?: string;
  infer_datetime_format?: boolean;
  chunk_size?: number;
}

export interface DatasetInfo {
  id: string;
  name: string;
  path: string;
  rows: number;
  columns: number;
  missing_count: number;
  missing_percentage: number;
  file_size: number;
  column_names: string[];
  column_types: Record<string, string>;
  date_range?: {
    start: string;
    end: string;
  };
}

export interface DatasetStatistics {
  basic_stats: Record<string, {
    count: number;
    mean: number;
    std: number;
    min: number;
    max: number;
    missing: number;
  }>;
  missing_patterns: {
    pattern: string;
    count: number;
    percentage: number;
  }[];
  correlations: Record<string, Record<string, number>>;
}

export interface DataPreview {
  columns: string[];
  data: any[][];
  total_rows: number;
}

class DataService {
  async loadDataset(path: string, options?: DatasetLoadOptions): Promise<DatasetInfo> {
    try {
      return await invoke<DatasetInfo>('load_dataset', {
        path,
        options: options || {}
      });
    } catch (error) {
      console.error('Failed to load dataset:', error);
      throw error;
    }
  }

  async saveDataset(datasetId: string, path?: string): Promise<void> {
    try {
      const savePath = path || await save({
        filters: [{
          name: 'CSV Files',
          extensions: ['csv']
        }]
      });

      if (savePath) {
        await invoke('save_dataset', {
          datasetId,
          path: savePath
        });
      }
    } catch (error) {
      console.error('Failed to save dataset:', error);
      throw error;
    }
  }

  async validateDataset(datasetId: string): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    try {
      return await invoke('validate_dataset', { datasetId });
    } catch (error) {
      console.error('Failed to validate dataset:', error);
      throw error;
    }
  }

  async getDatasetStatistics(datasetId: string): Promise<DatasetStatistics> {
    try {
      return await invoke<DatasetStatistics>('get_dataset_statistics', { datasetId });
    } catch (error) {
      console.error('Failed to get dataset statistics:', error);
      throw error;
    }
  }

  async previewDataset(
    datasetId: string,
    startRow?: number,
    endRow?: number
  ): Promise<DataPreview> {
    try {
      return await invoke<DataPreview>('preview_dataset', {
        datasetId,
        startRow: startRow || 0,
        endRow: endRow || 100
      });
    } catch (error) {
      console.error('Failed to preview dataset:', error);
      throw error;
    }
  }

  async importFromMultipleSources(
    sources: string[],
    mergeStrategy: 'concat' | 'merge' | 'join',
    options?: any
  ): Promise<DatasetInfo> {
    try {
      return await invoke<DatasetInfo>('import_from_multiple_sources', {
        sources,
        mergeStrategy,
        options
      });
    } catch (error) {
      console.error('Failed to import from multiple sources:', error);
      throw error;
    }
  }

  async selectDataFile(): Promise<string | null> {
    try {
      const selected = await open({
        multiple: false,
        filters: [{
          name: 'Data Files',
          extensions: ['csv', 'xlsx', 'xls', 'json', 'parquet', 'hdf5', 'h5', 'nc']
        }]
      });
      
      return selected as string | null;
    } catch (error) {
      console.error('Failed to select file:', error);
      return null;
    }
  }

  async analyzeDataQuality(datasetId: string): Promise<{
    quality_score: number;
    issues: {
      severity: 'low' | 'medium' | 'high';
      type: string;
      description: string;
      affected_columns?: string[];
    }[];
    recommendations: string[];
  }> {
    try {
      return await invoke('analyze_data_quality', { datasetId });
    } catch (error) {
      console.error('Failed to analyze data quality:', error);
      throw error;
    }
  }

  async detectOutliers(
    datasetId: string,
    method: 'iqr' | 'zscore' | 'isolation_forest',
    threshold?: number
  ): Promise<{
    outliers: Record<string, number[]>;
    total_outliers: number;
  }> {
    try {
      return await invoke('detect_outliers', {
        datasetId,
        method,
        threshold
      });
    } catch (error) {
      console.error('Failed to detect outliers:', error);
      throw error;
    }
  }

  async removeDataset(datasetId: string): Promise<void> {
    try {
      await invoke('remove_dataset', { datasetId });
    } catch (error) {
      console.error('Failed to remove dataset:', error);
      throw error;
    }
  }
}

export default new DataService();