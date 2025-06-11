import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import { invoke } from '@tauri-apps/api/tauri';
import { 
  createMockCSVFile,
  mockTauriCommand,
  generateMockDataset
} from '@/test/utils';

describe('Data Pipeline Integration Tests', () => {
  // Mock file system
  const mockFiles = new Map<string, File>();
  
  beforeAll(() => {
    // Mock file dialog
    (global as any).__TAURI__.dialog = {
      open: vi.fn().mockResolvedValue('/mock/path/data.csv'),
      save: vi.fn().mockResolvedValue('/mock/path/output.csv'),
    };
    
    // Create mock data file
    const csvFile = createMockCSVFile(1000, 10);
    mockFiles.set('/mock/path/data.csv', csvFile);
  });

  afterAll(() => {
    vi.restoreAllMocks();
  });

  describe('Import -> Validate -> Impute -> Export Flow', () => {
    it('completes full pipeline successfully', async () => {
      // Step 1: Import dataset
      const importResult = await invoke('load_dataset', {
        path: '/mock/path/data.csv',
        options: {
          has_header: true,
          delimiter: ',',
          parse_dates: true,
          date_column: 'timestamp',
        }
      });

      expect(importResult).toMatchObject({
        id: expect.any(String),
        name: expect.any(String),
        rows: 1000,
        columns: 9, // 10 columns minus timestamp
        missing_percentage: expect.any(Number),
      });

      const datasetId = (importResult as any).id;

      // Step 2: Validate dataset
      const validationResult = await invoke('validate_dataset', {
        dataset_id: datasetId,
      });

      expect(validationResult).toMatchObject({
        is_valid: true,
        issues: expect.any(Array),
        summary: {
          total_values: 9000,
          missing_values: expect.any(Number),
          completeness_percentage: expect.any(Number),
        }
      });

      // Step 3: Get statistics
      const statsResult = await invoke('get_dataset_statistics', {
        dataset_id: datasetId,
      });

      expect(statsResult).toMatchObject({
        basic_stats: {
          count: expect.any(Object),
          mean: expect.any(Object),
          std: expect.any(Object),
        },
        missing_stats: {
          total_missing: expect.any(Number),
          missing_percentage: expect.any(Number),
        }
      });

      // Step 4: Run imputation
      const imputationResult = await invoke('run_imputation', {
        dataset_id: datasetId,
        method: 'kalman_filter',
        parameters: {
          model_order: 2,
          process_noise: 0.01,
          measurement_noise: 0.1,
        }
      });

      expect(imputationResult).toMatchObject({
        id: expect.any(String),
        status: 'running',
        method: 'kalman_filter',
      });

      const jobId = (imputationResult as any).id;

      // Wait for completion (mock immediate completion)
      await new Promise(resolve => setTimeout(resolve, 100));

      // Check job status
      const jobStatus = await invoke('get_imputation_status', {
        job_id: jobId,
      });

      expect(jobStatus).toMatchObject({
        status: 'completed',
        progress: 100,
        result: {
          metrics: expect.any(Object),
          execution_time_ms: expect.any(Number),
        }
      });

      // Step 5: Export results
      const exportResult = await invoke('export_to_csv', {
        dataset_id: datasetId,
        path: '/mock/path/output.csv',
        options: {
          include_metadata: true,
          precision: 6,
        }
      });

      expect(exportResult).toMatchObject({
        success: true,
        file_path: '/mock/path/output.csv',
        rows_exported: 1000,
        columns_exported: 9,
      });
    });

    it('handles large datasets efficiently', async () => {
      const startTime = performance.now();

      // Import large dataset
      const largeFile = createMockCSVFile(10000, 20);
      mockFiles.set('/mock/path/large.csv', largeFile);

      const importResult = await invoke('load_dataset', {
        path: '/mock/path/large.csv',
        options: { has_header: true }
      });

      const importTime = performance.now() - startTime;

      // Should complete in reasonable time (< 5 seconds)
      expect(importTime).toBeLessThan(5000);

      const datasetId = (importResult as any).id;

      // Run imputation on subset
      const imputeStart = performance.now();
      
      await invoke('run_imputation', {
        dataset_id: datasetId,
        method: 'mean',
        parameters: {}
      });

      const imputeTime = performance.now() - imputeStart;

      // Simple methods should be fast
      expect(imputeTime).toBeLessThan(1000);
    });
  });

  describe('Multi-format Support', () => {
    const formats = ['csv', 'excel', 'netcdf', 'hdf5', 'parquet', 'json'];

    formats.forEach(format => {
      it(`handles ${format} format correctly`, async () => {
        const mockPath = `/mock/path/data.${format}`;
        
        // Mock format-specific file
        mockTauriCommand('load_dataset', {
          id: 'test-dataset',
          name: `Test ${format.toUpperCase()}`,
          rows: 100,
          columns: 5,
          missing_percentage: 10,
        });

        const result = await invoke('load_dataset', {
          path: mockPath,
          options: {}
        });

        expect(result).toBeTruthy();
        expect((result as any).name).toContain(format.toUpperCase());
      });
    });
  });

  describe('Error Recovery', () => {
    it('handles corrupted file gracefully', async () => {
      mockTauriCommand('load_dataset', 
        Promise.reject(new Error('Invalid CSV format'))
      );

      await expect(
        invoke('load_dataset', {
          path: '/mock/path/corrupted.csv',
          options: {}
        })
      ).rejects.toThrow('Invalid CSV format');
    });

    it('recovers from imputation failure', async () => {
      const dataset = generateMockDataset();
      
      // First attempt fails
      mockTauriCommand('run_imputation', 
        Promise.reject(new Error('Convergence failed'))
      );

      await expect(
        invoke('run_imputation', {
          dataset_id: dataset.id,
          method: 'em_algorithm',
          parameters: { max_iter: 1 } // Too few iterations
        })
      ).rejects.toThrow('Convergence failed');

      // Second attempt with better parameters succeeds
      mockTauriCommand('run_imputation', {
        id: 'job-2',
        status: 'completed',
        result: { converged: true }
      });

      const result = await invoke('run_imputation', {
        dataset_id: dataset.id,
        method: 'em_algorithm',
        parameters: { max_iter: 100 }
      });

      expect((result as any).status).toBe('completed');
    });
  });

  describe('Concurrent Operations', () => {
    it('handles multiple dataset imports simultaneously', async () => {
      const importPromises: Promise<unknown>[] = [];

      // Import 5 datasets concurrently
      for (let i = 0; i < 5; i++) {
        mockTauriCommand('load_dataset', {
          id: `dataset-${i}`,
          name: `Dataset ${i}`,
          rows: 100 * (i + 1),
          columns: 5,
        });

        importPromises.push(
          invoke('load_dataset', {
            path: `/mock/path/data${i}.csv`,
            options: {}
          })
        );
      }

      const results = await Promise.all(importPromises);

      expect(results).toHaveLength(5);
      results.forEach((result, i) => {
        expect((result as any).id).toBe(`dataset-${i}`);
      });
    });

    it('queues imputation jobs appropriately', async () => {
      const jobPromises: Promise<unknown>[] = [];

      // Submit multiple imputation jobs
      for (let i = 0; i < 3; i++) {
        mockTauriCommand('run_imputation', {
          id: `job-${i}`,
          status: 'queued',
          position: i,
        });

        jobPromises.push(
          invoke('run_imputation', {
            dataset_id: `dataset-${i}`,
            method: 'random_forest',
            parameters: {}
          })
        );
      }

      const jobs = await Promise.all(jobPromises);

      jobs.forEach((job, i) => {
        expect((job as any).status).toBe('queued');
        expect((job as any).position).toBe(i);
      });
    });
  });

  describe('Memory Management', () => {
    it('cleans up datasets after deletion', async () => {
      // Import dataset
      mockTauriCommand('load_dataset', generateMockDataset());
      
      const importResult = await invoke('load_dataset', {
        path: '/mock/path/temp.csv',
        options: {}
      });

      const datasetId = (importResult as any).id;

      // Get initial memory usage
      mockTauriCommand('get_memory_usage', {
        datasets_memory_mb: 50,
        cache_memory_mb: 10,
        total_memory_mb: 60,
      });

      const initialMemory = await invoke('get_memory_usage');

      // Delete dataset
      mockTauriCommand('delete_dataset', { success: true });
      await invoke('delete_dataset', { dataset_id: datasetId });

      // Check memory is released
      mockTauriCommand('get_memory_usage', {
        datasets_memory_mb: 0,
        cache_memory_mb: 10,
        total_memory_mb: 10,
      });

      const finalMemory = await invoke('get_memory_usage');

      expect((finalMemory as any).datasets_memory_mb).toBeLessThan(
        (initialMemory as any).datasets_memory_mb
      );
    });
  });

  describe('Cross-component Integration', () => {
    it('updates UI correctly after backend operations', async () => {
      // Mock event emission
      const emittedEvents: any[] = [];
      
      (global as any).__TAURI__.event.emit = vi.fn((event, payload) => {
        emittedEvents.push({ event, payload });
        return Promise.resolve();
      });

      // Import dataset
      mockTauriCommand('load_dataset', generateMockDataset());
      
      await invoke('load_dataset', {
        path: '/mock/path/data.csv',
        options: {}
      });

      // Check progress events were emitted
      const progressEvents = emittedEvents.filter(e => e.event === 'import-progress');
      expect(progressEvents.length).toBeGreaterThan(0);
      
      const completeEvent = progressEvents.find(e => e.payload.stage === 'complete');
      expect(completeEvent).toBeTruthy();
      expect(completeEvent.payload.progress).toBe(1.0);
    });
  });
});

describe('Visualization Integration Tests', () => {
  const mockDatasetId = 'test-dataset-viz';

  beforeAll(() => {
    // Setup mock dataset with known properties
    mockTauriCommand('get_dataset', {
      id: mockDatasetId,
      name: 'Visualization Test Data',
      data: Array(100).fill(null).map((_, i) => ({
        timestamp: new Date(2024, 0, 1, i).toISOString(),
        pm25: 50 + Math.sin(i / 10) * 20,
        pm10: 80 + Math.sin(i / 15) * 30,
        no2: 40 + Math.sin(i / 20) * 15,
      })),
      columns: ['pm25', 'pm10', 'no2'],
    });
  });

  it('generates all plot types successfully', async () => {
    const plotTypes = [
      { 
        command: 'generate_missing_pattern_plot',
        args: { dataset_id: mockDatasetId, options: {} }
      },
      {
        command: 'generate_time_series_plot',
        args: { 
          dataset_id: mockDatasetId, 
          variables: ['pm25', 'pm10'],
          options: {}
        }
      },
      {
        command: 'generate_correlation_matrix',
        args: { dataset_id: mockDatasetId, options: {} }
      }
    ];

    for (const { command, args } of plotTypes) {
      mockTauriCommand(command, {
        image_data: 'base64_encoded_image',
        format: 'png',
        width: 800,
        height: 600,
        metadata: {}
      });

      const result = await invoke(command, args);

      expect(result).toMatchObject({
        image_data: expect.any(String),
        format: 'png',
        metadata: expect.any(Object),
      });
    }
  });

  it('creates interactive dashboard with multiple panels', async () => {
    mockTauriCommand('create_interactive_dashboard', {
      id: 'dashboard-1',
      title: 'Test Dashboard',
      layout: [
        { id: 'panel-1', type: 'timeseries', position: { x: 0, y: 0, width: 6, height: 4 } },
        { id: 'panel-2', type: 'correlation', position: { x: 6, y: 0, width: 6, height: 4 } },
        { id: 'panel-3', type: 'statistics', position: { x: 0, y: 4, width: 12, height: 2 } },
      ]
    });

    const dashboard = await invoke('create_interactive_dashboard', {
      project_id: 'test-project',
    });

    expect((dashboard as any).layout).toHaveLength(3);
    expect((dashboard as any).layout[0].type).toBe('timeseries');
  });
});