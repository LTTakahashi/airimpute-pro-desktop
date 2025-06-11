import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BenchmarkDashboard } from '../benchmarking/BenchmarkDashboard';
import { 
  renderWithProviders, 
  mockTauriCommand,
  generateMockDataset,
  checkAccessibility,
  waitForLoadingToFinish
} from '@/test/utils';

// Mock data
const mockDatasets = [
  generateMockDataset({ id: 'dataset-1', name: 'Air Quality SP 2023' }),
  generateMockDataset({ id: 'dataset-2', name: 'Air Quality SP 2024' }),
];

const mockMethods = [
  { id: 'mean', name: 'Mean Imputation', category: 'simple' },
  { id: 'kalman_filter', name: 'Kalman Filter', category: 'statistical' },
  { id: 'random_forest', name: 'Random Forest', category: 'ml' },
];

const mockBenchmarkResults = {
  results: [
    {
      dataset_id: 'dataset-1',
      method: 'mean',
      metrics: {
        rmse: 12.5,
        mae: 8.3,
        r2: 0.85,
        mape: 15.2,
        execution_time_ms: 125,
      },
    },
    {
      dataset_id: 'dataset-1',
      method: 'kalman_filter',
      metrics: {
        rmse: 8.2,
        mae: 5.1,
        r2: 0.92,
        mape: 10.5,
        execution_time_ms: 450,
      },
    },
  ],
  metadata: {
    total_runs: 2,
    total_time_ms: 575,
    gpu_used: false,
  },
};

describe('BenchmarkDashboard', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    mockTauriCommand('get_datasets', mockDatasets);
    mockTauriCommand('get_imputation_methods', mockMethods);
    mockTauriCommand('check_gpu_availability', { available: true, devices: ['NVIDIA GTX 3080'] });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders all main sections', async () => {
      const { getByText, getByRole } = renderWithProviders(<BenchmarkDashboard />);

      await waitForLoadingToFinish();

      expect(getByText('Benchmarking Dashboard')).toBeInTheDocument();
      expect(getByText('Dataset Selection')).toBeInTheDocument();
      expect(getByText('Method Selection')).toBeInTheDocument();
      expect(getByRole('button', { name: /run benchmark/i })).toBeInTheDocument();
    });

    it('loads datasets and methods on mount', async () => {
      const { getByText } = renderWithProviders(<BenchmarkDashboard />);

      await waitFor(() => {
        expect(getByText('Air Quality SP 2023')).toBeInTheDocument();
        expect(getByText('Air Quality SP 2024')).toBeInTheDocument();
        expect(getByText('Mean Imputation')).toBeInTheDocument();
        expect(getByText('Kalman Filter')).toBeInTheDocument();
      });
    });

    it('displays GPU availability status', async () => {
      const { getByText } = renderWithProviders(<BenchmarkDashboard />);

      await waitFor(() => {
        expect(getByText(/GPU Available/i)).toBeInTheDocument();
        expect(getByText('NVIDIA GTX 3080')).toBeInTheDocument();
      });
    });

    it('passes accessibility checks', async () => {
      const { container } = renderWithProviders(<BenchmarkDashboard />);
      await waitForLoadingToFinish();
      await checkAccessibility(container);
    });
  });

  describe('Interactions', () => {
    it('allows selecting datasets', async () => {
      const { getByLabelText, getByText } = renderWithProviders(<BenchmarkDashboard />);

      await waitForLoadingToFinish();

      const dataset1Checkbox = getByLabelText('Air Quality SP 2023');
      await user.click(dataset1Checkbox);

      expect(dataset1Checkbox).toBeChecked();
      expect(getByText('1 dataset selected')).toBeInTheDocument();
    });

    it('allows selecting methods by category', async () => {
      const { getByText } = renderWithProviders(<BenchmarkDashboard />);

      await waitForLoadingToFinish();

      // Click on Statistical tab
      const statisticalTab = getByText('Statistical');
      await user.click(statisticalTab);

      // Check that Kalman Filter is visible and selectable
      const kalmanCheckbox = getByText('Kalman Filter')
        .closest('label')
        ?.querySelector('input[type="checkbox"]');
      
      expect(kalmanCheckbox).toBeInTheDocument();
      await user.click(kalmanCheckbox!);
      expect(kalmanCheckbox).toBeChecked();
    });

    it('toggles GPU acceleration', async () => {
      const { getByLabelText } = renderWithProviders(<BenchmarkDashboard />);

      await waitForLoadingToFinish();

      const gpuToggle = getByLabelText(/use GPU acceleration/i);
      expect(gpuToggle).toBeChecked(); // Default state

      await user.click(gpuToggle);
      expect(gpuToggle).not.toBeChecked();
    });

    it('validates selections before running benchmark', async () => {
      const { getByRole, getByText } = renderWithProviders(<BenchmarkDashboard />);

      await waitForLoadingToFinish();

      const runButton = getByRole('button', { name: /run benchmark/i });
      await user.click(runButton);

      // Should show validation error
      expect(getByText(/Please select at least one dataset/i)).toBeInTheDocument();
    });

    it('runs benchmark with valid selections', async () => {
      mockTauriCommand('run_benchmark', mockBenchmarkResults);

      const { getByLabelText, getByRole, getByText, queryByText } = renderWithProviders(
        <BenchmarkDashboard />
      );

      await waitForLoadingToFinish();

      // Select dataset and method
      await user.click(getByLabelText('Air Quality SP 2023'));
      await user.click(getByLabelText('Mean Imputation'));

      // Run benchmark
      const runButton = getByRole('button', { name: /run benchmark/i });
      await user.click(runButton);

      // Check loading state
      expect(getByText(/Running benchmark/i)).toBeInTheDocument();

      // Wait for results
      await waitFor(() => {
        expect(queryByText(/Running benchmark/i)).not.toBeInTheDocument();
        expect(getByText(/Benchmark completed/i)).toBeInTheDocument();
      });

      // Verify results are displayed
      expect(getByText('12.5')).toBeInTheDocument(); // RMSE
      expect(getByText('8.3')).toBeInTheDocument(); // MAE
    });
  });

  describe('Results Display', () => {
    beforeEach(() => {
      mockTauriCommand('run_benchmark', mockBenchmarkResults);
    });

    it('displays results in table format', async () => {
      const { getByLabelText, getByRole, getByText } = renderWithProviders(
        <BenchmarkDashboard />
      );

      await waitForLoadingToFinish();

      // Run benchmark
      await user.click(getByLabelText('Air Quality SP 2023'));
      await user.click(getByLabelText('Mean Imputation'));
      await user.click(getByLabelText('Kalman Filter'));
      await user.click(getByRole('button', { name: /run benchmark/i }));

      await waitFor(() => {
        expect(getByText(/Benchmark completed/i)).toBeInTheDocument();
      });

      // Check table headers
      const table = getByRole('table');
      expect(within(table).getByText('Method')).toBeInTheDocument();
      expect(within(table).getByText('RMSE')).toBeInTheDocument();
      expect(within(table).getByText('MAE')).toBeInTheDocument();
      expect(within(table).getByText('R²')).toBeInTheDocument();

      // Check results
      const rows = within(table).getAllByRole('row');
      expect(rows).toHaveLength(3); // Header + 2 results
    });

    it('allows switching between visualization types', async () => {
      const { getByLabelText, getByRole, getByText, getByTestId } = renderWithProviders(
        <BenchmarkDashboard />
      );

      await waitForLoadingToFinish();

      // Run benchmark
      await user.click(getByLabelText('Air Quality SP 2023'));
      await user.click(getByLabelText('Mean Imputation'));
      await user.click(getByRole('button', { name: /run benchmark/i }));

      await waitFor(() => {
        expect(getByText(/Benchmark completed/i)).toBeInTheDocument();
      });

      // Switch to chart view
      const chartTab = getByText('Charts');
      await user.click(chartTab);

      // Should display chart container
      expect(getByTestId('benchmark-charts')).toBeInTheDocument();
    });

    it('exports results in different formats', async () => {
      const mockExport = vi.fn();
      mockTauriCommand('export_benchmark_results', mockExport);

      const { getByLabelText, getByRole, getByText } = renderWithProviders(
        <BenchmarkDashboard />
      );

      await waitForLoadingToFinish();

      // Run benchmark
      await user.click(getByLabelText('Air Quality SP 2023'));
      await user.click(getByLabelText('Mean Imputation'));
      await user.click(getByRole('button', { name: /run benchmark/i }));

      await waitFor(() => {
        expect(getByText(/Benchmark completed/i)).toBeInTheDocument();
      });

      // Click export button
      const exportButton = getByRole('button', { name: /export/i });
      await user.click(exportButton);

      // Select CSV format
      const csvOption = getByText('CSV');
      await user.click(csvOption);

      expect(mockExport).toHaveBeenCalledWith(
        expect.objectContaining({
          format: 'csv',
          results: expect.any(Array),
        })
      );
    });
  });

  describe('Real-time Updates', () => {
    it('shows progress during benchmark execution', async () => {
      let progressCallback: (progress: any) => void;
      
      // Mock Tauri event listener
      const mockListen = vi.fn((event, callback) => {
        if (event === 'benchmark-progress') {
          progressCallback = callback;
        }
        return Promise.resolve(() => {});
      });

      (global as any).__TAURI__.event.listen = mockListen;

      const { getByLabelText, getByRole, getByText } = renderWithProviders(
        <BenchmarkDashboard />
      );

      await waitForLoadingToFinish();

      // Select and run
      await user.click(getByLabelText('Air Quality SP 2023'));
      await user.click(getByLabelText('Mean Imputation'));
      await user.click(getByRole('button', { name: /run benchmark/i }));

      // Simulate progress updates
      act(() => {
        progressCallback({
          payload: {
            current: 1,
            total: 2,
            method: 'mean',
            dataset: 'Air Quality SP 2023',
            progress: 0.5,
          },
        });
      });

      expect(getByText(/Processing: mean on Air Quality SP 2023/i)).toBeInTheDocument();
      expect(getByText('50%')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error when benchmark fails', async () => {
      mockTauriCommand('run_benchmark', Promise.reject(new Error('GPU out of memory')));

      const { getByLabelText, getByRole, getByText } = renderWithProviders(
        <BenchmarkDashboard />
      );

      await waitForLoadingToFinish();

      // Run benchmark
      await user.click(getByLabelText('Air Quality SP 2023'));
      await user.click(getByLabelText('Mean Imputation'));
      await user.click(getByRole('button', { name: /run benchmark/i }));

      await waitFor(() => {
        expect(getByText(/GPU out of memory/i)).toBeInTheDocument();
      });

      // Should show retry button
      expect(getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });

    it('handles empty results gracefully', async () => {
      mockTauriCommand('run_benchmark', { results: [], metadata: {} });

      const { getByLabelText, getByRole, getByText } = renderWithProviders(
        <BenchmarkDashboard />
      );

      await waitForLoadingToFinish();

      // Run benchmark
      await user.click(getByLabelText('Air Quality SP 2023'));
      await user.click(getByLabelText('Mean Imputation'));
      await user.click(getByRole('button', { name: /run benchmark/i }));

      await waitFor(() => {
        expect(getByText(/No results to display/i)).toBeInTheDocument();
      });
    });
  });

  describe('Advanced Features', () => {
    it('saves and loads benchmark configurations', async () => {
      const mockSaveConfig = vi.fn();
      mockTauriCommand('save_benchmark_config', mockSaveConfig);

      const { getByLabelText, getByRole } = renderWithProviders(
        <BenchmarkDashboard />
      );

      await waitForLoadingToFinish();

      // Make selections
      await user.click(getByLabelText('Air Quality SP 2023'));
      await user.click(getByLabelText('Mean Imputation'));
      await user.click(getByLabelText('Kalman Filter'));

      // Save configuration
      const saveButton = getByRole('button', { name: /save configuration/i });
      await user.click(saveButton);

      // Enter config name
      const nameInput = getByLabelText(/configuration name/i);
      await user.type(nameInput, 'My Benchmark Config');

      const confirmButton = getByRole('button', { name: /save/i });
      await user.click(confirmButton);

      expect(mockSaveConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'My Benchmark Config',
          datasets: ['dataset-1'],
          methods: ['mean', 'kalman_filter'],
          useGPU: true,
        })
      );
    });

    it('compares results across multiple runs', async () => {
      const { getByRole, getByText } = renderWithProviders(<BenchmarkDashboard />);

      await waitForLoadingToFinish();

      // Assume we have previous results loaded
      const compareButton = getByRole('button', { name: /compare with previous/i });
      await user.click(compareButton);

      // Should show comparison view
      expect(getByText(/Comparison View/i)).toBeInTheDocument();
    });
  });
});