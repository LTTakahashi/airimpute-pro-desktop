/**
 * Integration test setup for Windows and cross-platform compatibility
 * Following CLAUDE.md specifications
 */

import '@testing-library/jest-dom';
import { expect, afterEach, vi, beforeAll, afterAll } from 'vitest';
import { cleanup } from '@testing-library/react';
import * as matchers from '@testing-library/jest-dom/matchers';
import { platform } from 'os';

// Extend Vitest's expect with jest-dom matchers
expect.extend(matchers);

// Platform detection
const IS_WINDOWS = platform() === 'win32';
const IS_MAC = platform() === 'darwin';
const IS_LINUX = platform() === 'linux';

// Windows-specific timeout adjustments
const BASE_TIMEOUT = IS_WINDOWS ? 30000 : 15000;
const CLEANUP_TIMEOUT = IS_WINDOWS ? 5000 : 2000;

// Cleanup after each test with proper timeout
afterEach(async () => {
  cleanup();
  vi.clearAllMocks();
  
  // Give Windows extra time for cleanup
  if (IS_WINDOWS) {
    await new Promise(resolve => setTimeout(resolve, 100));
  }
});

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Enhanced Tauri API mock for integration tests
const mockTauriState = {
  datasets: new Map(),
  imputationJobs: new Map(),
  memoryUsage: {
    datasets_memory_mb: 0,
    cache_memory_mb: 0,
    total_memory_mb: 0,
  },
  eventListeners: new Map<string, Set<Function>>(),
};

// Mock file system operations
const mockFileSystem = {
  files: new Map<string, any>(),
  directories: new Map<string, string[]>(),
};

// Initialize mock directories
mockFileSystem.directories.set('/', ['mock']);
mockFileSystem.directories.set('/mock', ['path']);
mockFileSystem.directories.set('/mock/path', []);

// Mock Tauri invoke with realistic behavior
const mockInvoke = vi.fn(async (cmd: string, args?: any) => {
  // Simulate async delay for Windows
  if (IS_WINDOWS) {
    await new Promise(resolve => setTimeout(resolve, 10));
  }

  switch (cmd) {
    case 'load_dataset': {
      const datasetId = `dataset-${Date.now()}`;
      const dataset = {
        id: datasetId,
        name: args.path.split('/').pop() || 'Unknown',
        rows: 1000,
        columns: 9,
        missing_percentage: Math.random() * 20,
        path: args.path,
      };
      mockTauriState.datasets.set(datasetId, dataset);
      mockTauriState.memoryUsage.datasets_memory_mb += 50;
      mockTauriState.memoryUsage.total_memory_mb += 50;
      
      // Emit progress events
      const emitProgress = (stage: string, progress: number) => {
        const listeners = mockTauriState.eventListeners.get('import-progress');
        if (listeners) {
          listeners.forEach(fn => fn({ stage, progress }));
        }
      };
      
      emitProgress('reading', 0.3);
      emitProgress('parsing', 0.6);
      emitProgress('validating', 0.9);
      emitProgress('complete', 1.0);
      
      return dataset;
    }
    
    case 'validate_dataset': {
      const dataset = mockTauriState.datasets.get(args.dataset_id);
      if (!dataset) throw new Error('Dataset not found');
      
      return {
        is_valid: true,
        issues: [],
        summary: {
          total_values: dataset.rows * dataset.columns,
          missing_values: Math.floor(dataset.rows * dataset.columns * dataset.missing_percentage / 100),
          completeness_percentage: 100 - dataset.missing_percentage,
        }
      };
    }
    
    case 'get_dataset_statistics': {
      const dataset = mockTauriState.datasets.get(args.dataset_id);
      if (!dataset) throw new Error('Dataset not found');
      
      return {
        basic_stats: {
          count: {},
          mean: {},
          std: {},
        },
        missing_stats: {
          total_missing: Math.floor(dataset.rows * dataset.columns * dataset.missing_percentage / 100),
          missing_percentage: dataset.missing_percentage,
        }
      };
    }
    
    case 'run_imputation': {
      const jobId = `job-${Date.now()}`;
      const job = {
        id: jobId,
        status: 'running',
        method: args.method,
        dataset_id: args.dataset_id,
        progress: 0,
      };
      mockTauriState.imputationJobs.set(jobId, job);
      
      // Simulate job completion
      setTimeout(() => {
        job.status = 'completed';
        job.progress = 100;
        (job as any).result = {
          metrics: {
            rmse: Math.random() * 10,
            mae: Math.random() * 5,
          },
          execution_time_ms: Math.random() * 1000,
        };
      }, IS_WINDOWS ? 200 : 100);
      
      return job;
    }
    
    case 'get_imputation_status': {
      const job = mockTauriState.imputationJobs.get(args.job_id);
      if (!job) throw new Error('Job not found');
      return job;
    }
    
    case 'export_to_csv': {
      const dataset = mockTauriState.datasets.get(args.dataset_id);
      if (!dataset) throw new Error('Dataset not found');
      
      mockFileSystem.files.set(args.path, {
        content: 'CSV export content',
        size: dataset.rows * dataset.columns * 8,
      });
      
      return {
        success: true,
        file_path: args.path,
        rows_exported: dataset.rows,
        columns_exported: dataset.columns,
      };
    }
    
    case 'get_memory_usage': {
      return { ...mockTauriState.memoryUsage };
    }
    
    case 'delete_dataset': {
      const dataset = mockTauriState.datasets.get(args.dataset_id);
      if (dataset) {
        mockTauriState.datasets.delete(args.dataset_id);
        mockTauriState.memoryUsage.datasets_memory_mb = Math.max(0, mockTauriState.memoryUsage.datasets_memory_mb - 50);
        mockTauriState.memoryUsage.total_memory_mb = Math.max(0, mockTauriState.memoryUsage.total_memory_mb - 50);
      }
      return { success: true };
    }
    
    case 'get_dataset': {
      return mockTauriState.datasets.get(args.dataset_id);
    }
    
    case 'generate_missing_pattern_plot':
    case 'generate_time_series_plot':
    case 'generate_correlation_matrix': {
      return {
        image_data: 'base64_encoded_image_data_here',
        format: 'png',
        width: 800,
        height: 600,
        metadata: {
          generated_at: new Date().toISOString(),
          method: cmd,
        }
      };
    }
    
    case 'create_interactive_dashboard': {
      return {
        id: 'dashboard-1',
        title: 'Test Dashboard',
        layout: [
          { id: 'panel-1', type: 'timeseries', position: { x: 0, y: 0, width: 6, height: 4 } },
          { id: 'panel-2', type: 'correlation', position: { x: 6, y: 0, width: 6, height: 4 } },
          { id: 'panel-3', type: 'statistics', position: { x: 0, y: 4, width: 12, height: 2 } },
        ]
      };
    }
    
    default:
      throw new Error(`Unknown command: ${cmd}`);
  }
});

// Mock Tauri event system
const mockEvent = {
  listen: vi.fn((event: string, handler: Function) => {
    if (!mockTauriState.eventListeners.has(event)) {
      mockTauriState.eventListeners.set(event, new Set());
    }
    mockTauriState.eventListeners.get(event)!.add(handler);
    
    return Promise.resolve(() => {
      mockTauriState.eventListeners.get(event)?.delete(handler);
    });
  }),
  
  emit: vi.fn((event: string, payload?: any) => {
    const listeners = mockTauriState.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(fn => fn(payload));
    }
    return Promise.resolve();
  }),
  
  once: vi.fn((event: string, handler: Function) => {
    const wrappedHandler = (payload: any) => {
      handler(payload);
      mockTauriState.eventListeners.get(event)?.delete(wrappedHandler);
    };
    return mockEvent.listen(event, wrappedHandler);
  }),
};

// Mock file dialog with Windows path handling
const mockDialog = {
  open: vi.fn(async (options?: any) => {
    // Return Windows-style paths on Windows
    if (IS_WINDOWS) {
      return 'C:\\mock\\path\\data.csv';
    }
    return '/mock/path/data.csv';
  }),
  
  save: vi.fn(async (options?: any) => {
    if (IS_WINDOWS) {
      return 'C:\\mock\\path\\output.csv';
    }
    return '/mock/path/output.csv';
  }),
};

// Mock path module for cross-platform compatibility
const mockPath = {
  sep: IS_WINDOWS ? '\\' : '/',
  join: (...parts: string[]) => {
    return parts.join(IS_WINDOWS ? '\\' : '/');
  },
  normalize: (path: string) => {
    if (IS_WINDOWS) {
      return path.replace(/\//g, '\\');
    }
    return path.replace(/\\/g, '/');
  },
};

// Setup global Tauri mocks
(global as any).__TAURI__ = {
  invoke: mockInvoke,
  event: mockEvent,
  dialog: mockDialog,
  path: mockPath,
  window: {
    appWindow: {
      emit: mockEvent.emit,
      listen: mockEvent.listen,
    },
  },
};

(global as any).__TAURI_IPC__ = vi.fn();

// Mock crypto.randomUUID
if (!global.crypto) {
  global.crypto = {} as any;
}
global.crypto.randomUUID = () => ('test-uuid-' + Math.random().toString(36).substring(7)) as `${string}-${string}-${string}-${string}-${string}`;

// Setup performance monitoring for Windows
let performanceWarnings: string[] = [];

beforeAll(() => {
  if (IS_WINDOWS) {
    // Monitor long-running operations
    const originalSetTimeout = global.setTimeout;
    global.setTimeout = ((fn: Function, delay: number, ...args: any[]) => {
      if (delay > 10000) {
        performanceWarnings.push(`Long timeout detected: ${delay}ms`);
      }
      return originalSetTimeout(fn, delay, ...args);
    }) as any;
  }
});

afterAll(() => {
  if (performanceWarnings.length > 0) {
    console.warn('Performance warnings:', performanceWarnings);
  }
  
  // Clean up all mocks
  vi.restoreAllMocks();
  mockTauriState.datasets.clear();
  mockTauriState.imputationJobs.clear();
  mockTauriState.eventListeners.clear();
  mockFileSystem.files.clear();
});

// Export utilities for tests
export {
  mockTauriState,
  mockFileSystem,
  IS_WINDOWS,
  IS_MAC,
  IS_LINUX,
  BASE_TIMEOUT,
  CLEANUP_TIMEOUT,
};

// Suppress console errors during tests unless explicitly needed
const originalError = console.error;
beforeAll(() => {
  console.error = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      (args[0].includes('Warning: ReactDOM.render') ||
       args[0].includes('Warning: `ReactDOMTestUtils.act`') ||
       args[0].includes('Not implemented: HTMLFormElement.prototype.requestSubmit'))
    ) {
      return;
    }
    originalError.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
});