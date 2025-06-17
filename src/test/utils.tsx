import type { ReactElement } from 'react';
import React from 'react';
import type { RenderOptions, RenderResult } from '@testing-library/react';
import { render, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@/components/providers/ThemeProvider';
import { expect, describe, it } from 'vitest';

// Test IDs for common elements
export const TEST_IDS = {
  // Layout
  HEADER: 'app-header',
  SIDEBAR: 'app-sidebar',
  MAIN_CONTENT: 'main-content',
  STATUS_BAR: 'status-bar',
  
  // Forms
  SUBMIT_BUTTON: 'submit-button',
  CANCEL_BUTTON: 'cancel-button',
  FORM_ERROR: 'form-error',
  FIELD_ERROR: 'field-error',
  
  // Data Display
  LOADING_SPINNER: 'loading-spinner',
  ERROR_MESSAGE: 'error-message',
  EMPTY_STATE: 'empty-state',
  DATA_TABLE: 'data-table',
  
  // Modals
  MODAL: 'modal',
  MODAL_CLOSE: 'modal-close',
  MODAL_CONFIRM: 'modal-confirm',
} as const;

// Mock data generators
export const generateMockDataset = (overrides?: Partial<any>) => ({
  id: 'test-dataset-1',
  name: 'Test Dataset',
  rows: 1000,
  columns: 10,
  missingPercentage: 15.5,
  createdAt: new Date().toISOString(),
  ...overrides,
});

export const generateMockImputationJob = (overrides?: Partial<any>) => ({
  id: 'test-job-1',
  datasetId: 'test-dataset-1',
  method: 'kalman_filter',
  status: 'completed',
  progress: 100,
  startedAt: new Date().toISOString(),
  completedAt: new Date().toISOString(),
  metrics: {
    rmse: 0.123,
    mae: 0.089,
    r2: 0.945,
  },
  ...overrides,
});

export const generateMockTimeSeries = (length: number = 100) => {
  const now = Date.now();
  return Array.from({ length }, (_, i) => ({
    timestamp: new Date(now - (length - i) * 3600000).toISOString(),
    value: Math.sin(i / 10) * 50 + 50 + Math.random() * 10,
    missing: Math.random() > 0.85,
  }));
};

// Custom render function with providers
interface ExtendedRenderOptions extends Omit<RenderOptions, 'queries'> {
  queryClient?: QueryClient;
  route?: string;
}

export function renderWithProviders(
  ui: ReactElement,
  {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    }),
    route = '/',
    ...renderOptions
  }: ExtendedRenderOptions = {}
): RenderResult & { queryClient: QueryClient } {
  window.history.pushState({}, 'Test page', route);

  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <ThemeProvider>
            {children}
          </ThemeProvider>
        </BrowserRouter>
      </QueryClientProvider>
    );
  }

  const result = render(ui, { wrapper: Wrapper, ...renderOptions });
  
  return {
    queryClient,
    ...result,
  };
}

// Async utilities
export const waitForLoadingToFinish = async () => {
  const { waitFor } = await import('@testing-library/react');
  await waitFor(
    () => {
      const loadingElements = document.querySelectorAll('[data-testid="loading-spinner"]');
      expect(loadingElements.length).toBe(0);
    },
    { timeout: 5000 }
  );
};

// Mock Tauri commands
export const mockTauriCommand = (command: string, response: any) => {
  const tauri = (global as any).__TAURI__;
  tauri.invoke.mockImplementation((cmd: string) => {
    if (cmd === command) {
      return Promise.resolve(response);
    }
    return Promise.reject(new Error(`Unknown command: ${cmd}`));
  });
};

// File upload testing utilities
export const createMockFile = (
  content: string,
  fileName: string,
  mimeType: string = 'text/csv'
): File => {
  const blob = new Blob([content], { type: mimeType });
  return new File([blob], fileName, { type: mimeType });
};

export const createMockCSVFile = (rows: number = 100, columns: number = 5) => {
  const headers = Array.from({ length: columns }, (_, i) => `Column_${i + 1}`);
  const data = Array.from({ length: rows }, () =>
    Array.from({ length: columns }, () => Math.random() * 100)
  );
  
  const csvContent = [
    headers.join(','),
    ...data.map(row => row.join(',')),
  ].join('\n');
  
  return createMockFile(csvContent, 'test-data.csv');
};

// Accessibility testing utilities
export const checkAccessibility = async (container: HTMLElement) => {
  // TODO: Implement accessibility testing with vitest-axe or another alternative
  // For now, just do basic checks
  const images = container.querySelectorAll('img');
  images.forEach(img => {
    expect(img).toHaveAttribute('alt');
  });
  
  const buttons = container.querySelectorAll('button');
  buttons.forEach(button => {
    expect(button).toHaveAccessibleName();
  });
};

// Performance testing utilities
export const measureRenderTime = async (
  component: () => ReactElement,
  iterations: number = 10
) => {
  const times: number[] = [];
  
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    const { unmount } = render(component());
    const end = performance.now();
    times.push(end - start);
    unmount();
  }
  
  return {
    average: times.reduce((a, b) => a + b) / times.length,
    min: Math.min(...times),
    max: Math.max(...times),
    median: times.sort((a, b) => a - b)[Math.floor(times.length / 2)],
  };
};

// Snapshot testing utilities
export const createSnapshotTest = (
  componentName: string,
  component: ReactElement,
  states: Record<string, any> = {}
) => {
  describe(`${componentName} Snapshots`, () => {
    it('renders default state correctly', () => {
      const { container } = render(component);
      expect(container.firstChild).toMatchSnapshot();
    });

    Object.entries(states).forEach(([stateName, props]) => {
      it(`renders ${stateName} state correctly`, () => {
        const { container } = render(React.cloneElement(component, props));
        expect(container.firstChild).toMatchSnapshot();
      });
    });
  });
};

// Event testing utilities
export const fireClickEvent = (element: HTMLElement) => {
  fireEvent.mouseDown(element);
  fireEvent.mouseUp(element);
  fireEvent.click(element);
};

// Form testing utilities
export const fillForm = async (
  container: HTMLElement,
  values: Record<string, string | number | boolean>
) => {
  const { fireEvent } = await import('@testing-library/react');
  
  for (const [name, value] of Object.entries(values)) {
    const input = container.querySelector(`[name="${name}"]`) as HTMLInputElement;
    if (!input) {
      throw new Error(`Input with name "${name}" not found`);
    }
    
    if (input.type === 'checkbox') {
      if (input.checked !== value) {
        fireEvent.click(input);
      }
    } else {
      fireEvent.change(input, { target: { value } });
    }
  }
};

// API mocking utilities
export const createMockResponse = <T,>(
  data: T,
  delay: number = 0,
  status: number = 200
): Promise<T> => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (status >= 200 && status < 300) {
        resolve(data);
      } else {
        reject(new Error(`API Error: ${status}`));
      }
    }, delay);
  });
};

// Export everything for easy access
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';