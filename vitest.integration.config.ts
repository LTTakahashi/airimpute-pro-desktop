/**
 * Vitest configuration for integration tests
 * Following CLAUDE.md specifications for cross-platform testing
 */

import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';
import { platform } from 'os';

const IS_WINDOWS = platform() === 'win32';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'happy-dom',
    globals: true,
    setupFiles: ['./src/test/setup.integration.ts'],
    include: ['src/**/*.integration.{test,spec}.{ts,tsx}', 'src/__tests__/integration/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['node_modules', 'dist', 'build', 'src-tauri'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/mockData.ts'
      ]
    },
    // Longer timeouts for integration tests, especially on Windows
    testTimeout: IS_WINDOWS ? 120000 : 60000,
    hookTimeout: IS_WINDOWS ? 120000 : 60000,
    // Reduce parallelism for integration tests to avoid resource conflicts
    maxConcurrency: IS_WINDOWS ? 1 : 2,
    maxWorkers: IS_WINDOWS ? 1 : '50%',
    // Retry flaky tests on Windows
    retry: IS_WINDOWS ? 2 : 1,
    // Pool options for better isolation
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: IS_WINDOWS,
        isolate: true,
      }
    },
    // Reporter configuration
    reporters: IS_WINDOWS ? ['verbose'] : ['default'],
    // Ensure proper cleanup between tests
    restoreMocks: true,
    clearMocks: true,
    mockReset: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  // Optimization for Windows
  optimizeDeps: {
    include: ['@tauri-apps/api', 'react', 'react-dom'],
    exclude: ['@tauri-apps/cli']
  }
});