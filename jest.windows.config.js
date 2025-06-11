/**
 * Jest configuration for Windows
 * Following CLAUDE.md specifications for Windows testing
 */

module.exports = {
  ...require('./jest.config.js'),
  testTimeout: 30000, // Windows needs longer timeouts
  maxWorkers: '50%', // Prevent Windows resource exhaustion
  testEnvironment: 'node',
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
    '/__tests__/unix-only/', // Skip Unix-specific tests
    '/src-tauri/'
  ],
  moduleNameMapper: {
    // Handle Windows path separators
    '^@/(.*)$': '<rootDir>/src/$1',
    // Mock native modules that might not work on Windows CI
    'node-gyp': '<rootDir>/__mocks__/node-gyp.js'
  },
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: 'tsconfig.json',
      isolatedModules: true
    }]
  },
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
    '/__tests__/',
    '/src-tauri/'
  ],
  // Windows-specific test setup
  setupFilesAfterEnv: ['<rootDir>/jest.setup.windows.js'],
  // Fail tests on console errors (Windows tends to be more verbose)
  silent: false,
  verbose: true,
  // Windows file watching configuration
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname'
  ]
};