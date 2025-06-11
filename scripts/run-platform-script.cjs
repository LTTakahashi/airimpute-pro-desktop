#!/usr/bin/env node

/**
 * Platform-specific script runner
 * Replaces run-script-os functionality with a simple Node.js script
 */

const os = require('os');
const { spawn } = require('child_process');

const platform = os.platform();
const isWindows = platform === 'win32';

// Get the script name from command line arguments
const scriptName = process.argv[2];

if (!scriptName) {
  console.error('Error: No script name provided');
  console.error('Usage: node run-platform-script.cjs <script-name>');
  process.exit(1);
}

// Determine the platform-specific script name
const platformScript = isWindows 
  ? `${scriptName}:windows`
  : `${scriptName}:default`;

console.log(`Running platform script: npm run ${platformScript}`);

// Run the platform-specific script
const npm = isWindows ? 'npm.cmd' : 'npm';
const child = spawn(npm, ['run', platformScript], {
  stdio: 'inherit',
  shell: isWindows,
  env: process.env
});

child.on('error', (error) => {
  console.error(`Failed to run ${platformScript}:`, error.message);
  process.exit(1);
});

child.on('exit', (code) => {
  process.exit(code || 0);
});