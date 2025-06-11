#!/usr/bin/env node

/**
 * Unified post-install script that handles platform detection
 * Replaces the need for run-script-os
 */

const os = require('os');
const { spawn } = require('child_process');
const path = require('path');

const platform = os.platform();
const isWindows = platform === 'win32';

console.log(`Running post-install for platform: ${platform}`);

// Determine which script to run
const scriptName = isWindows ? 'windows-post-install.cjs' : 'post-install.cjs';
const scriptPath = path.join(__dirname, scriptName);

// Run the appropriate script
const child = spawn('node', [scriptPath], {
  stdio: 'inherit',
  shell: isWindows,
  env: process.env
});

child.on('error', (error) => {
  console.error(`Failed to run ${scriptName}:`, error.message);
  // Don't fail the install process
  process.exit(0);
});

child.on('exit', (code) => {
  if (code !== 0) {
    console.warn(`${scriptName} exited with code ${code}`);
  }
  // Always exit with 0 to not break npm install
  process.exit(0);
});