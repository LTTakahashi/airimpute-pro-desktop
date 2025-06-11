#!/usr/bin/env node

/**
 * Cross-platform post-install script for non-Windows systems
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const os = require('os');
const fs = require('fs');
const path = require('path');

const execAsync = promisify(exec);
const platform = os.platform();

async function postInstall() {
  console.log(`Running post-install setup for ${platform}...\n`);

  // Create required directories
  console.log('Creating project directories...');
  const dirs = [
    'dist',
    'build',
    'temp',
    path.join('src-tauri', 'icons')
  ];

  for (const dir of dirs) {
    const fullPath = path.join(process.cwd(), dir);
    if (!fs.existsSync(fullPath)) {
      try {
        fs.mkdirSync(fullPath, { recursive: true });
        console.log(`✓ Created ${dir}`);
      } catch (error) {
        console.error(`✗ Failed to create ${dir}:`, error.message);
      }
    }
  }

  // Platform-specific setup
  if (platform === 'darwin') {
    console.log('\nRunning macOS-specific setup...');
    // macOS specific tasks
  } else if (platform === 'linux') {
    console.log('\nRunning Linux-specific setup...');
    // Check for required system libraries
    try {
      await execAsync('which pkg-config');
      console.log('✓ pkg-config found');
    } catch (error) {
      console.warn('⚠ pkg-config not found. Install with: sudo apt-get install pkg-config');
    }
  }

  // Check Rust installation
  console.log('\nChecking Rust installation...');
  try {
    const { stdout } = await execAsync('rustc --version');
    console.log(`✓ Rust found: ${stdout.trim()}`);
  } catch (error) {
    console.error('✗ Rust not found. Install from: https://rustup.rs/');
  }

  // Rebuild native dependencies if needed
  console.log('\nChecking native dependencies...');
  const packageJson = require('../package.json');
  const hasNativeDeps = checkForNativeDependencies(packageJson);
  
  if (hasNativeDeps) {
    console.log('ℹ Native dependencies detected. Rebuilding...');
    try {
      await execAsync('npm rebuild');
      console.log('✓ Native dependencies rebuilt');
    } catch (error) {
      console.error('✗ Failed to rebuild native dependencies:', error.message);
    }
  }

  console.log('\n✓ Post-install complete!');
}

function checkForNativeDependencies(packageJson) {
  const allDeps = {
    ...packageJson.dependencies,
    ...packageJson.devDependencies,
    ...packageJson.optionalDependencies
  };

  const nativeDepPatterns = [
    'node-gyp',
    'sodium',
    'bcrypt',
    'sharp',
    'canvas',
    'sqlite3',
    'serialport',
    'usb',
    'bluetooth'
  ];

  return Object.keys(allDeps).some(dep => 
    nativeDepPatterns.some(pattern => dep.includes(pattern))
  );
}

postInstall().catch(error => {
  console.error('Post-install failed:', error);
  // Don't exit with error to not break npm install
  process.exit(0);
});