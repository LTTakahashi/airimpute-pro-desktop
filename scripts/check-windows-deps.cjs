/**
 * Check Windows dependencies for building the application
 * This script verifies that all required Windows build tools are available
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const os = require('os');
const fs = require('fs');
const path = require('path');

const execAsync = promisify(exec);

const isWindows = os.platform() === 'win32';

async function checkWindowsDependencies() {
  if (!isWindows) {
    console.log('Not running on Windows, skipping Windows dependency checks.');
    return;
  }

  console.log('Checking Windows build dependencies...\n');

  const checks = {
    python: false,
    visualStudio: false,
    nodeGyp: false,
    rustTarget: false
  };

  // Check Python - try multiple commands
  const pythonCommands = ['python', 'python3', 'py'];
  for (const cmd of pythonCommands) {
    try {
      const { stdout } = await execAsync(`${cmd} --version`);
      const version = stdout.trim();
      console.log(`✓ Python found (${cmd}): ${version}`);
      checks.python = true;
      break;
    } catch (error) {
      // Continue to next command
    }
  }
  
  if (!checks.python) {
    console.warn('⚠ Python not found. Please install Python 3.x');
    console.warn('  Download from: https://www.python.org/downloads/');
  }

  // Check for Visual Studio Build Tools
  try {
    // Check if MSBuild is available
    await execAsync('where msbuild');
    console.log('✓ MSBuild found');
    checks.visualStudio = true;
  } catch (error) {
    console.error('✗ Visual Studio Build Tools not found');
    console.error('  Install with: choco install visualstudio2022buildtools');
    console.error('  Or download from: https://visualstudio.microsoft.com/downloads/');
  }

  // Check node-gyp
  try {
    const { stdout } = await execAsync('npm list -g node-gyp');
    if (stdout.includes('node-gyp')) {
      console.log('✓ node-gyp found globally');
      checks.nodeGyp = true;
    }
  } catch (error) {
    console.warn('⚠ node-gyp not found globally (will be installed locally if needed)');
    checks.nodeGyp = true; // Not critical if missing globally
  }

  // Check Rust Windows target
  try {
    const { stdout } = await execAsync('rustup target list --installed');
    if (stdout.includes('x86_64-pc-windows-msvc')) {
      console.log('✓ Rust Windows target installed: x86_64-pc-windows-msvc');
      checks.rustTarget = true;
    } else {
      console.warn('⚠ Rust Windows target not installed');
      console.log('  Install with: rustup target add x86_64-pc-windows-msvc');
    }
  } catch (error) {
    console.error('✗ Rustup not found. Please install Rust from https://rustup.rs/');
  }

  // Check environment variables (npm 9+ uses these instead of config)
  console.log('\nEnvironment Configuration:');
  console.log(`  npm_config_msvs_version: ${process.env.npm_config_msvs_version || 'not set'}`);
  console.log(`  npm_config_python: ${process.env.npm_config_python || 'not set'}`);
  console.log(`  GYP_MSVS_VERSION: ${process.env.GYP_MSVS_VERSION || 'not set'}`);
  console.log(`  PYTHON: ${process.env.PYTHON || 'not set'}`);
  
  // Check npm version to provide appropriate guidance
  try {
    const { stdout: npmVersion } = await execAsync('npm --version');
    const majorVersion = parseInt(npmVersion.split('.')[0]);
    if (majorVersion >= 9) {
      console.log(`\n  ℹ npm ${npmVersion.trim()} detected - use environment variables instead of npm config`);
    }
  } catch (error) {
    console.warn('  ⚠ Could not determine npm version');
  }

  // Summary
  console.log('\n=== Dependency Check Summary ===');
  const allChecksPassed = Object.values(checks).every(check => check);
  
  if (allChecksPassed) {
    console.log('✓ All Windows dependencies are satisfied');
  } else {
    console.warn('⚠ Some dependencies are missing (non-blocking)');
    console.log('\nTo fix missing dependencies:');
    console.log('1. Install Python 3.x from https://www.python.org/');
    console.log('2. Install Visual Studio Build Tools:');
    console.log('   choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"');
    console.log('3. Configure build environment:');
    console.log('   For npm 9+, set environment variables:');
    console.log('   export npm_config_msvs_version=2022');
    console.log('   export npm_config_python=python3');
    console.log('   Or use install flags: npm install --msvs_version=2022');
    console.log('4. Add Rust Windows target:');
    console.log('   rustup target add x86_64-pc-windows-msvc');
  }
  
  // Always exit with 0 to avoid blocking npm install
  process.exit(0);
}

checkWindowsDependencies().catch(error => {
  console.error('Error checking dependencies:', error);
  // Don't fail npm install even on error
  process.exit(0);
});