/**
 * Windows-specific post-install script
 * Handles Windows-specific setup after npm install
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const os = require('os');
const fs = require('fs');
const path = require('path');

const execAsync = promisify(exec);
const isWindows = os.platform() === 'win32';

async function windowsPostInstall() {
  if (!isWindows) {
    console.log('Not running on Windows, skipping Windows post-install.');
    return;
  }

  console.log('Running Windows post-install setup...\n');

  // Set npm configuration for Windows
  console.log('Configuring npm for Windows build tools...');
  try {
    await execAsync('npm config set msvs_version 2022');
    await execAsync('npm config set python python3');
    console.log('✓ NPM configured for Windows');
  } catch (error) {
    console.warn('⚠ Could not configure npm:', error.message);
  }

  // Check and fix long path support
  console.log('\nChecking Windows long path support...');
  try {
    // This would require admin privileges in real scenario
    console.log('ℹ To enable long path support, run as administrator:');
    console.log('  reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1');
  } catch (error) {
    console.warn('⚠ Could not check long path support');
  }

  // Handle native dependencies
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

  // Create required directories with Windows-safe paths
  console.log('\nCreating project directories...');
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

  // Set up git configuration for Windows
  console.log('\nConfiguring git for Windows...');
  try {
    await execAsync('git config core.autocrlf false');
    await execAsync('git config core.longpaths true');
    console.log('✓ Git configured for Windows');
  } catch (error) {
    console.warn('⚠ Could not configure git:', error.message);
  }

  // Check Tauri prerequisites
  console.log('\nChecking Tauri prerequisites...');
  try {
    // Use a more reliable method to check for WebView2
    const { stdout } = await execAsync('powershell -Command "Get-ItemProperty -Path \'HKLM:\\SOFTWARE\\WOW6432Node\\Microsoft\\EdgeUpdate\\Clients\\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}\' -Name pv -ErrorAction SilentlyContinue | Select-Object -ExpandProperty pv"').catch(() => ({ stdout: '' }));
    
    if (stdout.trim()) {
      console.log(`✓ WebView2 Runtime found: ${stdout.trim()}`);
    } else {
      console.log('⚠ WebView2 Runtime might not be installed');
      console.log('  Note: This check may fail in CI environments');
      console.log('  Download from: https://developer.microsoft.com/en-us/microsoft-edge/webview2/');
    }
  } catch (error) {
    console.warn('⚠ Could not check WebView2 Runtime (this is normal in CI)');
  }

  console.log('\n✓ Windows post-install complete!');
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

windowsPostInstall().catch(error => {
  console.error('Windows post-install failed:', error);
  // Don't exit with error to not break npm install
  process.exit(0);
});