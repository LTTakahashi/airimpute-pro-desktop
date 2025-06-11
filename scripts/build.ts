/**
 * Cross-platform build script
 * Following CLAUDE.md specifications for platform-agnostic builds
 */

import { spawn } from 'child_process';
import { PlatformUtils } from '../src/utils/platform';
import * as fs from 'fs/promises';
import * as path from 'path';

interface BuildOptions {
  target?: string;
  release?: boolean;
  verbose?: boolean;
}

/**
 * @description Execute the build process with platform-specific handling
 * @param {BuildOptions} options - Build configuration options
 * @returns {Promise<void>}
 * @throws {Error} If build fails
 * @complexity O(n) where n is the number of build steps
 * @space O(1)
 */
async function build(options: BuildOptions = {}): Promise<void> {
  const { target, release = true, verbose = false } = options;
  
  console.log('🚀 Starting build process...');
  console.log(`Platform: ${PlatformUtils.getPlatformInfo().platform}`);
  console.log(`Architecture: ${PlatformUtils.getPlatformInfo().arch}`);
  console.log(`Node version: ${PlatformUtils.getPlatformInfo().nodeVersion}`);
  
  try {
    // Step 1: Clean previous builds
    console.log('\n📦 Cleaning previous builds...');
    await cleanBuildArtifacts();
    
    // Step 2: Install dependencies
    console.log('\n📦 Installing dependencies...');
    await runCommand('npm', ['ci', '--no-audit']);
    
    // Step 3: Type checking
    console.log('\n🔍 Running type checks...');
    await runCommand('npm', ['run', 'type-check']);
    
    // Step 4: Build frontend
    console.log('\n🏗️ Building frontend...');
    await runCommand('npm', ['run', 'build:frontend']);
    
    // Step 5: Build Tauri application
    console.log('\n🦀 Building Tauri application...');
    const tauriArgs = ['run', 'tauri', 'build'];
    
    if (target) {
      tauriArgs.push('--', '--target', target);
    }
    
    if (!release) {
      tauriArgs.push('--debug');
    }
    
    if (verbose) {
      tauriArgs.push('--verbose');
    }
    
    await runCommand('npm', tauriArgs, {
      env: {
        ...process.env,
        NODE_ENV: 'production',
        FORCE_COLOR: '1'
      }
    });
    
    // Step 6: Verify build artifacts
    console.log('\n✅ Verifying build artifacts...');
    await verifyBuildArtifacts();
    
    console.log('\n🎉 Build completed successfully!');
  } catch (error) {
    console.error('\n❌ Build failed:', error);
    process.exit(1);
  }
}

/**
 * @description Run a command with platform-specific handling
 * @param {string} command - Command to execute
 * @param {string[]} args - Command arguments
 * @param {object} options - Additional options
 * @returns {Promise<void>}
 * @complexity O(1)
 * @space O(1)
 */
function runCommand(
  command: string,
  args: string[] = [],
  options: any = {}
): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = PlatformUtils.spawnProcess(command, args, {
      stdio: 'inherit',
      ...options
    });
    
    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed with code ${code}: ${command} ${args.join(' ')}`));
      }
    });
    
    child.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * @description Clean build artifacts from previous builds
 * @returns {Promise<void>}
 * @complexity O(n) where n is number of files/directories
 * @space O(1)
 */
async function cleanBuildArtifacts(): Promise<void> {
  const artifactPaths = [
    'dist',
    'build',
    PlatformUtils.joinPath('src-tauri', 'target', 'release', 'bundle')
  ];
  
  for (const artifactPath of artifactPaths) {
    try {
      await fs.rm(artifactPath, { recursive: true, force: true });
      console.log(`  ✓ Cleaned ${artifactPath}`);
    } catch (error) {
      // Ignore errors for non-existent directories
    }
  }
}

/**
 * @description Verify that build artifacts were created successfully
 * @returns {Promise<void>}
 * @throws {Error} If expected artifacts are missing
 * @complexity O(n) where n is number of artifacts to check
 * @space O(1)
 */
async function verifyBuildArtifacts(): Promise<void> {
  const requiredArtifacts = [
    'dist',
    PlatformUtils.joinPath('src-tauri', 'target', 'release')
  ];
  
  for (const artifact of requiredArtifacts) {
    try {
      const stats = await fs.stat(artifact);
      if (!stats.isDirectory()) {
        throw new Error(`Expected directory not found: ${artifact}`);
      }
      console.log(`  ✓ Verified ${artifact}`);
    } catch (error) {
      throw new Error(`Missing build artifact: ${artifact}`);
    }
  }
  
  // Platform-specific artifact verification
  if (PlatformUtils.IS_WINDOWS) {
    const windowsArtifacts = [
      PlatformUtils.joinPath('src-tauri', 'target', 'release', 'bundle', 'msi'),
      PlatformUtils.joinPath('src-tauri', 'target', 'release', 'bundle', 'nsis')
    ];
    
    for (const artifact of windowsArtifacts) {
      try {
        const files = await fs.readdir(artifact);
        const installers = files.filter(f => f.endsWith('.msi') || f.endsWith('.exe'));
        if (installers.length === 0) {
          throw new Error(`No installer files found in ${artifact}`);
        }
        console.log(`  ✓ Found ${installers.length} installer(s) in ${artifact}`);
      } catch (error) {
        console.warn(`  ⚠ Warning: ${error.message}`);
      }
    }
  }
}

// Parse command line arguments
const args = process.argv.slice(2);
const options: BuildOptions = {
  target: args.find(arg => arg.startsWith('--target='))?.split('=')[1],
  release: !args.includes('--debug'),
  verbose: args.includes('--verbose')
};

// Run build
build(options);