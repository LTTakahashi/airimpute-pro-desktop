#!/usr/bin/env node

/**
 * Build verification script for AirImpute Pro Desktop
 * Ensures the project is ready for Windows compilation
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, readFileSync } from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

const requiredFiles = [
  'package.json',
  'package-lock.json',
  'tsconfig.json',
  'src-tauri/tauri.conf.json',
  'src-tauri/Cargo.toml',
  'src-tauri/build.rs',
  'index.html',
  'vite.config.ts',
  '.github/workflows/windows-build.yml'
];

const requiredDirs = [
  'src',
  'src-tauri/src',
  'src-tauri/icons',
  '.github/actions/setup-build-env'
];

async function checkFile(path) {
  const fullPath = join(projectRoot, path);
  if (!existsSync(fullPath)) {
    console.error(`❌ Missing required file: ${path}`);
    return false;
  }
  console.log(`✅ Found: ${path}`);
  return true;
}

async function checkDir(path) {
  const fullPath = join(projectRoot, path);
  if (!existsSync(fullPath)) {
    console.error(`❌ Missing required directory: ${path}`);
    return false;
  }
  console.log(`✅ Found: ${path}`);
  return true;
}

async function checkDependencies() {
  console.log('\n🔍 Checking dependencies...');
  
  try {
    const packageJson = JSON.parse(readFileSync(join(projectRoot, 'package.json'), 'utf8'));
    
    const requiredDeps = [
      '@tauri-apps/api',
      'react',
      'react-dom'
    ];
    
    const requiredDevDeps = [
      '@tauri-apps/cli',
      'vite',
      '@vitejs/plugin-react',
      'typescript'
    ];
    
    let allFound = true;
    
    for (const dep of requiredDeps) {
      if (!packageJson.dependencies || !packageJson.dependencies[dep]) {
        console.error(`❌ Missing dependency: ${dep}`);
        allFound = false;
      } else {
        console.log(`✅ Dependency: ${dep}`);
      }
    }
    
    for (const dep of requiredDevDeps) {
      if (!packageJson.devDependencies || !packageJson.devDependencies[dep]) {
        console.error(`❌ Missing dev dependency: ${dep}`);
        allFound = false;
      } else {
        console.log(`✅ Dev dependency: ${dep}`);
      }
    }
    
    return allFound;
  } catch (error) {
    console.error('❌ Failed to check dependencies:', error.message);
    return false;
  }
}

async function checkCommands() {
  console.log('\n🔍 Checking build commands...');
  
  const commands = [
    { cmd: 'node --version', name: 'Node.js' },
    { cmd: 'npm --version', name: 'npm' },
    { cmd: 'rustc --version', name: 'Rust', optional: true },
    { cmd: 'cargo --version', name: 'Cargo', optional: true }
  ];
  
  let allFound = true;
  
  for (const { cmd, name, optional } of commands) {
    try {
      const { stdout } = await execAsync(cmd);
      console.log(`✅ ${name}: ${stdout.trim()}`);
    } catch (error) {
      if (optional) {
        console.warn(`⚠️  ${name} not found (optional for CI)`);
      } else {
        console.error(`❌ ${name} not found`);
        allFound = false;
      }
    }
  }
  
  return allFound;
}

async function verifyBuild() {
  console.log('🚀 AirImpute Pro Desktop - Build Verification\n');
  
  let success = true;
  
  // Check required files
  console.log('📁 Checking required files...');
  for (const file of requiredFiles) {
    if (!await checkFile(file)) {
      success = false;
    }
  }
  
  // Check required directories
  console.log('\n📁 Checking required directories...');
  for (const dir of requiredDirs) {
    if (!await checkDir(dir)) {
      success = false;
    }
  }
  
  // Check dependencies
  if (!await checkDependencies()) {
    success = false;
  }
  
  // Check commands
  if (!await checkCommands()) {
    success = false;
  }
  
  // Summary
  console.log('\n' + '='.repeat(50));
  if (success) {
    console.log('✅ Build verification PASSED!');
    console.log('The project is ready for Windows compilation.');
    console.log('\nNext steps:');
    console.log('1. Run: npm run build');
    console.log('2. Run: npm run tauri build');
    console.log('3. Check src-tauri/target/release/bundle/ for installers');
  } else {
    console.log('❌ Build verification FAILED!');
    console.log('Please fix the issues above before proceeding.');
    process.exit(1);
  }
}

// Run verification
verifyBuild().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});