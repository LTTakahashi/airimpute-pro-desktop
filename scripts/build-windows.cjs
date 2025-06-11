#!/usr/bin/env node

/**
 * Windows build script that ensures production TypeScript config is used
 * This script works around issues where the TypeScript config flag might be ignored
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('Starting Windows build with production TypeScript configuration...\n');

// Function to strip JSON comments
function stripJsonComments(jsonString) {
  // Remove single line comments
  jsonString = jsonString.replace(/\/\/.*$/gm, '');
  // Remove multi-line comments
  jsonString = jsonString.replace(/\/\*[\s\S]*?\*\//g, '');
  // Remove trailing commas
  jsonString = jsonString.replace(/,\s*([}\]])/g, '$1');
  return jsonString;
}

// Change to the project directory
const projectDir = path.join(__dirname, '..');
process.chdir(projectDir);

// Backup current tsconfig.json
console.log('Backing up tsconfig.json...');
const tsconfigPath = path.join(projectDir, 'tsconfig.json');
const tsconfigBackupPath = path.join(projectDir, 'tsconfig.json.backup');
const tsconfigProdPath = path.join(projectDir, 'tsconfig.production.json');

try {
  // Read the production config
  const prodConfigRaw = fs.readFileSync(tsconfigProdPath, 'utf8');
  const baseConfigRaw = fs.readFileSync(tsconfigPath, 'utf8');
  
  // Parse configs after stripping comments
  const prodConfig = JSON.parse(stripJsonComments(prodConfigRaw));
  const baseConfig = JSON.parse(stripJsonComments(baseConfigRaw));
  
  // Merge production settings into base config
  const mergedConfig = {
    ...baseConfig,
    compilerOptions: {
      ...baseConfig.compilerOptions,
      ...prodConfig.compilerOptions
    }
  };
  
  // Backup original
  fs.copyFileSync(tsconfigPath, tsconfigBackupPath);
  
  // Write merged config
  fs.writeFileSync(tsconfigPath, JSON.stringify(mergedConfig, null, 2));
  console.log('Applied production TypeScript configuration');
  
  // Run the build
  console.log('\nRunning TypeScript compilation...');
  execSync('tsc', { stdio: 'inherit' });
  
  console.log('\nRunning Vite build...');
  execSync('vite build', { stdio: 'inherit' });
  
  console.log('\n✓ Build completed successfully!');
  
} catch (error) {
  console.error('\n✗ Build failed:', error.message);
  if (error.stack) {
    console.error('Stack trace:', error.stack);
  }
  process.exit(1);
} finally {
  // Restore original tsconfig.json
  if (fs.existsSync(tsconfigBackupPath)) {
    fs.copyFileSync(tsconfigBackupPath, tsconfigPath);
    fs.unlinkSync(tsconfigBackupPath);
    console.log('\nRestored original tsconfig.json');
  }
}