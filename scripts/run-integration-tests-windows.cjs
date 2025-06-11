#!/usr/bin/env node

/**
 * Windows-specific integration test runner
 * Ensures proper environment setup for Windows CI/CD
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Detect if running in CI
const isCI = process.env.CI === 'true' || process.env.GITHUB_ACTIONS === 'true';

// Windows-specific environment variables
const windowsEnv = {
  ...process.env,
  NODE_ENV: 'test',
  FORCE_COLOR: '1',
  // Increase memory for Windows
  NODE_OPTIONS: '--max-old-space-size=4096',
  // Disable Windows Defender scanning for test files (CI only)
  ...(isCI && { WINDOWS_DEFENDER_EXCLUSION: 'true' }),
};

// Check for required Windows dependencies
function checkWindowsDependencies() {
  console.log('🔍 Checking Windows dependencies...');
  
  const checks = [
    {
      name: 'Node.js version',
      check: () => {
        const version = process.version;
        const major = parseInt(version.split('.')[0].substring(1));
        return major >= 18;
      },
      error: 'Node.js 18 or higher is required',
    },
    {
      name: 'Visual C++ Redistributable',
      check: () => {
        // Check for common VC++ runtime DLLs
        try {
          const system32 = path.join(process.env.WINDIR || 'C:\\Windows', 'System32');
          return fs.existsSync(path.join(system32, 'vcruntime140.dll'));
        } catch {
          return false;
        }
      },
      error: 'Visual C++ Redistributable 2015-2022 is required',
    },
    {
      name: 'Temp directory access',
      check: () => {
        try {
          const tempDir = process.env.TEMP || process.env.TMP || 'C:\\Temp';
          const testFile = path.join(tempDir, 'airimpute-test-' + Date.now());
          fs.writeFileSync(testFile, 'test');
          fs.unlinkSync(testFile);
          return true;
        } catch {
          return false;
        }
      },
      error: 'Cannot write to temp directory',
    },
  ];

  let allPassed = true;
  
  for (const { name, check, error } of checks) {
    try {
      if (check()) {
        console.log(`✅ ${name}`);
      } else {
        console.error(`❌ ${name}: ${error}`);
        allPassed = false;
      }
    } catch (e) {
      console.error(`❌ ${name}: ${e.message}`);
      allPassed = false;
    }
  }

  return allPassed;
}

// Clean up test artifacts
function cleanupTestArtifacts() {
  console.log('🧹 Cleaning up test artifacts...');
  
  const artifacts = [
    'coverage',
    'test-results',
    '.vitest',
    'node_modules/.cache',
  ];

  for (const artifact of artifacts) {
    try {
      if (fs.existsSync(artifact)) {
        fs.rmSync(artifact, { recursive: true, force: true });
        console.log(`  Removed ${artifact}`);
      }
    } catch (e) {
      console.warn(`  Warning: Could not remove ${artifact}: ${e.message}`);
    }
  }
}

// Main test runner
async function runTests() {
  console.log('🚀 Starting Windows integration tests...');
  console.log(`Platform: ${process.platform}`);
  console.log(`CI Environment: ${isCI ? 'Yes' : 'No'}`);
  console.log(`Node.js: ${process.version}`);
  console.log('');

  // Check dependencies
  if (!checkWindowsDependencies()) {
    console.error('\n❌ Dependency check failed. Please install missing dependencies.');
    process.exit(1);
  }

  // Clean up before tests
  cleanupTestArtifacts();

  // Prepare test command
  const vitestPath = path.join('node_modules', '.bin', 'vitest.cmd');
  const args = [
    'run',
    '--config', 'vitest.integration.config.ts',
    '--reporter', isCI ? 'junit' : 'verbose',
    '--reporter', 'json',
    '--outputFile', 'test-results/integration-results.json',
  ];

  // Add coverage in CI
  if (isCI) {
    args.push('--coverage');
  }

  // Add any additional arguments passed to this script
  args.push(...process.argv.slice(2));

  console.log('\n📝 Running command:');
  console.log(`  ${vitestPath} ${args.join(' ')}\n`);

  // Run tests
  const testProcess = spawn(vitestPath, args, {
    env: windowsEnv,
    stdio: 'inherit',
    shell: true,
  });

  // Handle process events
  testProcess.on('error', (error) => {
    console.error('❌ Failed to start test process:', error);
    process.exit(1);
  });

  testProcess.on('close', (code) => {
    console.log(`\n✅ Tests completed with exit code: ${code}`);
    
    // Generate report in CI
    if (isCI && code === 0) {
      generateCIReport();
    }
    
    process.exit(code);
  });
}

// Generate CI-friendly report
function generateCIReport() {
  console.log('\n📊 Generating CI report...');
  
  try {
    const resultsPath = 'test-results/integration-results.json';
    if (fs.existsSync(resultsPath)) {
      const results = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
      
      console.log('\nTest Summary:');
      console.log(`  Total Tests: ${results.numTotalTests || 0}`);
      console.log(`  Passed: ${results.numPassedTests || 0}`);
      console.log(`  Failed: ${results.numFailedTests || 0}`);
      console.log(`  Duration: ${results.duration || 0}ms`);
      
      // Create GitHub Actions summary if available
      if (process.env.GITHUB_STEP_SUMMARY) {
        const summary = `
## Integration Test Results

| Metric | Value |
|--------|-------|
| Total Tests | ${results.numTotalTests || 0} |
| Passed | ${results.numPassedTests || 0} |
| Failed | ${results.numFailedTests || 0} |
| Duration | ${results.duration || 0}ms |
| Platform | Windows |
`;
        fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, summary);
      }
    }
  } catch (e) {
    console.warn('Could not generate CI report:', e.message);
  }
}

// Error handler
process.on('unhandledRejection', (error) => {
  console.error('❌ Unhandled rejection:', error);
  process.exit(1);
});

// Run tests
runTests().catch((error) => {
  console.error('❌ Test runner error:', error);
  process.exit(1);
});