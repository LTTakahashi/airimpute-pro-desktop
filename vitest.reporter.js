/**
 * Custom reporter for GitHub Actions
 * Provides better output formatting and error reporting in CI
 */

import { DefaultReporter } from 'vitest/reporters';

export default class GitHubActionsReporter extends DefaultReporter {
  constructor(options) {
    super(options);
    this.isGitHubActions = process.env.GITHUB_ACTIONS === 'true';
  }

  onTaskUpdate(packs) {
    super.onTaskUpdate(packs);
    
    if (!this.isGitHubActions) return;

    // Output GitHub Actions annotations for failures
    for (const pack of packs) {
      const taskResult = pack[1];
      if (taskResult?.state === 'fail') {
        for (const test of taskResult.tests || []) {
          if (test.state === 'fail' && test.error) {
            this.reportGitHubError(test);
          }
        }
      }
    }
  }

  reportGitHubError(test) {
    const error = test.error;
    const file = test.file?.filepath || 'unknown';
    const line = error.stack ? this.extractLineNumber(error.stack) : 1;
    
    // GitHub Actions error annotation format
    console.log(`::error file=${file},line=${line}::${test.name} - ${error.message}`);
    
    // Add additional context
    if (error.stack) {
      console.log(`::group::Stack trace for: ${test.name}`);
      console.log(error.stack);
      console.log('::endgroup::');
    }
  }

  extractLineNumber(stack) {
    const match = stack.match(/:(\d+):\d+/);
    return match ? parseInt(match[1], 10) : 1;
  }

  onFinished(files, errors) {
    super.onFinished(files, errors);
    
    if (!this.isGitHubActions) return;

    // Add summary to GitHub Actions
    const summary = this.generateSummary(files);
    console.log('\n::group::Test Summary');
    console.log(summary);
    console.log('::endgroup::');

    // Set output variables for workflow
    if (errors.length > 0) {
      console.log(`::set-output name=test-failed::true`);
      console.log(`::set-output name=error-count::${errors.length}`);
    } else {
      console.log(`::set-output name=test-failed::false`);
      console.log(`::set-output name=error-count::0`);
    }
  }

  generateSummary(files) {
    let totalTests = 0;
    let passedTests = 0;
    let failedTests = 0;
    let skippedTests = 0;
    let totalTime = 0;

    for (const file of files) {
      const tasks = file.tasks || [];
      for (const task of tasks) {
        totalTests += task.result?.tests?.length || 0;
        
        for (const test of task.result?.tests || []) {
          if (test.state === 'pass') passedTests++;
          else if (test.state === 'fail') failedTests++;
          else if (test.state === 'skip') skippedTests++;
        }
        
        totalTime += task.result?.duration || 0;
      }
    }

    return `
Test Results:
- Total: ${totalTests}
- Passed: ${passedTests} ✅
- Failed: ${failedTests} ❌
- Skipped: ${skippedTests} ⏭️
- Duration: ${(totalTime / 1000).toFixed(2)}s
- Platform: ${process.platform}
`;
  }
}