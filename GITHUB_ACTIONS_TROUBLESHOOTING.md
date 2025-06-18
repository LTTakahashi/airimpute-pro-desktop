# GitHub Actions Troubleshooting Guide

## Why Your Workflow Might Not Be Appearing

### 1. Check Repository Settings

Go to your repository on GitHub:
1. Click **Settings** → **Actions** → **General**
2. Ensure **Actions permissions** is set to one of:
   - "Allow all actions and reusable workflows" (recommended for testing)
   - "Allow {owner} actions and reusable workflows"
   
If it says "Disable actions", that's your problem!

### 2. Check Default Branch

The workflow MUST be on your default branch (usually `main` or `master`) to appear with `workflow_dispatch`.

Verify:
```bash
git branch -r --contains HEAD | grep origin
```

### 3. Manual Run Button

The "Run workflow" button only appears if:
1. The workflow has `workflow_dispatch:` in the `on:` section
2. The workflow file is on the default branch
3. You have write permissions to the repository

### 4. Automatic Triggers

If the workflow has `push:` triggers, it will run automatically when you push to the specified branches. You won't see a "Run" button for automatic workflows until they've run at least once.

### 5. Workflow File Validation

GitHub silently ignores invalid workflow files. Common issues:
- Using tabs instead of spaces (YAML requires spaces)
- Missing `jobs:` section
- Invalid job configuration

### 6. Check Actions Tab

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. Look at the left sidebar - your workflows should be listed there
4. If you see "Get started with GitHub Actions", Actions might be disabled

### 7. Organization Restrictions

If your repository is in an organization, the org admin might have:
- Disabled Actions entirely
- Restricted which actions can be used
- Required approval for first-time contributors

### 8. Caching/Indexing Delay

Sometimes there's a delay (1-5 minutes) between pushing a workflow and it appearing. Try:
- Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
- Clear browser cache
- Use incognito/private mode

## Testing Your Setup

We've added a simple test workflow (`.github/workflows/test-actions.yml`) that should:
1. Run automatically on push to main/master
2. Be manually runnable via the Actions tab

If this test workflow doesn't appear or run, then GitHub Actions is likely disabled for your repository.

## Current Workflows

1. **Build Windows** - Main workflow for building the Tauri app with embedded Python
2. **Test GitHub Actions** - Simple test to verify Actions are enabled

Both should appear in the Actions tab sidebar once properly configured.

## Security Notes

Our workflows include security hardening:
- Restricted permissions by default (`contents: read`)
- Pinned action versions to specific commits
- Only granted necessary permissions for specific jobs

## Next Steps

1. Check the Actions tab - do you see any workflows listed?
2. If not, check Settings → Actions → General
3. If Actions are enabled but workflows don't appear, wait 5 minutes and refresh
4. Try running the test workflow manually if it appears

If none of this works, the issue might be at the GitHub organization level or with your account permissions.