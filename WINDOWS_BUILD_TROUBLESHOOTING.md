# Windows Build Troubleshooting Guide

## Quick Diagnostics

If the Windows build is failing in GitHub Actions:

1. **Check Recent Changes**
   ```bash
   # Look for changes to dependencies or build configuration
   git log --oneline -n 10 -- package.json package-lock.json .github/workflows/
   ```

2. **Common Issues and Solutions**

### Issue: npm install fails with Python errors

**Symptoms:**
- `gyp ERR! find Python`
- `gyp ERR! find VS`

**Solution:**
The workflow now tries multiple Python commands (`python`, `python3`, `py`). The scripts have been updated to be non-blocking.

### Issue: Visual Studio Build Tools not found

**Symptoms:**
- `MSBUILD : error MSB1009`
- `cannot find vcvarsall.bat`

**Solution:**
The workflow installs Visual Studio Build Tools via Chocolatey. Ensure the installation completes:
```yaml
- name: Install Visual Studio Build Tools
  shell: pwsh
  run: |
    choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive" -y --no-progress
```

### Issue: Tauri CLI not found

**Symptoms:**
- `tauri: command not found`
- `npm ERR! could not determine executable to run`

**Solution:**
The workflow now installs Tauri CLI locally as a dev dependency instead of globally.

### Issue: Path too long errors

**Symptoms:**
- `The specified path, file name, or both are too long`
- Error code `0x80010135`

**Solution:**
The workflow configures git for long paths:
```yaml
git config --global core.longpaths true
```

### Issue: Line ending problems

**Symptoms:**
- `warning: LF will be replaced by CRLF`
- Build works locally but fails in CI

**Solution:**
The workflow disables auto CRLF conversion:
```yaml
git config --global core.autocrlf false
```

## Testing Locally

To replicate the CI environment locally on Windows:

1. **Run the test script:**
   ```powershell
   cd airimpute-pro-desktop
   .\scripts\test-windows-build-local.ps1
   ```

2. **Manual steps:**
   ```powershell
   # Clean everything
   Remove-Item -Recurse -Force node_modules, package-lock.json, dist, src-tauri/target
   
   # Configure npm
   npm config set msvs_version 2022
   npm config set python python3
   
   # Install and build
   npm install --no-audit --no-fund --loglevel verbose
   npm run build
   npm run tauri build -- --target x86_64-pc-windows-msvc
   ```

## Debugging GitHub Actions

1. **Enable debug logging:**
   - Set repository secrets:
     - `ACTIONS_RUNNER_DEBUG`: `true`
     - `ACTIONS_STEP_DEBUG`: `true`

2. **Add diagnostic steps:**
   ```yaml
   - name: Diagnostic Info
     shell: pwsh
     run: |
       Write-Host "Node version: $(node --version)"
       Write-Host "NPM version: $(npm --version)"
       Write-Host "Python: $((Get-Command python -ErrorAction SilentlyContinue).Path)"
       Write-Host "Working directory: $(Get-Location)"
       npm config list
   ```

3. **Check artifacts on failure:**
   The workflow uploads logs even on failure:
   ```yaml
   - name: Upload build logs on failure
     if: failure()
     uses: actions/upload-artifact@v4
     with:
       name: windows-build-logs-${{ matrix.node-version }}
       path: |
         airimpute-pro-desktop/npm-install.log
         airimpute-pro-desktop/npm-debug.log
   ```

## Performance Optimization

1. **Use cache effectively:**
   - The workflow caches Rust dependencies via `Swatinem/rust-cache@v2`
   - Node modules are cached via `setup-node` cache option

2. **Parallel builds:**
   - Matrix builds for different Node versions run in parallel
   - Use `fail-fast: false` to see all failures

3. **Reduce build time:**
   - Use `npm ci` instead of `npm install` when possible
   - Limit test parallelization with `--maxWorkers=2`

## Emergency Fixes

If the build is completely broken and blocking development:

1. **Temporary skip (NOT RECOMMENDED):**
   ```yaml
   continue-on-error: true  # Add to failing step
   ```

2. **Revert to last working state:**
   ```bash
   # Find last successful build
   git log --grep="build" --oneline
   git checkout <last-working-commit> -- .github/workflows/
   ```

3. **Minimal build test:**
   Create a minimal workflow to isolate the issue:
   ```yaml
   name: Minimal Windows Test
   on: workflow_dispatch
   jobs:
     test:
       runs-on: windows-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-node@v4
           with:
             node-version: '18'
         - run: npm --version
         - run: npm install
         - run: npm run build
   ```

## Getting Help

1. Check GitHub Actions status: https://www.githubstatus.com/
2. Review Tauri Windows requirements: https://tauri.app/v1/guides/getting-started/prerequisites#windows
3. Search for similar issues: https://github.com/tauri-apps/tauri/issues

## Prevention

1. **Always test Windows-specific changes locally** using the provided PowerShell script
2. **Review the checklist** before pushing:
   - [ ] No Unix-specific commands or paths
   - [ ] All scripts use `shell: pwsh` on Windows
   - [ ] Dependencies are Windows-compatible
   - [ ] Path operations use `path.join()`
   - [ ] No hardcoded forward slashes in paths

3. **Monitor build times** - increasing build times often indicate cache issues