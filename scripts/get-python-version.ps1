# PowerShell script to read Python version configuration
param(
    [string]$Property = "version"
)

$configPath = Join-Path $PSScriptRoot ".." "python-version.json"

if (-not (Test-Path $configPath)) {
    Write-Error "Python version configuration not found at: $configPath"
    exit 1
}

try {
    $config = Get-Content $configPath | ConvertFrom-Json
    
    switch ($Property) {
        "version" { Write-Output $config.version }
        "major" { Write-Output $config.major }
        "minor" { Write-Output $config.minor }
        "patch" { Write-Output $config.patch }
        "dll_name" { Write-Output $config.dll_name }
        "lib_name" { Write-Output $config.lib_name }
        "choco_package" { Write-Output $config.choco_package }
        "embeddable_url" { Write-Output $config.embeddable_url }
        default { Write-Output $config.version }
    }
} catch {
    Write-Error "Failed to read Python version configuration: $_"
    exit 1
}