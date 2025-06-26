# Certificate Management Script for AirImpute Pro
# This script provides utilities for managing code signing certificates

param(
    [Parameter(Position=0)]
    [ValidateSet('list', 'install', 'test', 'export', 'validate', 'monitor')]
    [string]$Action = 'list',
    
    [string]$CertPath,
    [string]$Password,
    [string]$Thumbprint,
    [string]$OutputPath,
    [switch]$Verbose
)

$ErrorActionPreference = 'Stop'

function Write-ColorOutput($message, $color = 'White') {
    Write-Host $message -ForegroundColor $color
}

function Get-CodeSigningCertificates {
    Write-ColorOutput "`n=== Code Signing Certificates ===" "Cyan"
    
    $certs = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert
    
    if ($certs.Count -eq 0) {
        Write-ColorOutput "No code signing certificates found in personal store" "Yellow"
        return
    }
    
    $certs | ForEach-Object {
        Write-ColorOutput "`nThumbprint: $($_.Thumbprint)" "Green"
        Write-Host "Subject: $($_.Subject)"
        Write-Host "Issuer: $($_.Issuer)"
        Write-Host "Valid From: $($_.NotBefore)"
        Write-Host "Valid To: $($_.NotAfter)"
        
        $daysRemaining = ($_.NotAfter - (Get-Date)).Days
        if ($daysRemaining -lt 30) {
            Write-ColorOutput "WARNING: Certificate expires in $daysRemaining days!" "Red"
        } elseif ($daysRemaining -lt 90) {
            Write-ColorOutput "Certificate expires in $daysRemaining days" "Yellow"
        } else {
            Write-ColorOutput "Certificate expires in $daysRemaining days" "Green"
        }
    }
}

function Install-CodeSigningCertificate {
    if (-not $CertPath) {
        Write-ColorOutput "Error: -CertPath is required for install action" "Red"
        exit 1
    }
    
    if (-not (Test-Path $CertPath)) {
        Write-ColorOutput "Error: Certificate file not found: $CertPath" "Red"
        exit 1
    }
    
    Write-ColorOutput "`n=== Installing Certificate ===" "Cyan"
    
    try {
        if ($Password) {
            $securePwd = ConvertTo-SecureString -String $Password -AsPlainText -Force
            $cert = Import-PfxCertificate -FilePath $CertPath -CertStoreLocation Cert:\CurrentUser\My -Password $securePwd
        } else {
            # Prompt for password
            $securePwd = Read-Host "Enter certificate password" -AsSecureString
            $cert = Import-PfxCertificate -FilePath $CertPath -CertStoreLocation Cert:\CurrentUser\My -Password $securePwd
        }
        
        Write-ColorOutput "Certificate installed successfully!" "Green"
        Write-Host "Thumbprint: $($cert.Thumbprint)"
        Write-Host "Subject: $($cert.Subject)"
        
        # Update tauri.conf.json with thumbprint
        $configPath = Join-Path $PSScriptRoot "..\src-tauri\tauri.conf.json"
        if (Test-Path $configPath) {
            Write-ColorOutput "`nUpdating tauri.conf.json with thumbprint..." "Yellow"
            $config = Get-Content $configPath -Raw | ConvertFrom-Json
            $config.tauri.bundle.windows.certificateThumbprint = $cert.Thumbprint
            $config | ConvertTo-Json -Depth 10 | Set-Content $configPath
            Write-ColorOutput "Configuration updated!" "Green"
        }
        
    } catch {
        Write-ColorOutput "Error installing certificate: $_" "Red"
        exit 1
    }
}

function Test-CodeSigning {
    Write-ColorOutput "`n=== Testing Code Signing ===" "Cyan"
    
    # Create test executable
    $testExe = Join-Path $env:TEMP "test-signing.exe"
    
    # Use a simple executable for testing (copy powershell.exe)
    Copy-Item "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe" $testExe -Force
    
    try {
        if ($Thumbprint) {
            $cert = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Thumbprint -eq $Thumbprint }
        } else {
            $cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | Select-Object -First 1
        }
        
        if (-not $cert) {
            Write-ColorOutput "No certificate found for signing" "Red"
            return
        }
        
        Write-Host "Using certificate: $($cert.Subject)"
        
        # Sign the test file
        $result = Set-AuthenticodeSignature -FilePath $testExe -Certificate $cert -TimestampServer "http://timestamp.digicert.com"
        
        if ($result.Status -eq 'Valid') {
            Write-ColorOutput "Test signing successful!" "Green"
            
            # Verify signature
            $verification = Get-AuthenticodeSignature -FilePath $testExe
            Write-Host "`nSignature Details:"
            Write-Host "Status: $($verification.Status)"
            Write-Host "Signer: $($verification.SignerCertificate.Subject)"
            Write-Host "Timestamp: $($verification.TimeStamperCertificate.Subject)"
        } else {
            Write-ColorOutput "Test signing failed: $($result.Status)" "Red"
            Write-Host $result.StatusMessage
        }
        
    } finally {
        Remove-Item $testExe -Force -ErrorAction SilentlyContinue
    }
}

function Export-CertificateForCI {
    if (-not $Thumbprint) {
        Write-ColorOutput "Error: -Thumbprint is required for export action" "Red"
        exit 1
    }
    
    if (-not $OutputPath) {
        $OutputPath = Join-Path $PSScriptRoot "exported-cert.pfx"
    }
    
    Write-ColorOutput "`n=== Exporting Certificate for CI/CD ===" "Cyan"
    
    try {
        $cert = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Thumbprint -eq $Thumbprint }
        
        if (-not $cert) {
            Write-ColorOutput "Certificate not found with thumbprint: $Thumbprint" "Red"
            exit 1
        }
        
        Write-Host "Exporting certificate: $($cert.Subject)"
        
        # Prompt for export password
        $exportPwd = Read-Host "Enter password for exported certificate" -AsSecureString
        $exportPwdConfirm = Read-Host "Confirm password" -AsSecureString
        
        # Convert to plain text for comparison
        $pwd1 = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($exportPwd))
        $pwd2 = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($exportPwdConfirm))
        
        if ($pwd1 -ne $pwd2) {
            Write-ColorOutput "Passwords do not match!" "Red"
            exit 1
        }
        
        # Export certificate
        Export-PfxCertificate -Cert $cert -FilePath $OutputPath -Password $exportPwd
        
        Write-ColorOutput "Certificate exported to: $OutputPath" "Green"
        
        # Convert to base64 for GitHub secrets
        Write-ColorOutput "`nConverting to base64 for GitHub Secrets..." "Yellow"
        $certBytes = [System.IO.File]::ReadAllBytes($OutputPath)
        $base64 = [System.Convert]::ToBase64String($certBytes)
        
        $base64Path = "$OutputPath.base64"
        $base64 | Set-Content $base64Path
        
        Write-ColorOutput "Base64 encoded certificate saved to: $base64Path" "Green"
        Write-ColorOutput "`nTo add to GitHub Secrets:" "Cyan"
        Write-Host "1. Copy the contents of $base64Path"
        Write-Host "2. Add as WINDOWS_CERTIFICATE secret"
        Write-Host "3. Add the password as WINDOWS_CERTIFICATE_PASSWORD secret"
        
    } catch {
        Write-ColorOutput "Error exporting certificate: $_" "Red"
        exit 1
    }
}

function Test-SignedBinary {
    if (-not $CertPath) {
        Write-ColorOutput "Error: -CertPath is required for validate action" "Red"
        exit 1
    }
    
    if (-not (Test-Path $CertPath)) {
        Write-ColorOutput "Error: File not found: $CertPath" "Red"
        exit 1
    }
    
    Write-ColorOutput "`n=== Validating Signed Binary ===" "Cyan"
    
    # Get signature
    $sig = Get-AuthenticodeSignature -FilePath $CertPath
    
    Write-Host "File: $CertPath"
    Write-Host "Status: $($sig.Status)"
    Write-Host "Status Message: $($sig.StatusMessage)"
    
    if ($sig.SignerCertificate) {
        Write-Host "`nSigner Certificate:"
        Write-Host "  Subject: $($sig.SignerCertificate.Subject)"
        Write-Host "  Issuer: $($sig.SignerCertificate.Issuer)"
        Write-Host "  Thumbprint: $($sig.SignerCertificate.Thumbprint)"
        Write-Host "  Valid From: $($sig.SignerCertificate.NotBefore)"
        Write-Host "  Valid To: $($sig.SignerCertificate.NotAfter)"
    }
    
    if ($sig.TimeStamperCertificate) {
        Write-Host "`nTimestamp Certificate:"
        Write-Host "  Subject: $($sig.TimeStamperCertificate.Subject)"
        Write-Host "  Timestamp: $($sig.TimeStampTime)"
    }
    
    # Use signtool for detailed verification
    $signtool = "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe"
    if (-not (Test-Path $signtool)) {
        # Try alternative paths
        $signtool = "${env:ProgramFiles(x86)}\Windows Kits\10\bin\x64\signtool.exe"
    }
    
    if (Test-Path $signtool) {
        Write-ColorOutput "`nDetailed verification with signtool:" "Yellow"
        & $signtool verify /pa /v $CertPath
    }
}

function Monitor-CertificateExpiry {
    Write-ColorOutput "`n=== Certificate Expiry Monitor ===" "Cyan"
    
    $certs = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert
    $warnings = @()
    
    foreach ($cert in $certs) {
        $daysRemaining = ($cert.NotAfter - (Get-Date)).Days
        
        if ($daysRemaining -lt 90) {
            $warnings += [PSCustomObject]@{
                Subject = $cert.Subject
                Thumbprint = $cert.Thumbprint
                ExpiryDate = $cert.NotAfter
                DaysRemaining = $daysRemaining
                Status = if ($daysRemaining -lt 30) { "Critical" } elseif ($daysRemaining -lt 60) { "Warning" } else { "Info" }
            }
        }
    }
    
    if ($warnings.Count -gt 0) {
        Write-ColorOutput "Certificates requiring attention:" "Yellow"
        $warnings | Format-Table -AutoSize
        
        # Generate renewal report
        $reportPath = Join-Path $PSScriptRoot "certificate-renewal-report.txt"
        $report = @"
Certificate Renewal Report
Generated: $(Get-Date)

Certificates Requiring Renewal:
"@
        
        foreach ($warning in $warnings) {
            $report += @"

Certificate: $($warning.Subject)
Thumbprint: $($warning.Thumbprint)
Expires: $($warning.ExpiryDate)
Days Remaining: $($warning.DaysRemaining)
Status: $($warning.Status)
"@
        }
        
        $report | Set-Content $reportPath
        Write-ColorOutput "`nRenewal report saved to: $reportPath" "Green"
    } else {
        Write-ColorOutput "All certificates are valid for more than 90 days" "Green"
    }
}

# Main execution
switch ($Action) {
    'list' { Get-CodeSigningCertificates }
    'install' { Install-CodeSigningCertificate }
    'test' { Test-CodeSigning }
    'export' { Export-CertificateForCI }
    'validate' { Test-SignedBinary }
    'monitor' { Monitor-CertificateExpiry }
}

if ($Verbose) {
    Write-ColorOutput "`n=== Certificate Store Locations ===" "Cyan"
    Write-Host "Current User Personal: Cert:\CurrentUser\My"
    Write-Host "Local Machine Personal: Cert:\LocalMachine\My"
    Write-Host "Current User Trusted Publishers: Cert:\CurrentUser\TrustedPublisher"
    Write-Host "Local Machine Trusted Publishers: Cert:\LocalMachine\TrustedPublisher"
}