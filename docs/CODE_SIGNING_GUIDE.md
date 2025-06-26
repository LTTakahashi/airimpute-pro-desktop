# Windows Code Signing Guide for AirImpute Pro

## Overview

This guide provides comprehensive instructions for implementing Windows code signing for AirImpute Pro desktop application. Code signing ensures users can trust your application and prevents Windows SmartScreen warnings.

## Table of Contents

1. [Certificate Types and Requirements](#certificate-types-and-requirements)
2. [Certificate Procurement](#certificate-procurement)
3. [Local Development Setup](#local-development-setup)
4. [CI/CD Integration](#cicd-integration)
5. [Security Best Practices](#security-best-practices)
6. [Troubleshooting](#troubleshooting)

## Certificate Types and Requirements

### Code Signing Certificate Types

1. **Standard Code Signing Certificate**
   - Basic trust level
   - Requires building reputation with SmartScreen
   - Lower cost (~$200-400/year)
   - Suitable for internal or small-scale distribution

2. **EV (Extended Validation) Code Signing Certificate**
   - Immediate SmartScreen reputation
   - Hardware token required
   - Higher cost (~$400-800/year)
   - Recommended for public distribution

### Requirements

- Certificate must be from a trusted Certificate Authority (CA)
- Must support SHA-256 signing
- Valid for the duration of your release cycle
- Compatible with Windows 7 and later

## Certificate Procurement

### Recommended Certificate Authorities

1. **DigiCert**
   - Industry leader
   - Excellent support
   - EV certificates available
   - Price: $474/year (Standard), $699/year (EV)

2. **Sectigo (formerly Comodo)**
   - Cost-effective
   - Good for startups
   - Price: $179/year (Standard), $399/year (EV)

3. **GlobalSign**
   - Enterprise-focused
   - Strong reputation
   - Price: $299/year (Standard), $599/year (EV)

### Procurement Process

1. **Choose Certificate Type**
   - Evaluate distribution needs
   - Consider budget constraints
   - Plan for renewal cycles

2. **Prepare Documentation**
   - Business registration documents
   - Domain ownership verification
   - Phone number for verification
   - DUNS number (for EV certificates)

3. **Purchase and Validation**
   - Complete online application
   - Submit required documents
   - Complete phone verification
   - Receive certificate (1-5 business days)

## Local Development Setup

### Installing the Certificate

1. **For PFX/P12 Certificates**
   ```powershell
   # Import certificate to personal store
   $cert = Get-PfxCertificate -FilePath "path\to\certificate.pfx"
   $store = New-Object System.Security.Cryptography.X509Certificates.X509Store "My", "CurrentUser"
   $store.Open("ReadWrite")
   $store.Add($cert)
   $store.Close()
   ```

2. **For Hardware Tokens (EV)**
   - Install token drivers
   - Connect token to USB
   - Use token password when signing

### Finding Certificate Thumbprint

```powershell
# List all code signing certificates
Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert | 
    Format-Table Subject, Thumbprint, NotAfter

# Or search by subject
Get-ChildItem Cert:\CurrentUser\My | 
    Where-Object { $_.Subject -like "*AirImpute*" } |
    Format-Table Subject, Thumbprint
```

### Configuring Tauri

Update `src-tauri/tauri.conf.json`:

```json
{
  "tauri": {
    "bundle": {
      "windows": {
        "certificateThumbprint": "YOUR_CERTIFICATE_THUMBPRINT_HERE",
        "digestAlgorithm": "sha256",
        "timestampUrl": "https://timestamp.digicert.com",
        "webviewInstallMode": {
          "type": "embedBootstrapper"
        }
      }
    }
  }
}
```

## CI/CD Integration

### GitHub Actions Setup

1. **Store Certificate in Secrets**

   Convert certificate to base64:
   ```powershell
   # Convert PFX to base64
   $cert = Get-Content -Path "certificate.pfx" -Encoding Byte
   [System.Convert]::ToBase64String($cert) | Set-Clipboard
   ```

   Add to GitHub Secrets:
   - `WINDOWS_CERTIFICATE`: Base64 encoded certificate
   - `WINDOWS_CERTIFICATE_PASSWORD`: Certificate password

2. **Update GitHub Workflow**

   See `.github/workflows/build-windows-signed.yml` for complete implementation.

### Azure Key Vault Integration (Recommended for Production)

1. **Setup Azure Key Vault**
   ```bash
   # Create Key Vault
   az keyvault create --name airimpute-signing \
     --resource-group airimpute-rg \
     --location eastus

   # Import certificate
   az keyvault certificate import \
     --vault-name airimpute-signing \
     --name code-signing-cert \
     --file certificate.pfx
   ```

2. **Configure Service Principal**
   ```bash
   # Create service principal
   az ad sp create-for-rbac --name airimpute-signing-sp

   # Grant access to Key Vault
   az keyvault set-policy --name airimpute-signing \
     --spn <SERVICE_PRINCIPAL_ID> \
     --certificate-permissions get list
   ```

## Security Best Practices

### Certificate Storage

1. **Never commit certificates to source control**
2. **Use encrypted storage for local certificates**
3. **Implement certificate rotation procedures**
4. **Maintain audit logs for certificate usage**

### Access Control

1. **Limit certificate access to authorized personnel**
2. **Use separate certificates for development and production**
3. **Implement multi-factor authentication for certificate access**
4. **Regular access reviews and revocation**

### Signing Process

1. **Always use timestamp servers**
2. **Verify signatures after building**
3. **Implement dual-signing for compatibility**
4. **Monitor certificate expiration dates**

## Troubleshooting

### Common Issues

1. **Certificate Not Found**
   ```powershell
   # Verify certificate is installed
   Get-ChildItem Cert:\CurrentUser\My
   
   # Check certificate store permissions
   certutil -store My
   ```

2. **Timestamp Server Failures**
   - Use fallback timestamp servers
   - Implement retry logic
   - Monitor server availability

3. **SmartScreen Warnings**
   - Build reputation over time
   - Consider EV certificate upgrade
   - Submit files to Microsoft for analysis

### Validation Tools

```powershell
# Verify signature
Get-AuthenticodeSignature -FilePath "AirImputePro.exe"

# Detailed signature information
signtool verify /pa /v "AirImputePro.exe"

# Check certificate chain
certutil -verify -urlfetch certificate.cer
```

## Automated Certificate Management

### Certificate Renewal Automation

1. **Monitor Expiration**
   ```powershell
   # Check certificate expiration
   $cert = Get-ChildItem Cert:\CurrentUser\My -CodeSigningCert
   $daysUntilExpiry = ($cert.NotAfter - (Get-Date)).Days
   
   if ($daysUntilExpiry -lt 30) {
       Write-Warning "Certificate expires in $daysUntilExpiry days!"
   }
   ```

2. **Automated Renewal Process**
   - Set up calendar reminders
   - Implement monitoring alerts
   - Prepare renewal documentation
   - Test new certificate before production

### Key Rotation

1. **Establish rotation schedule**
2. **Maintain overlap period**
3. **Update all systems simultaneously**
4. **Verify old signatures remain valid**

## References

- [Microsoft Code Signing Documentation](https://docs.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools)
- [Tauri Bundle Configuration](https://tauri.app/v1/api/config#bundleconfig)
- [Windows Authenticode](https://docs.microsoft.com/en-us/windows-hardware/drivers/install/authenticode)