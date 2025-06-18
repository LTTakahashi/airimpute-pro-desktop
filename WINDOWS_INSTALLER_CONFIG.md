# Windows Installer Configuration

## Installer Choice: NSIS

We've chosen NSIS (Nullsoft Scriptable Install System) over WiX for the Windows installer based on:

- **Simplicity**: NSIS has a simpler configuration for straightforward installations
- **Size**: Generally produces smaller installers
- **Sufficient for our needs**: Perfect for "copy files and create shortcuts" scenarios

## Security Considerations

### Current Status (Development)
- **Unsigned installer**: `certificateThumbprint: null`
- **Per-user installation**: Installs to user's AppData, no admin required
- **Timestamp URL configured**: Ensures signatures remain valid after cert expiry

### Required for Production
1. **Code Signing Certificate**: 
   - Purchase from trusted CA (DigiCert, Sectigo, etc.)
   - Configure in CI/CD secrets:
     - `TAURI_PRIVATE_KEY`: Path to .pfx file
     - `TAURI_KEY_PASSWORD`: Certificate password
   - Update `certificateThumbprint` in tauri.conf.json

2. **Branding Assets** (Optional but recommended):
   - `installerIcon`: .ico file for installer
   - `headerImage`: .bmp file (150x57 pixels)
   - `sidebarImage`: .bmp file (164x314 pixels)
   - `license`: EULA.rtf or EULA.txt file

## Installation Mode: perUser

We use `perUser` installation mode for better security:
- No admin privileges required
- Installs to `%LOCALAPPDATA%`
- Limits potential security vulnerabilities
- Suitable for scientific/professional applications

## Future Considerations

If enterprise deployment becomes a requirement:
- Consider migrating to WiX for MSI packages
- MSI supports Group Policy deployment
- Better for complex enterprise scenarios