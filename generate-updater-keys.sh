#!/bin/bash
# Generate Tauri updater signing keys

echo "Generating Tauri updater signing keys..."

# Create directory for keys
mkdir -p keys

# Generate a keypair using openssl (alternative to interactive tauri signer)
openssl genrsa -out keys/private.key 4096
openssl rsa -in keys/private.key -pubout -out keys/public.key

# Convert to format expected by Tauri
echo "Converting keys to Tauri format..."

# Extract the public key content
PUBLIC_KEY=$(cat keys/public.key | grep -v "BEGIN" | grep -v "END" | tr -d '\n')

echo "Keys generated successfully!"
echo ""
echo "Public key (add this to tauri.conf.json):"
echo "$PUBLIC_KEY"
echo ""
echo "Private key location: keys/private.key"
echo ""
echo "IMPORTANT: Keep the private key secure and never commit it to version control!"