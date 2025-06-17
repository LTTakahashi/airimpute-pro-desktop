#!/bin/bash

# Disable all workflows except build-windows.yml
cd .github/workflows/

for file in *.yml; do
    if [ "$file" != "build-windows.yml" ]; then
        echo "Disabling workflow: $file"
        mv "$file" "$file.disabled"
    fi
done

echo "All workflows disabled except build-windows.yml"
ls -la