#!/bin/bash
# PyO3 configuration wrapper for Linux builds

export PYO3_PYTHON=/usr/bin/python3
export PYTHON=/usr/bin/python3

# Get Python configuration
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")
PYTHON_LIB=$(python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")

export PYO3_CROSS_LIB_DIR=$PYTHON_LIB
export PYO3_CROSS_PYTHON_VERSION=$PYTHON_VERSION

echo "Configured PyO3 for Python $PYTHON_VERSION"
echo "Include: $PYTHON_INCLUDE"
echo "Lib: $PYTHON_LIB"

# Run the actual cargo command
exec "$@"