#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
export WEBKIT_DISABLE_COMPOSITING_MODE=1
exec "$SCRIPT_DIR/airimpute-pro" "$@"
