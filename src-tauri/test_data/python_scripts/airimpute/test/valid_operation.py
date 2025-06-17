#!/usr/bin/env python3
"""Test script that performs a valid operation."""

import json
import sys

def main():
    """Read JSON from stdin, process it, and return JSON to stdout."""
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Perform simple processing
        result = {
            "status": "success",
            "value": 42,
            "input_received": input_data,
            "message": "Operation completed successfully"
        }
        
        # Write result to stdout
        print(json.dumps(result))
        sys.exit(0)
        
    except Exception as e:
        # Return error in structured format
        error_result = {
            "status": "error",
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()