#!/usr/bin/env python3
"""Test script that intentionally fails."""

import sys

def main():
    """Raise an exception to test error handling."""
    raise ValueError("This operation intentionally fails for testing purposes")

if __name__ == "__main__":
    main()