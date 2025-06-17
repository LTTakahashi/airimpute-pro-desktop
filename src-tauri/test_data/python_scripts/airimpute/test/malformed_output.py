#!/usr/bin/env python3
"""Test script that produces malformed output."""

def main():
    """Print non-JSON output to test deserialization error handling."""
    print("This is not valid JSON output!")
    print("It contains multiple lines")
    print("And no JSON structure at all")

if __name__ == "__main__":
    main()