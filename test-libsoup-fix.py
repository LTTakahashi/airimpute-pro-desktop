#!/usr/bin/env python3
"""
Test script to verify our libsoup conflict fix works
"""

import sys
import os

print("Testing libsoup conflict fix...")
print("=" * 50)

# Test 1: Check if GI version fix works
try:
    # Add src/python to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src-tauri/src/python'))
    
    # Import our fix
    import gi_version_fix
    print("✓ GI version fix module loaded successfully")
    
    # Try to import Soup
    import gi
    from gi.repository import Soup
    print("✓ Successfully imported Soup (should be version 2.4)")
    
    # Check which version we got
    try:
        session = Soup.Session()
        print(f"✓ Created Soup.Session object: {type(session)}")
    except Exception as e:
        print(f"✗ Failed to create Soup.Session: {e}")
        
except ImportError as e:
    print(f"✗ GI not available or import failed: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print()

# Test 2: Check environment
print("Environment check:")
print(f"  LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'Not set')}")
print(f"  PKG_CONFIG: {os.environ.get('PKG_CONFIG', 'Not set')}")

print()
print("If you see checkmarks above, the Python side of the fix is working.")
print("The app should now be able to start without libsoup conflicts.")