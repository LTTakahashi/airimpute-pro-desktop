"""
GObject Introspection Version Fixing
Forces GI to use libsoup 2.4 instead of 3.0 to prevent conflicts
"""

import sys

def fix_gi_versions():
    """Force GI to use compatible versions before any imports"""
    try:
        import gi
        
        # Force specific versions BEFORE any repository imports
        # This MUST happen before any other gi.repository imports anywhere
        gi.require_version('Soup', '2.4')
        gi.require_version('WebKit2', '4.0')
        gi.require_version('Gtk', '3.0')
        
        print("GI versions fixed: Soup 2.4, WebKit2 4.0, Gtk 3.0", file=sys.stderr)
        return True
    except ImportError:
        # gi not installed, no problem
        return True
    except Exception as e:
        print(f"Warning: Could not fix GI versions: {e}", file=sys.stderr)
        return False

# Run immediately on import
fix_gi_versions()

# Prevent any module from importing the wrong versions
if 'gi' in sys.modules:
    import gi
    # Monkey patch to prevent version conflicts
    _original_require = gi.require_version
    
    def _safe_require_version(namespace, version):
        """Override version requirements to prevent conflicts"""
        if namespace == 'Soup' and version != '2.4':
            print(f"Warning: Attempted to load Soup {version}, forcing 2.4", file=sys.stderr)
            version = '2.4'
        return _original_require(namespace, version)
    
    gi.require_version = _safe_require_version