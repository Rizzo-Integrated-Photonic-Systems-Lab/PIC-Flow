#!/usr/bin/env python3
"""
Test script to verify import structure after FDTD reorganization.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

def test_import_structure():
    """Test that the import structure works correctly."""
    try:
        # Test importing from the main FDTD package
        print("Testing FDTD package structure...")

        # This should work now with the fixed imports
        from FDTD.utils import neff_siwire_from_tables
        print("✓ FDTD.utils import successful")

        from FDTD.devices_base import Device2DBase
        print("✓ FDTD.devices_base import successful")

        # Test the straight_waveguide submodule
        print("\nTesting straight_waveguide submodule...")
        # We can't import the class without meep, but we can check the module structure
        import FDTD.straight_waveguide
        print("✓ FDTD.straight_waveguide module import successful")

        print("\n🎉 All import structure tests passed!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_import_structure()
    sys.exit(0 if success else 1)
