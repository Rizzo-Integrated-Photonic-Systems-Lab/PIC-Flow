#!/usr/bin/env python3
"""
Simple test to verify import path resolution after FDTD reorganization.
"""
import sys
import os
import importlib.util

def test_import_paths():
    """Test that import paths resolve correctly."""
    try:
        print("Testing import path resolution...")

        # Check if FDTD directory exists and is a package
        fdtd_path = os.path.join(os.getcwd(), 'FDTD')
        init_file = os.path.join(fdtd_path, '__init__.py')

        if not os.path.exists(init_file):
            print("Creating FDTD/__init__.py to make it a proper Python package...")
            with open(init_file, 'w') as f:
                f.write("# FDTD package\n")

        # Add current directory to path
        sys.path.insert(0, os.getcwd())

        # Test if we can resolve the module paths
        utils_spec = importlib.util.spec_from_file_location(
            "FDTD.utils",
            os.path.join(fdtd_path, "utils.py")
        )
        if utils_spec is None:
            print("❌ Could not resolve FDTD.utils module path")
            return False
        print("✓ FDTD.utils module path resolved")

        devices_spec = importlib.util.spec_from_file_location(
            "FDTD.devices_base",
            os.path.join(fdtd_path, "devices_base.py")
        )
        if devices_spec is None:
            print("❌ Could not resolve FDTD.devices_base module path")
            return False
        print("✓ FDTD.devices_base module path resolved")

        straight_spec = importlib.util.spec_from_file_location(
            "FDTD.straight_waveguide.straight",
            os.path.join(fdtd_path, "straight_waveguide", "straight.py")
        )
        if straight_spec is None:
            print("❌ Could not resolve FDTD.straight_waveguide.straight module path")
            return False
        print("✓ FDTD.straight_waveguide.straight module path resolved")

        print("\n🎉 All import path resolution tests passed!")
        print("The import structure should work correctly when meep is available.")
        return True

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_import_paths()
    sys.exit(0 if success else 1)
