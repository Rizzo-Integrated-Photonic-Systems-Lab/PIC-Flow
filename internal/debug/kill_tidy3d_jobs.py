#!/usr/bin/env python3
"""
Script to check Tidy3D job management capabilities
"""
import sys

print("Tidy3D Job Management Check")
print("=" * 40)

try:
    import tidy3d.web as web
    print("✓ Tidy3D web interface loaded")

    # Show available methods
    methods = [m for m in dir(web) if not m.startswith('_')]
    print(f"\nAvailable methods ({len(methods)}):")
    for method in sorted(methods)[:20]:  # Show first 20 methods
        print(f"  - {method}")
    if len(methods) > 20:
        print(f"  ... and {len(methods)-20} more")

    print("\n" + "=" * 40)
    print("JOB CANCELLATION:")
    print("This Tidy3D version doesn't support automatic job cancellation.")
    print("Please manually cancel jobs at:")
    print("https://tidy3d.simulation.cloud/jobs")
    print("=" * 40)

    print("\nFor your current jobs, you can:")
    print("1. Visit the web interface above")
    print("2. Cancel any running jobs manually")
    print("3. Then run your new cloud-based neff_extract.py")

except ImportError as e:
    print(f"✗ ERROR: {e}")
    print("Run this script in your SLURM job with 'conda activate PBFM'")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)
