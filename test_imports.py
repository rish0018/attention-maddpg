#!/usr/bin/env python3
"""Quick test to verify all modules can be imported."""

import sys
import traceback

print("Testing imports...")

modules_to_test = [
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("model", "Neural Network Models"),
    ("ddpg_agent", "DDPG Agent"),
    ("replaybuffers", "Replay Buffers"),
    ("utils", "Utilities"),
]

all_ok = True

for module_name, description in modules_to_test:
    try:
        if module_name in ("torch", "numpy"):
            __import__(module_name)
        else:
            __import__(module_name)
        print(f"✓ {description:30} - OK")
    except Exception as e:
        print(f"✗ {description:30} - FAILED")
        print(f"  Error: {e}")
        traceback.print_exc()
        all_ok = False

if all_ok:
    print("\n✅ All imports successful!")
    print("\nNote: To run training, you need to provide a Unity environment file.")
    print("Usage: python main.py --env_file <path_to_env> --episodes 2000")
    sys.exit(0)
else:
    print("\n❌ Some imports failed!")
    sys.exit(1)
