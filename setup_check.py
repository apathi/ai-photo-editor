"""
Setup verification script for AI Photo Editor.
Checks dependencies, environment, and provides helpful information.
"""

import sys
import subprocess
import importlib.util


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro}")
        print("  Python 3.9+ required")
        return False


def check_package(package_name, display_name=None):
    """Check if a package is installed."""
    if display_name is None:
        display_name = package_name

    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        print(f"  ✅ {display_name}")
        return True
    else:
        print(f"  ❌ {display_name} (not installed)")
        return False


def check_dependencies():
    """Check critical dependencies."""
    print("\nChecking dependencies...")

    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
        ("PIL", "Pillow"),
        ("flask", "Flask"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("skimage", "scikit-image"),
    ]

    results = []
    for package, display_name in packages:
        results.append(check_package(package, display_name))

    return all(results)


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU support...")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✅ CUDA available: {device_name}")
            print(f"  CUDA version: {torch.version.cuda}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("  ✅ MPS (Apple Silicon) available")
            return "mps"
        else:
            print("  ⚠️  No GPU detected - will use CPU")
            print("  Note: CPU inference will be significantly slower")
            return "cpu"
    except ImportError:
        print("  ❌ PyTorch not installed")
        return None


def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)

        if free_gb >= 20:
            print(f"  ✅ {free_gb}GB free (20GB+ recommended for models)")
            return True
        elif free_gb >= 10:
            print(f"  ⚠️  {free_gb}GB free (models require ~10GB)")
            return True
        else:
            print(f"  ❌ {free_gb}GB free (insufficient space)")
            return False
    except Exception as e:
        print(f"  ⚠️  Could not check disk space: {e}")
        return True


def check_memory():
    """Check available RAM."""
    print("\nChecking memory...")
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total // (2**30)

        if total_gb >= 16:
            print(f"  ✅ {total_gb}GB RAM (16GB+ recommended)")
            return True
        elif total_gb >= 8:
            print(f"  ⚠️  {total_gb}GB RAM (8GB minimum, 16GB recommended)")
            return True
        else:
            print(f"  ❌ {total_gb}GB RAM (insufficient memory)")
            return False
    except ImportError:
        print("  ⚠️  psutil not installed, cannot check memory")
        return True


def print_next_steps(all_checks_passed):
    """Print next steps."""
    print("\n" + "=" * 60)

    if all_checks_passed:
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the application:")
        print("   python app.py")
        print("\n2. Open browser:")
        print("   http://localhost:5000")
        print("\n3. Try the demo:")
        print("   python examples/demo.py")
        print("\n4. Run tests:")
        print("   python tests/test_pipeline.py")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before proceeding.")
        print("\nTo install missing dependencies:")
        print("   pip install -r requirements.txt")


def main():
    """Run all checks."""
    print("=" * 60)
    print("AI PHOTO EDITOR - SETUP VERIFICATION")
    print("=" * 60)

    checks = []

    # Run all checks
    checks.append(check_python_version())
    checks.append(check_dependencies())
    gpu_status = check_gpu()
    checks.append(gpu_status is not None)
    checks.append(check_disk_space())
    check_memory()  # Warning only, not required

    # Summary
    all_passed = all(checks)
    print_next_steps(all_passed)

    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
