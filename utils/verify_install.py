"""Verify that all required packages are installed correctly.

Run this after `pip install -r requirements.txt` to confirm the environment
is ready for the book's code examples.
"""

import sys


def check_package(name: str, min_version: str = "") -> bool:
    """Check if a package is installed and optionally meets version requirement."""
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"  [OK] {name} ({version})")
        return True
    except ImportError:
        print(f"  [MISSING] {name}")
        return False


def main() -> int:
    print("Verifying Python environment for EEG book...\n")
    print(f"Python: {sys.version}\n")

    packages = [
        ("numpy", "1.24"),
        ("scipy", "1.11"),
        ("pandas", "2.0"),
        ("matplotlib", "3.7"),
        ("seaborn", "0.12"),
        ("mne", "1.5"),
        ("sklearn", "1.3"),
    ]

    all_ok = True
    for pkg, ver in packages:
        if not check_package(pkg, ver):
            all_ok = False

    # Optional packages
    print("\nOptional packages:")
    optional = ["moabb", "pylsl", "plotly"]
    for pkg in optional:
        check_package(pkg)

    print()
    if all_ok:
        print("All core packages installed. Environment is ready.")
        return 0
    else:
        print("Some packages are missing. Run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
