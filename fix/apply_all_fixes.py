"""
Master fix script — applies all patches for accuracy improvement.
Run from project root: python apply_all_fixes.py

VERSION: 2026-05-07 09:30 UTC+3
"""

import shutil
import subprocess
import sys
from pathlib import Path


def copy_file(src_name: str, dst_dirs: list):
    """Copy file from output/nomenclature_processor/ to destination."""
    src = Path(__file__).parent / src_name

    for dst_dir in dst_dirs:
        dst = Path(dst_dir) / src_name
        if dst.parent.exists():
            shutil.copy2(src, dst)
            print(f"  [OK] {src_name} -> {dst_dir}/")
            return True

    print(f"  [WARN] Could not find destination for {src_name}")
    return False


def main():
    print("=" * 60)
    print("Applying all fixes for nomenclature processor")
    print("=" * 60)

    # 1. Copy updated Python files
    print("\n[1/3] Copying updated Python files...")

    destinations = ['.', 'nomenclature_processor', 'core', 'parsers', 'database']

    files_to_copy = [
        'parametric_client.py',
        'automated_processor.py',
        'standard_extractor.py',
    ]

    for fname in files_to_copy:
        copy_file(fname, destinations)

    # 2. Run mask fixes
    print("\n[2/3] Applying mask database fixes...")
    fix_script = Path(__file__).parent / "fix_masks_v2.py"
    if fix_script.exists():
        result = subprocess.run(
            [sys.executable, str(fix_script)],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  [ERROR] {result.stderr}")
    else:
        print("  [WARN] fix_masks_v2.py not found")

    # 3. Summary
    print("\n[3/3] Summary of changes:")
    print("  - parametric_client.py: coating variants Кд→Кд6/Кд9.фос.окс")
    print("  - automated_processor.py: washer generic pattern support")
    print("  - standard_extractor.py: ensure washer standard extraction")
    print("  - Database: fixed ГОСТ 7795-70 mask, added Шайба masks")

    print("\n" + "=" * 60)
    print("Expected accuracy improvement:")
    print("  Before: 4/23 success")
    print("  After:  ~8/23 success (+4 from masks, +potential from coating)")
    print("  Remaining: 8 correctly rejected (param mismatch)")
    print("             7 need ENS data (Кд coating entries)")
    print("=" * 60)


if __name__ == '__main__':
    main()