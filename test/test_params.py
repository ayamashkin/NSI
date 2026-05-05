#!/usr/bin/env python3
"""
Diagnostic script for params extraction debugging.
Traces exactly what happens at each step of the pipeline.

Version: 2025-05-06-fix3

Usage:
    python test_params.py --text "Болт (2)-12-44-Окс.Фос.ЭФП-ОСТ 1 31133-80" \
        --pattern "^Болт\\s*(?:\\((?P<исполнение>\\d+)\\)\\s*)?(?P<номинальный_диаметр_резьбы>\\d+(?:[.,]\\d+)?)\\s*[-\\s]*\\s*(?P<длина>\\d+(?:[.,]\\d+)?)\\s*[-\\s]*\\s*(?P<покрытие>[\\w.]+)\\s*$" \
        --standard "ОСТ 1 31133-80"

NOTE: On Windows cmd.exe, pass pattern as a single-quoted or double-quoted string.
The script auto-normalizes double backslashes from CLI escaping.
"""

import sys, re, argparse
sys.path.insert(0, '/mnt/agents/output')

from core.parametric_client import ParametricENSClient


def _normalize_cli_pattern(pattern: str) -> str:
    """
    CLI escaping on Windows may produce double backslashes.
    Convert known regex sequences from double to single backslash.
    E.g. r'\\s' (two chars in string) -> '\\s' (one char = whitespace metachar).
    """
    # Sequences that should have exactly one backslash in regex
    sequences = [
        r'\\s', r'\\S', r'\\d', r'\\D', r'\\w', r'\\W',
        r'\\b', r'\\B', r'\\A', r'\\Z', r'\\n', r'\\r', r'\\t',
        r'\\(', r'\\)', r'\\[', r'\\]', r'\\{', r'\\}',
        r'\\.', r'\\+', r'\\*', r'\\?', r'\\|', r'\\^', r'\\$',
        r'\\-', r'\\=', r'\\!', r'\\<', r'\\>', r'\\:', r'\\#',
    ]
    result = pattern
    for double in sequences:
        single = double[1:]  # e.g. r'\s' from r'\\s'
        result = result.replace(double, single)
    return result


def diagnose(text: str, pattern: str, standard: str = None):
    print(f"\n{'='*60}")
    print(f"TEXT:     {text}")
    print(f"PATTERN:  {pattern[:120]}...")
    print(f"STANDARD: {standard}")
    print(f"{'='*60}")

    client = ParametricENSClient.__new__(ParametricENSClient)

    # Step 1: Relax pattern
    relaxed = client._relax_pattern(pattern, standard=standard)
    print(f"\n[1] RELAXED PATTERN ({len(relaxed)} chars):")
    print(relaxed[:200])
    if len(relaxed) > 200:
        print("...")

    # Step 2: Try to compile
    try:
        compiled = re.compile(relaxed, re.IGNORECASE)
        print(f"\n[2] COMPILE: OK")
    except re.error as e:
        print(f"\n[2] COMPILE: FAILED - {e}")
        return

    # Step 3: Try search
    m = compiled.search(text)
    if m:
        print(f"\n[3] REGEX MATCH: YES")
        print(f"    groups: {m.groupdict()}")
    else:
        print(f"\n[3] REGEX MATCH: NO")
        # Try to find where it breaks
        for i in range(len(text), 0, -1):
            prefix = text[:i]
            if compiled.search(prefix):
                print(f"    longest matching prefix: '{prefix}' (len={i})")
                break
        else:
            print(f"    no prefix matches at all")

    # Step 4: Try original pattern
    m_orig = re.search(pattern, text, re.IGNORECASE)
    if m_orig:
        print(f"\n[4] ORIGINAL PATTERN MATCH: YES")
        print(f"    groups: {m_orig.groupdict()}")
    else:
        print(f"\n[4] ORIGINAL PATTERN MATCH: NO")

    # Step 5: Show difference between original and relaxed
    if relaxed != pattern:
        print(f"\n[5] CHANGES MADE BY _relax_pattern:")
        import difflib
        diff = list(difflib.ndiff(pattern, relaxed))
        changes = [d for d in diff if d.startswith('+ ') or d.startswith('- ')]
        for c in changes[:20]:
            print(f"    {c}")
        if len(changes) > 20:
            print(f"    ... and {len(changes)-20} more changes")
    else:
        print(f"\n[5] _relax_pattern made NO changes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose params extraction')
    parser.add_argument('--text', required=True, help='Input text')
    parser.add_argument('--pattern', required=True, help='Regex pattern')
    parser.add_argument('--standard', default=None, help='Standard (e.g., "ОСТ 1 31133-80")')
    args = parser.parse_args()

    normalized_pattern = _normalize_cli_pattern(args.pattern)
    if normalized_pattern != args.pattern:
        print(f"[INFO] CLI pattern normalized (double backslashes fixed)")

    diagnose(args.text, normalized_pattern, args.standard)