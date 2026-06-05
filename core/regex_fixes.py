# =============================================================================
# FILE: core/regex_fixes.py
# REPO: https://github.com/ayamashkin/NSI
# =============================================================================
# FEAT 2026-06-03 12:30:00 UTC+3:
# Post-processing fixes for LLM-generated regex patterns.
# Add to llm_mask_generator.py: import and call _fix_pattern(pattern) 
# after LLM response parsing, before re.compile().
# =============================================================================
"""
Post-processing fixes for LLM-generated regex patterns.

Usage in llm_mask_generator.py:
    from core.regex_fixes import _fix_pattern
    
    # After parsing LLM response:
    pattern = parsed['pattern']
    pattern = _fix_pattern(pattern)
    # Then compile...
"""

import re
import logging

logger = logging.getLogger(__name__)


def _fix_pattern(pattern: str) -> str:
    """Apply all post-processing fixes to an LLM-generated pattern.
    
    Fixes applied:
    1. Deduplicate group names (Python re forbids duplicate (?P<name>))
    2. Remove empty alternation branches (| followed by ) or (|)
    3. Fix common LLM mistakes: \\( → (, \\) → ), \\. → .
    """
    if not pattern:
        return pattern
    
    original = pattern
    
    # Fix 1: Deduplicate group names
    pattern = _deduplicate_group_names(pattern)
    
    # Fix 2: Remove empty alternation branches
    pattern = re.sub(r'\|+', '|', pattern)  # ||| → |
    pattern = re.sub(r'\(\|', '(', pattern)  # (|abc) → (abc)
    pattern = re.sub(r'\|\)', ')', pattern)  # (abc|) → (abc)
    
    # Fix 3: Double backslash fixes
    pattern = pattern.replace('\\\\(', '(')
    pattern = pattern.replace('\\\\)', ')')
    pattern = pattern.replace('\\\\.', '.')
    pattern = pattern.replace('\\\\-', '-')
    
    if pattern != original:
        logger.debug("[regex_fixes] Pattern modified: %s", 
                     f"{original[:80]}... → {pattern[:80]}...")
    
    return pattern


def _deduplicate_group_names(pattern: str) -> str:
    """Rename duplicate named groups to avoid re.error.
    
    Python's re module does not allow duplicate group names,
    even in different alternation branches.
    
    Example:
        (?P<a>\d+)|(?P<a>\d+) → (?P<a>\d+)|(?P<a_2>\d+)
    
    The original name is kept for the first occurrence.
    Subsequent duplicates get _2, _3, etc.
    
    Returns a tuple: (fixed_pattern, name_mapping) where name_mapping
    maps renamed groups back to original: {'a_2': 'a'}
    """
    seen = set()
    name_mapping = {}  # renamed -> original
    counter = {}       # original -> next suffix
    
    def replace_duplicate(match):
        prefix = match.group(1)   # (?P<
        name = match.group(2)     # group_name
        suffix = match.group(3)   # >...)
        
        if name not in seen:
            seen.add(name)
            return match.group(0)  # Keep original
        
        # Duplicate found — rename
        if name not in counter:
            counter[name] = 2
        new_name = f"{name}_{counter[name]}"
        counter[name] += 1
        name_mapping[new_name] = name
        
        logger.debug("[regex_fixes] Renamed duplicate group: %s → %s", name, new_name)
        return f"{prefix}{new_name}{suffix}"
    
    # Match (?P<name> where name is the group name
    fixed = re.sub(r'(\(\?P<)([^>]+)(>)', replace_duplicate, pattern)
    
    return fixed


def _merge_duplicate_groups(match_dict: dict, name_mapping: dict = None) -> dict:
    """Merge values from renamed duplicate groups back to original names.
    
    After deduplication, if (?P<a>) matched in first branch, it's 'a'.
    If in second branch, it's 'a_2'. This function merges 'a_2' into 'a'
    if 'a' is None.
    
    Usage after match.groupdict():
        params = match.groupdict()
        params = _merge_duplicate_groups(params, name_mapping)
    """
    if not name_mapping:
        return match_dict
    
    result = dict(match_dict)
    for renamed, original in name_mapping.items():
        if renamed in result and result[renamed] is not None:
            if original not in result or result[original] is None:
                result[original] = result[renamed]
            del result[renamed]
    
    return result
