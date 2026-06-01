#!/usr/bin/env python3
"""
Patch script for core/llm_mask_generator.py
Applies 3 minimal fixes to the original repo file.
"""
import re

with open('core/llm_mask_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# FIX 1: Broken re.sub that causes "unknown extension ?P\("
# Replace the invalid regex with a safe string replacement
old_fix1 = """        # FIX 2026-05-28 21:20 UTC+3: add optional separator between )? and next named group
        # e.g. (?:[-\\s]+\\((?P<<исполнение>\\d+)\\))?(?P<<номинальный_диаметр_резьбы>\\d+)
        pattern = re.sub(r'\\)\\?(?P\\(\\?P<<[^>]+>)', lambda m: f')?(?:[-\\s]+)?{m.group("next")}', pattern)"""

new_fix1 = """        # FIX 2026-05-28 21:20 UTC+3: add optional separator between )? and next named group
        # e.g. (?:[-\\s]+\\((?P<<исполнение>\\d+)\\))?(?P<<номинальный_диаметр_резьбы>\\d+)
        # FIX 2026-06-01 12:20 UTC+3: removed broken regex that caused "unknown extension ?P\\("
        pattern = pattern.replace(')?(?P<<', ')?(?:[-\\\\s]+)?(?P<<')"""

if old_fix1 in content:
    content = content.replace(old_fix1, new_fix1)
    print("✅ FIX 1 applied: broken re.sub replaced with safe string replacement")
else:
    print("⚠️ FIX 1: pattern not found, may already be patched or file differs from repo")

# FIX 2: Add auto-fix for dot separator before покрытие
# Insert after FIX 1
marker = "        # FIX 2026-06-01 12:20 UTC+3: removed broken regex that caused \"unknown extension ?P\\(\"\n        pattern = pattern.replace(')?(?P<<', ')?(?:[-\\\\s]+)?(?P<<')"
if marker in content and "auto-fix dot separator before покрытие" not in content:
    insert_after = "        pattern = pattern.replace(')?(?P<<', ')?(?:[-\\\\s]+)?(?P<<')"
    new_block = insert_after + "\n\n" + """        # FIX 2026-06-01 11:55 UTC+3: auto-fix dot separator before покрытие -> [-\\s]+
        # LLM sometimes generates \\.(?P<<покрытие>...) because coating values contain dots (Кд3.фос.окс)
        # But the actual separator in text is always hyphen or space.
        pattern = re.sub(
            r'\\.\\(?P<<покрытие>',
            r'[-\\s]+(?P<<покрытие>',
            pattern,
        )
        logger.debug("[LLMMaskGenerator] _fix_pattern: applied dot->separator fix for покрытие")"""
    content = content.replace(insert_after, new_block)
    print("✅ FIX 2 applied: added auto-fix for dot before покрытие")
else:
    print("⚠️ FIX 2: skipped (already present or marker not found)")

# FIX 3: SyntaxWarning on re.sub with \s inside []
old_warning1 = "        pattern = re.sub(r'\\s+\\[-\\s\\]\\+', lambda m: r'[-\\s]+', pattern)"
old_warning2 = "        pattern = re.sub(r'\\[-\\s\\]\\+\\s+', lambda m: r'[-\\s]+', pattern)"

new_warning1 = "        pattern = re.sub(r'\\s+\\[-\\\\s\\]\\+', lambda m: r'[-\\s]+', pattern)"
new_warning2 = "        pattern = re.sub(r'\\[-\\\\s\\]\\+\\s+', lambda m: r'[-\\s]+', pattern)"

if old_warning1 in content:
    content = content.replace(old_warning1, new_warning1)
    print("✅ FIX 3a applied: fixed SyntaxWarning in first re.sub")
else:
    print("⚠️ FIX 3a: first pattern not found")

if old_warning2 in content:
    content = content.replace(old_warning2, new_warning2)
    print("✅ FIX 3b applied: fixed SyntaxWarning in second re.sub")
else:
    print("⚠️ FIX 3b: second pattern not found")

# Verify syntax
import ast
try:
    ast.parse(content)
    print("\n✅ Syntax check passed!")
except SyntaxError as e:
    print(f"\n❌ SyntaxError: {e}")
    exit(1)

# Save
with open('core/llm_mask_generator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("💾 File saved: core/llm_mask_generator.py")
print("\nТеперь можно запускать:")
print('python cli.py generate-masks -d cache/masks.db -i models/ens_hardware_test.pkl --force --llm --domain hardware --validate --standard "ОСТ 1 31133-80" -so output/mask_stats.xlsx')