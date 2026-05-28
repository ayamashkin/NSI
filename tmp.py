import re

FILE = "core/llm_mask_generator.py"

with open(FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Fix 1: _filter_unambiguous
text = text.replace(
    'if k in self.SKIP_PARAMS:\n                    continue',
    'if k in self.SKIP_PARAMS or k in self._SKIP_META_PARAMS:\n                    continue'
)

# Fix 2: _get_visible_params_from_index
text = text.replace(
    'visible = visible - metadata - self.SKIP_PARAMS',
    'visible = visible - metadata - self.SKIP_PARAMS - self._SKIP_META_PARAMS'
)

# Fix 3: _fix_pattern — add dot-fix before return pattern
old_fix_end = '        return pattern\n\n    def _sanitize_mask_result'
new_fix_end = '''        # Fix dots in named group names (e.g., наименование.1 -> наименование_1)
        pattern = re.sub(r'\\?P<([a-zA-Zа-яА-Я0-9_]+)\\.(\\d+)>', r'?P<\\1_\\2>', pattern)
        return pattern

    def _sanitize_mask_result'''
text = text.replace(old_fix_end, new_fix_end)

# Fix 4: _sanitize_mask_result — add dot-sanitize at start
old_sanitize_start = '    def _sanitize_mask_result(self, result: MaskGenerationResult) -> MaskGenerationResult:\n        pattern = result.pattern\n        params = list(result.params)\n        required = list(result.required)'
new_sanitize_start = '''    def _sanitize_mask_result(self, result: MaskGenerationResult) -> MaskGenerationResult:
        # Fix dots in param names (LLM sometimes generates наименование.1)
        params = [p.replace(".", "_") for p in list(result.params)]
        required = [p.replace(".", "_") for p in list(result.required)]
        pattern = result.pattern'''
text = text.replace(old_sanitize_start, new_sanitize_start)

with open(FILE, "w", encoding="utf-8") as f:
    f.write(text)

print("Patch applied successfully")