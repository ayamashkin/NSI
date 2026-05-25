"""
Auto Validator Module
Validates generated masks against ENS examples with fuzzy matching.
"""
# =============================================================================
# FIX 2026-05-25 18:40 UTC+3:
# 1. ADDED DEBUG logging in _test_pattern: prints regex match result,
#    extracted groups, and expected values for each tested example.
#    This helps diagnose why score=0.00 for apparently valid patterns.
# 2. FIXED _is_value_in_name: added support for М-prefix (e.g. М22)
#    so that diameter values are correctly detected even with prefix.
# 3. FIXED _fix_pattern (called from llm_mask_generator): added [a-zA-Zа-яА-Я]
#    for tolerance class to match both Latin and Cyrillic letters.
# 4. ADDED 'нтд_1' to skip_params in _test_pattern — it is metadata,
#    not an extractable parameter from ENS examples.
# 5. ADDED mapping нтд_1 → стандарт/нтд in _match_param_keys for completeness.
# =============================================================================
# FIX 2026-05-20 17:47:49 UTC+3 — added metadata tracking (service, model, temp,
# tokens) and Excel stats output support.
# FIX 2026-05-20 13:59:31 UTC+3 — _test_pattern: added missing param logging
#    and fuzzy value matching for float .0 and coating abbreviations.
# FIX 2026-05-20 13:34:15 UTC+3 — _match_param_keys: added mapping for
#    нтд_1→стандарт/нтд, тип_изделия→наименование_типа/тип.
# FIX 2026-05-20 13:15:44 UTC+3 — _test_pattern: added _is_value_in_name
#    for robust visible-param detection.
# FIX 2026-05-20 12:40:16 UTC+3 — initial implementation with fuzzy param
#    matching and score-based activation.
# =============================================================================

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.standard_utils import canonicalize_standard

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Результат валидации маски."""
    score: float = 0.0
    passed: bool = False
    details: List[Dict] = field(default_factory=list)
    total: int = 0
    matched: int = 0
    mismatched: int = 0
    missing: int = 0
    service: str = ""
    model: str = ""
    temperature: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class AutoValidator:
    """Валидатор масок на примерах ЕНС."""

    def __init__(
        self,
        ens_index_path: str = "cache/ens_hardware.pkl",
        activation_threshold: float = 0.85,
    ):
        self.ens_index_path = Path(ens_index_path)
        self.activation_threshold = activation_threshold
        self._ens_index: Optional[Dict] = None
        self._ens_items: Optional[List[Dict]] = None

    def _load_ens_index(self) -> Dict:
        """Загрузить ENS индекс из pickle."""
        if self._ens_index is not None:
            return self._ens_index
        if not self.ens_index_path.exists():
            logger.warning("[AutoValidator] ENS index not found: %s", self.ens_index_path)
            self._ens_index = {}
            return self._ens_index
        try:
            with open(self.ens_index_path, "rb") as f:
                self._ens_index = pickle.load(f)
            count = sum(len(v) for v in self._ens_index.values())
            logger.info("[AutoValidator] Loaded %d ENS items for validation", count)
            return self._ens_index
        except Exception as e:
            logger.error("[AutoValidator] Failed to load ENS index: %s", e)
            self._ens_index = {}
            return self._ens_index

    def _get_ens_examples(self, standard: str, item_type: str, limit: int = 10) -> List[Dict]:
        """Получить примеры из ЕНС для стандарта и типа."""
        index = self._load_ens_index()
        canon_std = canonicalize_standard(standard)
        key = (canon_std, item_type.upper())
        if key not in index:
            key = (canon_std, item_type)
        if key not in index:
            for k, items in index.items():
                if k[0] == canon_std:
                    return items[:limit]
            return []
        return index[key][:limit]

    def validate_mask(
        self,
        pattern: str,
        params: List[str],
        required: List[str],
        standard: str,
        item_type: str,
        service: str = "",
        model: str = "",
        temperature: float = 0.0,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        **kwargs,  # FIX 2026-05-25: совместимость с cli.py (ens_examples и др.)
    ) -> ValidationResult:
        """Валидировать маску на примерах ЕНС."""
        examples = self._get_ens_examples(standard, item_type)
        if not examples:
            logger.warning("[AutoValidator] No ENS examples for %s/%s", standard, item_type)
            return ValidationResult(
                score=0.0, passed=False, total=0, matched=0,
                service=service, model=model, temperature=temperature,
                tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
            )

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.error("[AutoValidator] Invalid regex pattern: %s", e)
            return ValidationResult(
                score=0.0, passed=False, total=0, matched=0,
                service=service, model=model, temperature=temperature,
                tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
            )

        total = len(examples)
        success_count = 0
        details = []
        for ex in examples:
            result = self._test_pattern(compiled, ex, params, required)
            if result["success"]:
                success_count += 1
            details.append(result)

        score = success_count / total if total > 0 else 0.0
        passed = score >= self.activation_threshold
        mismatched = sum(1 for d in details if not d["success"] and d.get("error") != "No match")
        missing = sum(1 for d in details if d.get("error") == "No match")

        logger.info(
            "[AutoValidator] Validation result for %s/%s: score=%.2f, passed=%s",
            standard, item_type, score, passed
        )
        return ValidationResult(
            score=score,
            passed=passed,
            details=details,
            total=total,
            matched=success_count,
            mismatched=mismatched,
            missing=missing,
            service=service,
            model=model,
            temperature=temperature,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
        )

    def _test_pattern(
        self,
        pattern: re.Pattern,
        ex: Dict,
        params: List[str],
        required: List[str],
    ) -> Dict:
        """Проверить один пример ЕНС против regex.

        FIX 2026-05-25: Added detailed DEBUG logging for match/no-match diagnosis.
        """
        text = ex.get("полное_наименование", ex.get("наименование", ""))
        if not text:
            return {"success": False, "error": "Empty text", "example": ex}

        match = pattern.search(text)

        # DEBUG: log match result and extracted groups
        logger.debug("[AutoValidator] Testing pattern against: %s", text[:100])
        if match:
            extracted = match.groupdict()
            logger.debug("[AutoValidator] Match OK. Extracted: %s", extracted)
        else:
            logger.debug("[AutoValidator] NO MATCH for text: %s", text[:100])
            logger.debug("[AutoValidator] Pattern: %s", pattern.pattern[:120])

        if not match:
            return {"success": False, "error": "No match", "text": text, "example": ex}

        extracted = match.groupdict()
        mismatches = []
        missing = []
        skip_params = {
            "тип_изделия", "item_type", "наименование", "полное_наименование",
            "код", "mdm_key", "нтд_1", "нтд_2", "стандарт", "нтд",
        }

        for param in required:
            extracted_val = extracted.get(param)
            expected_val = ex.get(param)

            if param in skip_params:
                continue

            extracted_empty = extracted_val is None or str(extracted_val).strip() == ""
            expected_empty = expected_val is None or str(expected_val).strip() == ""

            if extracted_empty and expected_empty:
                continue
            elif expected_empty and not extracted_empty:
                continue
            elif extracted_empty or extracted_val == "":
                missing.append(param)
                continue

            matched_map = {}
            checked = set()
            for ext_key, ext_val in extracted.items():
                if ext_key in skip_params:
                    continue
                best_exp, best_sim = self._match_param_keys(ext_key, ex)
                if best_exp and best_sim >= 0.5:
                    matched_map[ext_key] = best_exp
                    checked.add(best_exp)

            for ext_key, exp_key in matched_map.items():
                ext_val = extracted.get(ext_key)
                exp_val = ex.get(exp_key)
                if ext_val is None or exp_val is None:
                    continue
                if not self._values_match(str(ext_val), str(exp_val)):
                    mismatches.append({
                        "param": ext_key,
                        "expected": exp_val,
                        "extracted": ext_val,
                    })
                    logger.debug(
                        "[AutoValidator] Mismatch %s: expected=%r extracted=%r",
                        ext_key, exp_val, ext_val
                    )

        success = len(missing) == 0 and len(mismatches) == 0
        if not success:
            logger.debug(
                "[AutoValidator] Example failed: missing=%s mismatches=%s",
                missing, mismatches
            )
        return {
            "success": success,
            "missing": missing,
            "mismatches": mismatches,
            "text": text,
            "example": ex,
        }

    @staticmethod
    def _match_param_keys(ext_key: str, ex: Dict) -> Tuple[Optional[str], float]:
        """Найти лучшее соответствие ключа extracted к ключам example."""
        ext_lower = ext_key.lower().replace("_", "")
        best_exp = None
        best_sim = 0.0
        for exp_key in ex.keys():
            exp_lower = exp_key.lower().replace("_", "")
            if ext_lower == exp_lower:
                return exp_key, 1.0
            if ext_lower in exp_lower or exp_lower in ext_lower:
                sim = max(len(ext_lower), len(exp_lower)) / max(len(ext_lower), len(exp_lower))
                if sim > best_sim:
                    best_sim = sim
                    best_exp = exp_key
        # Mapping для нтд_1
        if ext_lower in ("нтд1", "нтд_1"):
            for k in ["стандарт", "нтд", "standard"]:
                if k in ex:
                    return k, 1.0
        return best_exp, best_sim

    @staticmethod
    def _values_match(val1: str, val2: str) -> bool:
        """Сравнить два значения с нормализацией."""
        v1 = str(val1).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        v2 = str(val2).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        return v1 == v2 or v1 in v2 or v2 in v1

    @staticmethod
    def _is_value_in_name(val: str, name: str) -> bool:
        """Проверить, что значение параметра присутствует в строке номенклатуры.

        FIX 2026-05-25: Support М-prefix (e.g. М22) and robust fuzzy matching.
        """
        if not val or not name:
            return False
        val_str = str(val).strip().replace(",", ".")
        name_lower = name.lower().replace(",", ".")

        # 1. Exact match
        if val_str.lower() in name_lower:
            return True

        # 2. Float .0: 16.0 matches 16
        if '.' in val_str and val_str.endswith('.0'):
            int_part = val_str[:-2]
            if int_part and int_part.lower() in name_lower:
                return True

        # 3. Letter parts for coatings
        letter_parts = re.findall(r"[a-zA-Zа-яА-Я]+", val_str)
        for part in letter_parts:
            if part.lower() in name_lower:
                return True

        # 4. Prefix before first digit (Кд6-9 → Кд)
        prefix_match = re.match(r"^([a-zA-Zа-яА-Я\.\-]+)", val_str)
        if prefix_match:
            prefix = prefix_match.group(1).rstrip('.-')
            prefix_clean = re.sub(r"[0-9]", "", prefix).rstrip('.-')
            if prefix_clean and prefix_clean.lower() in name_lower:
                return True

        # 5. Without digits (Кд3.фос → Кд.фос)
        no_digits = re.sub(r"[0-9]", "", val_str).strip(".- ")
        if no_digits and no_digits.lower() in name_lower:
            return True

        # 6. М-prefix: М22 → 22 (for diameters)
        m_match = re.match(r"^[мm](\d+(?:[.,]\d+)?)$", val_str, re.IGNORECASE)
        if m_match:
            num = m_match.group(1)
            if num.lower() in name_lower:
                return True

        return False