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
        """Загрузить ENS индекс из pickle с поддержкой разных форматов.

        FIX 2026-05-25: robust loader — поддерживает tuple (index, vectorizer),
        dict с metadata keys, и прямые dict of lists.
        """
        if self._ens_index is not None:
            return self._ens_index
        if not self.ens_index_path.exists():
            logger.warning("[AutoValidator] ENS index not found: %s", self.ens_index_path)
            self._ens_index = {}
            return self._ens_index
        try:
            with open(self.ens_index_path, "rb") as f:
                data = pickle.load(f)

            # Format 0: Dict with 'items' key containing flat list
            if isinstance(data, dict) and 'items' in data and isinstance(data['items'], list):
                items = data['items']
                # Build index from flat list: group by (standard, item_type)
                index = {}
                for item in items:
                    std = item.get('стандарт', item.get('нтд_1', item.get('нтд', '')))
                    itype = item.get('наименование_типа', item.get('тип_изделия', item.get('тип', '')))
                    if not std or not itype:
                        continue
                    std = canonicalize_standard(str(std))
                    itype = str(itype).strip().upper()
                    key = (std, itype)
                    if key not in index:
                        index[key] = []
                    index[key].append(item)
                self._ens_index = index
                count = sum(len(v) for v in index.values())
                logger.info("[AutoValidator] Loaded %d ENS items into %d groups from 'items' key", count, len(index))
                return self._ens_index

            # Format 1: Direct dict {(std, type): [items]} — filter only list values
            if isinstance(data, dict):
                list_values = {k: v for k, v in data.items() if isinstance(v, list)}
                if list_values:
                    self._ens_index = list_values
                    count = sum(len(v) for v in list_values.values())
                    skipped = set(data.keys()) - set(list_values.keys())
                    if skipped:
                        logger.info("[AutoValidator] Skipped non-list keys: %s", skipped)
                    logger.info("[AutoValidator] Loaded %d ENS items from %d keys (filtered dict)",
                                count, len(list_values))
                    return self._ens_index

            # Format 2: Tuple (index_dict, vectorizer, ...)
            if isinstance(data, (tuple, list)) and len(data) > 0:
                first = data[0]
                if isinstance(first, dict):
                    self._ens_index = first
                    count = sum(len(v) for v in self._ens_index.values())
                    logger.info("[AutoValidator] Loaded %d ENS items from tuple[0]", count)
                    return self._ens_index

            # Format 3: Dict with metadata keys
            if isinstance(data, dict):
                for key in ['index', 'data', 'ens_index', 'items']:
                    if key in data and isinstance(data[key], dict):
                        subdata = data[key]
                        test_val = next(iter(subdata.values())) if subdata else None
                        if isinstance(test_val, list):
                            self._ens_index = subdata
                            count = sum(len(v) for v in subdata.values())
                            logger.info("[AutoValidator] Loaded %d ENS items from key '%s'", count, key)
                            return self._ens_index

            # Format 4: Filter dict — оставить только list-значения
            if isinstance(data, dict):
                filtered = {k: v for k, v in data.items() if isinstance(v, list)}
                if filtered:
                    self._ens_index = filtered
                    count = sum(len(v) for v in filtered.values())
                    logger.info("[AutoValidator] Loaded %d ENS items (filtered lists)", count)
                    return self._ens_index

            logger.error("[AutoValidator] Unknown pickle format: %s", type(data).__name__)
            self._ens_index = {}
            return self._ens_index

        except Exception as e:
            logger.error("[AutoValidator] Failed to load ENS index: %s", e)
            self._ens_index = {}
            return self._ens_index

    def _get_ens_examples(self, standard: str, item_type: str, limit: int = 10) -> List[Dict]:
        """Получить примеры из ЕНС для стандарта и типа.

        FIX 2026-05-25: добавлено fuzzy matching по стандарту (ОСТ1 → ОСТ 1).
        """
        index = self._load_ens_index()
        canon_std = canonicalize_standard(standard)
        itype = item_type.strip().upper()

        # Exact match
        key = (canon_std, itype)
        if key in index:
            return index[key][:limit]

        # Fuzzy: try without space in standard (ОСТ1 34507-80)
        alt_std = canon_std.replace("ОСТ 1", "ОСТ1").replace("ГОСТ ", "ГОСТ")
        key = (alt_std, itype)
        if key in index:
            return index[key][:limit]

        # Fuzzy: try with space in standard (ОСТ1 → ОСТ 1)
        alt_std2 = canon_std.replace("ОСТ1", "ОСТ 1").replace("ОСТ2", "ОСТ 2")
        key = (alt_std2, itype)
        if key in index:
            return index[key][:limit]

        # Partial match by standard only
        for k, items in index.items():
            if k[0] == canon_std or k[0] == alt_std or k[0] == alt_std2:
                return items[:limit]

        # Last resort: any key containing the standard
        for k, items in index.items():
            if canon_std in k[0] or alt_std in k[0]:
                return items[:limit]

        logger.warning("[AutoValidator] No ENS examples for %s/%s (canon=%s, keys=%d)",
                       standard, item_type, canon_std, len(index))
        return []

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

        FIX 2026-05-25:
        1. Added detailed DEBUG logging for match/no-match diagnosis.
        2. FIXED duplicate mismatches: each param checked exactly once.
        3. FIXED _values_match: comma→dot normalization, numeric float compare,
           coating token-based fuzzy match.
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
            if param in skip_params:
                continue

            extracted_val = extracted.get(param)
            # Найти соответствие в example по имени параметра
            best_exp_key, best_sim = self._find_expected_key(param, ex)
            expected_val = ex.get(best_exp_key) if best_exp_key else None

            extracted_empty = extracted_val is None or str(extracted_val).strip() == ""
            expected_empty = expected_val is None or str(expected_val).strip() == ""

            if extracted_empty and expected_empty:
                continue
            elif expected_empty and not extracted_empty:
                continue  # extra extracted is OK
            elif extracted_empty or extracted_val == "":
                missing.append(param)
                logger.debug("[AutoValidator] Missing param %s: expected=%r", param, expected_val)
                continue

            if not self._values_match(str(extracted_val), str(expected_val)):
                mismatches.append({
                    "param": param,
                    "expected": expected_val,
                    "extracted": extracted_val,
                })
                logger.debug(
                    "[AutoValidator] Mismatch %s: expected=%r extracted=%r",
                    param, expected_val, extracted_val
                )

        success = len(missing) == 0 and len(mismatches) == 0
        if not success:
            logger.debug(
                "[AutoValidator] Example failed: missing=%s mismatches=%s",
                missing, [m["param"] for m in mismatches]
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
        """Найти лучшее соответствие ключа extracted к ключам example (legacy)."""
        return AutoValidator._find_expected_key(ext_key, ex)

    @staticmethod
    def _find_expected_key(param: str, ex: Dict) -> Tuple[Optional[str], float]:
        """Найти лучшее соответствие ключа param в ключах example.

        FIX 2026-05-25: renamed from _match_param_keys, searches param→ex keys.
        """
        param_lower = param.lower().replace("_", "")
        best_key = None
        best_sim = 0.0
        for exp_key in ex.keys():
            exp_lower = exp_key.lower().replace("_", "")
            if param_lower == exp_lower:
                return exp_key, 1.0
            if param_lower in exp_lower or exp_lower in param_lower:
                sim = min(len(param_lower), len(exp_lower)) / max(len(param_lower), len(exp_lower))
                if sim > best_sim:
                    best_sim = sim
                    best_key = exp_key
        # Mapping для нтд_1
        if param_lower in ("нтд1", "нтд_1", "стандарт", "standard"):
            for k in ["стандарт", "нтд", "нтд_1", "standard"]:
                if k in ex:
                    return k, 1.0
        # Mapping для тип_изделия
        if param_lower in ("типизделия", "тип_изделия", "наименование_типа"):
            for k in ["наименование_типа", "тип_изделия", "тип"]:
                if k in ex:
                    return k, 1.0
        return best_key, best_sim

    @staticmethod
    def _values_match(val1: str, val2: str) -> bool:
        """Сравнить два значения с нормализацией.

        FIX 2026-05-25:
        - comma→dot normalization (0,1 ↔ 0.1)
        - numeric float comparison
        - coating token-based fuzzy match (Хим.Фос.прм ⊆ Хим.Фос.хр.прм)
        - coating prefix match (Ц9.хр ≈ Ц)
        - numeric dot-optional (5.8 ≈ 58, точка как разделитель)
        """
        v1_raw = str(val1).strip()
        v2_raw = str(val2).strip()
        v1 = v1_raw.lower().replace(" ", "").replace("-", "").replace("_", "").replace(",", ".")
        v2 = v2_raw.lower().replace(" ", "").replace("-", "").replace("_", "").replace(",", ".")

        # Exact / substring
        if v1 == v2 or v1 in v2 or v2 in v1:
            return True

        # Numeric: try float comparison
        try:
            f1 = float(v1)
            f2 = float(v2)
            return abs(f1 - f2) < 0.001
        except (ValueError, TypeError):
            pass

        # Numeric: dot as optional separator (5.8 ↔ 58)
        if '.' in v1 and v1.replace('.', '').isdigit():
            if v1.replace('.', '') == v2:
                return True
        if '.' in v2 and v2.replace('.', '').isdigit():
            if v2.replace('.', '') == v1:
                return True
        # 2-digit ↔ dotted (58 ↔ 5.8)
        if len(v1) == 2 and v1.isdigit() and len(v2) >= 3 and v2[0].isdigit() and v2[1] == '.' and v2[2:].isdigit():
            if v1 == v2.replace('.', ''):
                return True
        if len(v2) == 2 and v2.isdigit() and len(v1) >= 3 and v1[0].isdigit() and v1[1] == '.' and v1[2:].isdigit():
            if v2 == v1.replace('.', ''):
                return True

        # Coating / composite: token-based fuzzy match
        t1 = set(v1.split("."))
        t2 = set(v2.split("."))
        if t1 and t2:
            intersection = t1 & t2
            # If most tokens match (allow 1-2 missing)
            if len(intersection) >= max(1, len(t1 | t2) - 2):
                return True
            # If extracted is mostly contained in expected
            if len(t1 - t2) <= 1 and len(intersection) >= len(t1) * 0.5:
                return True
            # If expected is mostly contained in extracted
            if len(t2 - t1) <= 1 and len(intersection) >= len(t2) * 0.5:
                return True

        # Coating prefix match (Ц9.хр ≈ Ц, Кд6-9.хр ≈ Кд)
        cp1 = re.match(r"^([a-zA-Zа-яА-Я]+)", v1)
        cp2 = re.match(r"^([a-zA-Zа-яА-Я]+)", v2)
        if cp1 and cp2:
            if cp1.group(1) == cp2.group(1):
                return True

        return False

    @staticmethod
    def _is_value_in_name(val: str, name: str, param_key: str = "") -> bool:
        """Проверить, что значение параметра присутствует в строке номенклатуры.

        FIX 2026-05-25 v2: синхронизировано с llm_mask_generator.py.
        - Покрытия: token-based fuzzy (Ц9.хр → Ц)
        - Числа с разделителем: 5,8/5.8 ↔ 58
        - Марка материала: только exact match
        """
        if not val or not name:
            return False
        val_raw = str(val).strip()
        val_str = val_raw.lower().replace(",", ".")
        name_lower = name.lower().replace(",", ".")

        # 1. Exact / substring match
        if val_str in name_lower:
            return True

        # 2. Numeric with decimal separator ↔ without (5.8 ↔ 58)
        if re.match(r"^\d+[.,]\d+$", val_raw):
            no_sep = re.sub(r"[.,]", "", val_str)
            if no_sep in name_lower:
                return True

        # 3. Coatings / composite: token-based fuzzy match
        if param_key in ("покрытие", "coating", "покрытие_1") or \
           re.search(r"[a-zA-Zа-яА-Я]", val_str):
            tokens = re.split(r"[.\-]", val_str)
            tokens = [t for t in tokens if t and re.search(r"[a-zA-Zа-яА-Я]", t)]
            for tok in tokens:
                if tok in name_lower:
                    return True
            prefix = re.match(r"^([a-zA-Zа-яА-Я]+)", val_str)
            if prefix and prefix.group(1) in name_lower:
                return True

        # 4. Material grade: strict exact match only
        if param_key in ("марка_материала", "марка_материала_1", "материал"):
            return val_str in name_lower

        # 5. Float .0: 16.0 matches 16
        if '.' in val_str and val_str.endswith('.0'):
            int_part = val_str[:-2]
            if int_part and int_part in name_lower:
                return True

        # 6. General: tolerance classes etc.
        if re.match(r"^\d+[a-zA-Zа-яА-Я]+$", val_str):
            if val_str in name_lower:
                return True

        # 7. М-prefix: М22 → 22
        m_match = re.match(r"^[мm](\d+(?:[.,]\d+)?)$", val_raw, re.IGNORECASE)
        if m_match:
            num = m_match.group(1)
            if num.lower() in name_lower:
                return True

        return False