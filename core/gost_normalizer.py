# =============================================================================
# FILE: core/gost_normalizer.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 CHANGES (UTC+3):
# 2026-06-03 12:00:00 — FEAT: universal normalizer for ALL standards (GOST/DIN/ISO/NAS).
#   Added: composite группа_прочности.толщина_покрытия split,
#   comma→dot normalization, DIN/ISO coating formats (A2J, Ц6Х).
# 2026-05-29 13:30:00 — FEAT: GOST7795Normalizer splits .029→покрытие=02+толщина=9, .46→группа_прочности=4.6
# =============================================================================
"""
Standard-specific value normalizers for parametric extraction.
Converts regex-extracted string fragments into normalized ENS DB format.

Universal — works for GOST, DIN, ISO, NAS, and internal codes.
"""
import logging
import re
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class GOST7795Normalizer:
    """Universal normalizer for parametric extraction from ALL standards.

    Handles formats found in GOST, DIN, ISO, NAS nomenclature:
    - Группа прочности: 8,8 → 8.8; 8.8 → split into группа_прочности=8 + толщина_покрытия=8
    - Покрытие composite: 029 → покрытие=02 + толщина_покрытия=9
    - DIN/ISO coating: A2J → покрытие=A2 + толщина_покрытия=J; Ц6Х → покрытие=Ц + толщина=6Х
    - Properties: .46 → группа_прочности=4.6 (divide by 10 for GOST 7795-70)
    - Execution: 3 → 3.0
    """

    # Standards where свойства means группа_прочности * 10 (legacy GOST 7795-70 behavior)
    _PROPERTIES_DIV10_STANDARDS = {"7795-70", "7796-70", "7805-70", "7808-70"}

    # Composite coating patterns: буквы+цифры (DIN/ISO style)
    _COATING_LETTER_DIGIT_RE = re.compile(r'^([A-Za-zА-Яа-я]+)(\d+.*)$')

    @classmethod
    def normalize(cls, extracted: Dict[str, str], standard: str) -> Dict[str, str]:
        """Normalize extracted values to ENS DB format.

        Args:
            extracted: dict from regex match.groupdict()
            standard: canonical standard name

        Returns:
            Normalized dict with additional ENS fields.
        """
        result = dict(extracted)
        std_clean = cls._clean_standard(standard)

        # 1. Universal: normalize commas to dots in numeric fields
        cls._normalize_commas(result)

        # 2. Universal: split composite группа_прочности.толщина_покрытия
        cls._split_composite_strength(result)

        # 3. Universal: split composite покрытие (3-digit code, letter+digit, etc.)
        cls._split_composite_coating(result)

        # 4. Legacy: свойства=46 → группа_прочности=4.6 (only for specific GOSTs)
        cls._normalize_properties_legacy(result, std_clean)

        # 5. Universal: normalize исполнение
        cls._normalize_execution(result)

        return result

    @classmethod
    def denormalize(cls, expected: Dict[str, str], standard: str) -> Dict[str, str]:
        """Convert ENS DB values to regex-extracted format for comparison.

        Inverse of normalize() — used when comparing extracted values against DB.
        """
        result = dict(expected)
        std_clean = cls._clean_standard(standard)

        # группа_прочности=4.6 → свойства=46 (only for legacy standards)
        if "группа_прочности" in result and std_clean in cls._PROPERTIES_DIV10_STANDARDS:
            raw = result.pop("группа_прочности", "")
            try:
                val = float(raw)
                result["свойства"] = f"{int(val * 10):d}"
            except (ValueError, TypeError):
                result["свойства"] = raw

        # покрытие=02 + толщина_покрытия=9 → покрытие=029
        if "покрытие" in result and "толщина_покрытия" in result:
            pc = result.pop("покрытие")
            th = result.pop("толщина_покрытия")
            # If both are digits, merge; otherwise keep separate
            if pc and th and pc.isdigit() and th.isdigit():
                result["покрытие"] = f"{pc}{th}"
            else:
                result["покрытие"] = f"{pc}{th}"

        # исполнение=3.0 → 3
        if "исполнение" in result:
            raw = result["исполнение"]
            try:
                val = float(raw)
                if val == int(val):
                    result["исполнение"] = str(int(val))
            except (ValueError, TypeError):
                pass

        return result

    # ------------------------------------------------------------------
    # Universal normalizers (applied to ALL standards)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_commas(result: Dict[str, str]) -> None:
        """Convert commas to dots in all numeric-looking fields."""
        numeric_fields = {
            "группа_прочности", "толщина_покрытия", "исполнение",
            "номинальный_диаметр_резьбы", "шаг_резьбы", "длина",
            "толщина", "диаметр", "параметры", "свойства",
        }
        for field in numeric_fields:
            if field in result and result[field]:
                val = str(result[field])
                if "," in val:
                    result[field] = val.replace(",", ".")
                    logger.debug("[Normalizer] comma→dot: %s=%s", field, result[field])

    @classmethod
    def _split_composite_strength(cls, result: Dict[str, str]) -> None:
        """Split группа_прочности=8.8 into группа_прочности=8 + толщина_покрытия=8.

        Handles: 8.8, 8,8, 10.9, 12.8, etc.
        Only splits if толщина_покрытия is NOT already present.
        """
        gp = result.get("группа_прочности")
        if not gp:
            return
        # Already normalized commas → dots
        gp_str = str(gp).strip()
        # Check if it looks like a composite: digits.digits
        if "." in gp_str and gp_str.replace(".", "").isdigit():
            parts = gp_str.split(".", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                # Only split if толщина_покрытия not already set
                if not result.get("толщина_покрытия"):
                    result["группа_прочности"] = parts[0]
                    result["толщина_покрытия"] = parts[1]
                    logger.debug("[Normalizer] Split composite strength: %s → ГП=%s, ТП=%s",
                                 gp_str, parts[0], parts[1])

    @classmethod
    def _split_composite_coating(cls, result: Dict[str, str]) -> None:
        """Handle composite coating codes:
        - 029 → покрытие=02 + толщина_покрытия=9 (3-digit numeric)
        - Ц6Х → покрытие=Ц + толщина_покрытия=6Х (Cyrillic+digits, Russian standards)
        - A2J, A4-70 → LEFT AS-IS (DIN/ISO, do NOT split)
        - 8.8 → already handled by _split_composite_strength
        """
        coating = result.get("покрытие")
        if not coating:
            return
        raw = str(coating).strip()

        # Pattern 1: 3-digit numeric (e.g., 029)
        if len(raw) == 3 and raw.isdigit():
            result["покрытие"] = raw[:2]
            result["толщина_покрытия"] = raw[2]
            logger.debug("[Normalizer] Split 3-digit coating: %s → покрытие=%s, толщина=%s",
                         raw, raw[:2], raw[2])
            return

        # Pattern 2: letter prefix + digits (Cyrillic ONLY — Russian standards)
        # DIN/ISO codes (A2J, A4-70) are LEFT AS-IS
        m = cls._COATING_LETTER_DIGIT_RE.match(raw)
        if m:
            letter_part = m.group(1)
            digit_part = m.group(2)
            # Only split Cyrillic composites (Ц6Х, etc.)
            # Do NOT split Latin-only DIN/ISO codes (A2J, A4, etc.)
            has_cyrillic = bool(re.search(r"[А-Яа-я]", letter_part))
            is_simple = raw.lower() in ("кд", "кд.", "ан", "ц.", "ц")
            if has_cyrillic and not is_simple and len(digit_part) >= 1:
                result["покрытие"] = letter_part
                result["толщина_покрытия"] = digit_part
                logger.debug("[Normalizer] Split Cyrillic coating: %s → покрытие=%s, толщина=%s",
                             raw, letter_part, digit_part)
                return
            # Latin DIN/ISO: leave as-is (A2J, A4-70, etc.)
            logger.debug("[Normalizer] DIN/ISO coating kept as-is: %s", raw)

    @classmethod
    def _normalize_properties_legacy(cls, result: Dict[str, str], std_clean: str) -> None:
        """Legacy GOST 7795-70: свойства=46 → группа_прочности=4.6."""
        if std_clean not in cls._PROPERTIES_DIV10_STANDARDS:
            return
        if "свойства" not in result:
            return
        raw = result.pop("свойства", "")
        try:
            val = float(raw)
            result["группа_прочности"] = f"{val / 10:.1f}"
            logger.debug("[Normalizer] Legacy: свойства=%s → группа_прочности=%s", raw, result["группа_прочности"])
        except (ValueError, TypeError):
            result["группа_прочности"] = raw
            logger.warning("[Normalizer] Failed to normalize свойства=%s", raw)

    @staticmethod
    def _normalize_execution(result: Dict[str, str]) -> None:
        """исполнение=3 → исполнение=3.0"""
        if "исполнение" not in result:
            return
        raw = result["исполнение"]
        try:
            val = float(raw)
            result["исполнение"] = f"{val:.1f}"
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_standard(standard: str) -> str:
        """Extract clean standard code for matching."""
        if not standard:
            return ""
        # Remove prefix (GOST, OST, DIN, etc.) and spaces
        s = standard.upper()
        for prefix in ("ГОСТ", "GOST", "ОСТ", "OST", "DIN", "NAS", "MS", "AN", "ISO"):
            s = s.replace(prefix, "")
        return s.strip().replace(" ", "")


# =============================================================================
# NEW: Composite parameter utilities for fuzzy matching
# =============================================================================

class CompositeParamMatcher:
    """Utilities for matching composite parameters between extracted and ENS values.

    Handles cases where one side has composite value (e.g., '8.8') and other
    has split fields (группа_прочности=8, толщина_покрытия=8).
    """

    @staticmethod
    def try_match_composite(extracted_val: str, ens_val: str) -> bool:
        """Try to match values that may be in different formats.

        Examples:
        - extracted '8.8' vs ens '8' → match (group only)
        - extracted '8.8' vs ens '8.8' → exact match
        - extracted 'A2' vs ens 'A2J' → match (prefix)
        - extracted '8' vs ens '8.8' → match (group matches)
        """
        if not extracted_val or not ens_val:
            return False

        e = str(extracted_val).strip().replace(",", ".")
        v = str(ens_val).strip().replace(",", ".")

        if e == v:
            return True

        # Composite: e='8.8', v='8' → match first part
        if "." in e:
            e_parts = e.split(".", 1)
            if e_parts[0] == v:
                return True
            # e='8', v='8.8' → match first part
            if "." in v:
                v_parts = v.split(".", 1)
                if e_parts[0] == v_parts[0]:
                    return True

        # Prefix match for coatings: e='A2', v='A2J'
        if len(e) < len(v) and v.startswith(e):
            return True
        if len(v) < len(e) and e.startswith(v):
            return True

        return False