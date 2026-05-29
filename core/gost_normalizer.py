# =============================================================================
# FILE: core/gost_normalizer.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 CHANGES (UTC+3):
# 2026-05-29 13:30:00 — FEAT: GOST7795Normalizer splits .029→покрытие=02+толщина=9, .46→группа_прочности=4.6
# =============================================================================
"""
Standard-specific value normalizers for parametric extraction.
Converts regex-extracted string fragments into normalized ENS DB format.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GOST7795Normalizer:
    """Normalizer for ГОСТ 7795-70 bolt nomenclature.

    Handles specific encoding formats:
    - .46  → группа_прочности=4.6  (divide by 10)
    - .029 → покрытие=02 + толщина_покрытия=9  (2-digit code + 1-digit thickness)
    - 3    → исполнение=3.0  (add .0)
    """

    @staticmethod
    def normalize(extracted: Dict[str, str], standard: str) -> Dict[str, str]:
        """Normalize extracted values to ENS DB format.

        Args:
            extracted: dict from regex match.groupdict()
            standard: canonical standard name

        Returns:
            Normalized dict with additional ENS fields.
        """
        if "7795-70" not in standard:
            return extracted

        result = dict(extracted)

        # 1. свойства=46 → группа_прочности=4.6
        if "свойства" in result:
            raw = result.pop("свойства", "")
            try:
                val = float(raw)
                result["группа_прочности"] = f"{val / 10:.1f}"
                logger.debug("[GOST7795] Normalized свойства=%s → группа_прочности=%s", raw, result["группа_прочности"])
            except (ValueError, TypeError):
                result["группа_прочности"] = raw
                logger.warning("[GOST7795] Failed to normalize свойства=%s", raw)

        # 2. покрытие=029 → покрытие=02 + толщина_покрытия=9
        if "покрытие" in result:
            raw = result["покрытие"]
            if len(raw) == 3 and raw.isdigit():
                result["покрытие"] = raw[:2]
                result["толщина_покрытия"] = raw[2]
                logger.debug("[GOST7795] Split покрытие=%s → покрытие=%s, толщина=%s", raw, raw[:2], raw[2])
            elif len(raw) == 2 and raw.isdigit():
                # Already 2-digit code (e.g., "02"), thickness might be separate or absent
                pass

        # 3. исполнение=3 → исполнение=3.0
        if "исполнение" in result:
            raw = result["исполнение"]
            try:
                val = float(raw)
                result["исполнение"] = f"{val:.1f}"
            except (ValueError, TypeError):
                pass

        return result

    @staticmethod
    def denormalize(expected: Dict[str, str], standard: str) -> Dict[str, str]:
        """Convert ENS DB values to regex-extracted format for comparison.

        Inverse of normalize() — used when we want to compare
        extracted values against DB without modifying extracted.
        """
        if "7795-70" not in standard:
            return expected

        result = dict(expected)

        # группа_прочности=4.6 → свойства=46
        if "группа_прочности" in result:
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