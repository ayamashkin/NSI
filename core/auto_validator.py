# =============================================================================
# FILE: core/auto_validator.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 CHANGES (UTC+3):
# 2026-05-27 21:15:00 — _build_skip_params now uses DomainConfig.meta_regex_groups
# 2026-05-27 21:15:00 — Removed hardcoded нтд_1/тип_изделия from fallback skip set
# 2026-05-27 17:46:00 — Removed hardcoded skip_params, now reads from DomainConfig
# 2026-05-27 17:46:00 — _build_skip_params builds from meta_fields + retain_fields + skip_fields
# 2026-05-27 17:46:00 — Added _norm_field_name for field name normalization
# =============================================================================
"""
Auto Validator Module (Domain-based)
Validates generated masks against ENS examples from structured domain index.
"""
import glob
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
    """Mask validator on ENS examples with domain support."""

    def __init__(
        self,
        ens_index_path: str = "cache/ens_hardware.pkl",
        activation_threshold: float = 0.85,
        domain: Optional[str] = None,
    ):
        self.ens_index_path = Path(ens_index_path)
        self.activation_threshold = activation_threshold
        self.domain = domain
        self._domain_index: Optional[Dict] = None
        self._all_domain_indices: Optional[Dict[str, Dict]] = None
        self._skip_params = self._build_skip_params()

    @staticmethod
    def _norm_field_name(name: str) -> str:
        """Normalize field name for comparison."""
        return re.sub(r"[^\wа-яА-Я]", "", str(name).lower().strip())

    def _build_skip_params(self) -> set:
        """Build skip parameters set from domain config.

        Uses meta_fields, retain_fields and skip_fields from domain YAML.
        These fields do not participate in regex parsing (either metadata or service).
        """
        base = {
            "код", "mdm_key", "id",
            "автор_последнего_изменения", "дата_последнего_изменения",
        }
        if not self.domain:
            logger.warning("[AutoValidator] No domain specified, using fallback skip_params")
            return base | {
                "тип_изделия", "item_type", "наименование", "полное_наименование",
                "нтд_1", "нтд_2", "стандарт", "нтд",
                "марка_материала", "марка_материала_1", "толщина_покрытия", "наличие_бп",
            }
        try:
            from core.domain_config import DomainConfig
            cfg = DomainConfig.load(self.domain)
            # meta_fields and retain_fields are not visible in the name (not in regex)
            # skip_fields are already removed from index, but included just in case
            skip = set(cfg.skip_fields) | set(cfg.meta_fields) | set(cfg.retain_fields)
            skip_normalized = {self._norm_field_name(f) for f in skip}
            # Add canonical names that are definitely not extracted by regex
            skip_normalized |= {
                "тип_изделия", "item_type", "наименование", "полное_наименование",
                "нтд_1", "нтд_2", "стандарт", "нтд",
            }
            # Add meta_regex_groups from domain config
            for mg in cfg.meta_regex_groups:
                skip_normalized.add(self._norm_field_name(mg))
            logger.info("[AutoValidator] skip_params built from domain '%s': %d fields",
                        self.domain, len(skip_normalized))
            return base | skip_normalized
        except Exception as e:
            logger.warning("[AutoValidator] Failed to load domain config for skip_params: %s", e)
            return base | {
                "тип_изделия", "item_type", "наименование", "полное_наименование",
                "нтд_1", "нтд_2", "стандарт", "нтд",
                "марка_материала", "марка_материала_1", "толщина_покрытия", "наличие_бп",
            }

    def _load_domain_index(self, path: Optional[str] = None) -> Dict:
        target = Path(path) if path else self.ens_index_path
        try:
            with open(target, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                # Check if this is a structured index (ens_{domain}.pkl)
                # Format: {"ОСТ 1 31133-80": {"Болт": {"examples": [...], ...}}}
                first_std = next(iter(data.values())) if data else None
                if isinstance(first_std, dict):
                    first_type = next(iter(first_std.values())) if first_std else None
                    if isinstance(first_type, dict) and "examples" in first_type:
                        logger.info("[AutoValidator] Loaded structured domain index from %s", target)
                        return data
                # Legacy flat format fallback
                logger.info("[AutoValidator] Loaded legacy index from %s", target)
                return self._legacy_load(data)
        except Exception as e:
            logger.error("[AutoValidator] Failed to load ENS index %s: %s", target, e)
            return {}

    def _legacy_load(self, data: Any) -> Dict:
        """Convert legacy format to structured."""
        index: Dict[str, Dict[str, List[Dict]]] = {}
        items = []
        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            items = data["items"]
        elif isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try to extract lists
            for k, v in data.items():
                if isinstance(v, list):
                    items.extend(v)
        for item in items:
            std = canonicalize_standard(str(item.get("стандарт", item.get("нтд", ""))))
            itype = str(item.get("наименование_типа", item.get("тип_изделия", item.get("тип", "")))).strip()
            if not std or not itype:
                continue
            if std not in index:
                index[std] = {}
            if itype not in index[std]:
                index[std][itype] = []
            index[std][itype].append(item)
        return index

    def _load_all_domain_indices(self, cache_dir: str = "cache") -> Dict[str, Dict]:
        """Load all ens_*.pkl from cache_dir."""
        if self._all_domain_indices is not None:
            return self._all_domain_indices
        self._all_domain_indices = {}
        pattern = str(Path(cache_dir) / "ens_*.pkl")
        for p in glob.glob(pattern):
            domain_name = Path(p).stem.replace("ens_", "")
            try:
                self._all_domain_indices[domain_name] = self._load_domain_index(p)
                logger.info("[AutoValidator] Loaded domain index: %s -> %s", domain_name, p)
            except Exception as e:
                logger.warning("[AutoValidator] Failed to load %s: %s", p, e)
        return self._all_domain_indices

    def _get_ens_examples(self, standard: str, item_type: str, domain: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get examples from domain index."""
        dom = domain or self.domain
        if dom:
            index = self._load_domain_index(self.ens_index_path)
        else:
            # If domain not specified — try current index
            index = self._load_domain_index()

        canon_std = canonicalize_standard(standard)
        itype = item_type.strip()

        def _extract_from_index(idx: Dict) -> List[Dict]:
            if canon_std in idx and itype in idx[canon_std]:
                entry = idx[canon_std][itype]
                examples = entry.get("examples", [])
                return examples[:limit]
            # fuzzy fallback
            for s in idx:
                if canon_std in s or s in canon_std:
                    for t in idx[s]:
                        if itype.lower() == t.lower():
                            return idx[s][t].get("examples", [])[:limit]
            return []

        result = _extract_from_index(index)
        if result:
            return result

        # Multi-domain fallback
        if not dom:
            all_indices = self._load_all_domain_indices()
            for dname, idx in all_indices.items():
                result = _extract_from_index(idx)
                if result:
                    logger.info("[AutoValidator] Found examples in domain '%s' for %s/%s", dname, standard, item_type)
                    return result

        # Legacy fallback
        logger.warning("[AutoValidator] No ENS examples for %s/%s (domain=%s)", standard, item_type, dom)
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
        **kwargs,
    ) -> ValidationResult:
        examples = self._get_ens_examples(standard, item_type)
        if not examples:
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
        logger.info("[AutoValidator] Validation result for %s/%s: score=%.2f, passed=%s",
                    standard, item_type, score, passed)
        if not passed and logger.isEnabledFor(logging.DEBUG):
            failed = [d for d in details if not d["success"]]
            logger.debug("[AutoValidator] Failed examples (%d/%d):", len(failed), total)
            for fd in failed[:5]:
                err = fd.get("error", "mismatch")
                txt = fd.get("text", "")[:60]
                logger.debug("[AutoValidator]   FAIL: %s — %s", err, txt)
                if "missing" in fd and fd["missing"]:
                    logger.debug("[AutoValidator]     Missing: %s", fd["missing"])
                if "mismatches" in fd and fd["mismatches"]:
                    for mm in fd["mismatches"]:
                        logger.debug("[AutoValidator]     Mismatch: param=%s expected=%s extracted=%s",
                                     mm.get("param"), mm.get("expected"), mm.get("extracted"))
        return ValidationResult(
            score=score, passed=passed, details=details, total=total,
            matched=success_count, mismatched=mismatched, missing=missing,
            service=service, model=model, temperature=temperature,
            tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
        )

    def _test_pattern(
        self,
        pattern: re.Pattern,
        ex: Dict,
        params: List[str],
        required: List[str],
    ) -> Dict:
        """Check one ENS example against regex."""
        meta = ex.get("_meta", {})
        text = meta.get("full_name", meta.get("name", ""))
        if not text:
            text = ex.get("полное_наименование", ex.get("наименование", ""))
        if not text:
            return {"success": False, "error": "Empty text", "example": ex}

        skip_params = self._skip_params
        match = pattern.search(text)
        logger.debug("[AutoValidator] Testing pattern against: %s", text[:100])
        if match:
            extracted = match.groupdict()
            logger.debug("[AutoValidator] Match OK. Extracted: %s", extracted)
        else:
            logger.debug("[AutoValidator] NO MATCH for text: %s", text[:100])
            logger.debug("[AutoValidator] Pattern used: %s", pattern.pattern)
            # Log expected params from ENS for debugging
            expected_info = []
            for param in required:
                if param in skip_params:
                    continue
                best_exp_key, _ = self._find_expected_key(param, ex)
                expected_val = ex.get(best_exp_key) if best_exp_key else None
                expected_info.append(f"{param}={expected_val}")
            logger.debug("[AutoValidator] Expected params from ENS: %s", ", ".join(expected_info))
            try:
                partial = pattern.match(text)
                if partial:
                    logger.debug("[AutoValidator] Partial match up to: %s", partial.group())
                else:
                    logger.debug("[AutoValidator] No partial match — pattern fails at start")
            except Exception:
                pass
            return {"success": False, "error": "No match", "text": text, "example": ex}

        extracted = match.groupdict()
        mismatches = []
        missing = []
        debug_lines = []

        for param in required:
            if param in skip_params:
                continue
            extracted_val = extracted.get(param)
            best_exp_key, _ = self._find_expected_key(param, ex)
            expected_val = ex.get(best_exp_key) if best_exp_key else None
            extracted_empty = extracted_val is None or str(extracted_val).strip() == ""
            expected_empty = expected_val is None or str(expected_val).strip() == ""
            if extracted_empty and expected_empty:
                debug_lines.append(f"  {param}: OK (both empty)")
                continue
            elif expected_empty and not extracted_empty:
                debug_lines.append(f"  {param}: OK (expected empty, extracted={extracted_val})")
                continue
            elif extracted_empty or extracted_val == "":
                missing.append(param)
                debug_lines.append(f"  {param}: MISSING (expected={expected_val})")
                continue
            if not self._values_match(str(extracted_val), str(expected_val)):
                mismatches.append({"param": param, "expected": expected_val, "extracted": extracted_val})
                debug_lines.append(f"  {param}: MISMATCH expected={expected_val} extracted={extracted_val}")
            else:
                debug_lines.append(f"  {param}: OK expected={expected_val} extracted={extracted_val}")

        optional_params = set(params) - set(required) - skip_params
        for param in optional_params:
            extracted_val = extracted.get(param)
            best_exp_key, _ = self._find_expected_key(param, ex)
            expected_val = ex.get(best_exp_key) if best_exp_key else None
            extracted_empty = extracted_val is None or str(extracted_val).strip() == ""
            expected_empty = expected_val is None or str(expected_val).strip() == ""
            if expected_empty:
                debug_lines.append(f"  {param}: OK (expected empty, optional)")
                continue
            if extracted_empty:
                mismatches.append({"param": param, "expected": expected_val, "extracted": None})
                debug_lines.append(f"  {param}: MISMATCH expected={expected_val} extracted=None (optional)")
                continue
            if not self._values_match(str(extracted_val), str(expected_val)):
                mismatches.append({"param": param, "expected": expected_val, "extracted": extracted_val})
                debug_lines.append(f"  {param}: MISMATCH expected={expected_val} extracted={extracted_val} (optional)")
            else:
                debug_lines.append(f"  {param}: OK expected={expected_val} extracted={extracted_val} (optional)")

        success = len(missing) == 0 and len(mismatches) == 0
        if not success and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[AutoValidator] Validation details for: %s", text[:80])
            for line in debug_lines:
                logger.debug("[AutoValidator]%s", line)
            if missing:
                logger.debug("[AutoValidator] Missing required params: %s", missing)
            if mismatches:
                for mm in mismatches:
                    logger.debug("[AutoValidator] Mismatch param=%s expected=%s extracted=%s",
                                 mm.get("param"), mm.get("expected"), mm.get("extracted"))
        return {
            "success": success,
            "missing": missing,
            "mismatches": mismatches,
            "text": text,
            "example": ex,
        }

    @staticmethod
    def _find_expected_key(param: str, ex: Dict) -> Tuple[Optional[str], float]:
        param_lower = param.lower().replace("_", "")
        best_key = None
        best_sim = 0.0
        for exp_key in ex.keys():
            if exp_key.startswith("_"):
                continue
            exp_lower = exp_key.lower().replace("_", "")
            if param_lower == exp_lower:
                return exp_key, 1.0
            if param_lower in exp_lower or exp_lower in param_lower:
                sim = min(len(param_lower), len(exp_lower)) / max(len(param_lower), len(exp_lower))
                if sim > best_sim:
                    best_sim = sim
                    best_key = exp_key
        if param_lower in ("нтд1", "нтд_1", "стандарт", "standard"):
            for k in ["стандарт", "нтд", "нтд_1", "standard"]:
                if k in ex:
                    return k, 1.0
        if param_lower in ("типизделия", "тип_изделия", "наименование_типа"):
            for k in ["наименование_типа", "тип_изделия", "тип"]:
                if k in ex:
                    return k, 1.0
        return best_key, best_sim

    @staticmethod
    def _values_match(val1: str, val2: str) -> bool:
        v1_raw = str(val1).strip()
        v2_raw = str(val2).strip()
        v1 = v1_raw.lower().replace(" ", "").replace("-", "").replace("_", "").replace(",", ".")
        v2 = v2_raw.lower().replace(" ", "").replace("-", "").replace("_", "").replace(",", ".")
        if v1 == v2 or v1 in v2 or v2 in v1:
            return True
        try:
            f1 = float(v1)
            f2 = float(v2)
            return abs(f1 - f2) < 0.001
        except (ValueError, TypeError):
            pass
        if "." in v1 and v1.replace(".", "").isdigit():
            if v1.replace(".", "") == v2:
                return True
        if "." in v2 and v2.replace(".", "").isdigit():
            if v2.replace(".", "") == v1:
                return True
        if len(v1) == 2 and v1.isdigit() and len(v2) >= 3 and v2[0].isdigit() and v2[1] == "." and v2[2:].isdigit():
            if v1 == v2.replace(".", ""):
                return True
        if len(v2) == 2 and v2.isdigit() and len(v1) >= 3 and v1[0].isdigit() and v1[1] == "." and v1[2:].isdigit():
            if v2 == v1.replace(".", ""):
                return True
        t1 = set(v1.split("."))
        t2 = set(v2.split("."))
        if t1 and t2:
            intersection = t1 & t2
            if len(intersection) >= max(1, len(t1 | t2) - 2):
                return True
            if len(t1 - t2) <= 1 and len(intersection) >= len(t1) * 0.5:
                return True
            if len(t2 - t1) <= 1 and len(intersection) >= len(t2) * 0.5:
                return True
        cp1 = re.match(r"^([a-zA-Zа-яА-Я]+)", v1)
        cp2 = re.match(r"^([a-zA-Zа-яА-Я]+)", v2)
        if cp1 and cp2:
            if cp1.group(1) == cp2.group(1):
                return True
        return False

    @staticmethod
    def _is_value_in_name(val: str, name: str, param_key: str = "") -> bool:
        if not val or not name:
            return False
        val_raw = str(val).strip()
        val_str = val_raw.lower().replace(",", ".")
        name_lower = name.lower().replace(",", ".")
        if val_str in name_lower:
            return True
        if re.match(r"^\d+[.,]\d+$", val_raw):
            no_sep = re.sub(r"[.,]", "", val_str)
            if no_sep in name_lower:
                return True
        if re.search(r"[a-zA-Zа-яА-Я]", val_str):
            tokens = re.split(r"[.\\-]", val_str)
            tokens = [t for t in tokens if t and re.search(r"[a-zA-Zа-яА-Я]", t)]
            for tok in tokens:
                if tok in name_lower:
                    return True
        prefix = re.match(r"^([a-zA-Zа-яА-Я]+)", val_str)
        if prefix and prefix.group(1) in name_lower:
            return True
        if param_key in ("марка_материала", "марка_материала_1", "материал"):
            return val_str in name_lower
        if "." in val_str and val_str.endswith(".0"):
            int_part = val_str[:-2]
            if int_part and int_part in name_lower:
                return True
        if re.match(r"^\d+[a-zA-Zа-яА-Я]+$", val_str):
            if val_str in name_lower:
                return True
        m_match = re.match(r"^[мm](\d+(?:[.,]\d+)?)$", val_raw, re.IGNORECASE)
        if m_match:
            num = m_match.group(1)
            if num.lower() in name_lower:
                return True
        return False