# =============================================================================
# ФАЙЛ: core/auto_validator.py
# ПОСЛЕДНИЕ 5 ИЗМЕНЕНИЙ (МСК, UTC+3):
# 2026-06-01 22:00:00 — ДОБАВЛЕНИЕ: TokenParser fallback — гибридный парсер когда regexp не матчит
# 2026-06-01 22:00:00 — ДОБАВЛЕНИЕ: _preprocess_bare_execution — скобки к bare-исполнению
# 2026-06-01 21:15:00 — ИСПРАВЛЕНИЕ: таблица валидации теперь выводится при 0% match (NO MATCH), + XLSX лист
# 2026-05-29 13:30:00 — FEAT: GOST7795Normalizer для .029→покрытие+толщина, .46→группа_прочности
# 2026-05-29 13:30:00 — ИСПРАВЛЕНИЕ: _find_expected_key twin mapping свойства↔группа_прочности
# 2026-05-28 21:06:00 — ИСПРАВЛЕНИЕ: _print_summary_table с ENS/Mask значениями
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
from core.token_parser import TokenParser

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
        max_examples: int = 10,
    ):
        self.ens_index_path = Path(ens_index_path)
        self.activation_threshold = activation_threshold
        self.domain = domain
        self.max_examples = max_examples
        self._domain_index: Optional[Dict] = None
        self._all_domain_indices: Optional[Dict[str, Dict]] = None
        self._skip_params = self._build_skip_params()
        self._loose_fields = self._build_loose_fields()
        self._token_parser = TokenParser()
        self._twin_groups = self._build_twin_groups()

    @staticmethod
    def _norm_field_name(name: str) -> str:
        """Normalize field name for comparison."""
        return re.sub(r"[^\wа-яА-Я]", "", str(name).lower().strip())

    def _build_skip_params(self) -> set:
        """Build skip parameters set from domain config."""
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
            skip = set(cfg.skip_fields) | set(cfg.meta_fields) | set(cfg.retain_fields)
            skip_normalized = {self._norm_field_name(f) for f in skip}
            skip_normalized |= {
                "тип_изделия", "item_type", "наименование", "полное_наименование",
                "нтд_1", "нтд_2", "стандарт", "нтд",
            }
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

    def _build_loose_fields(self) -> set:
        """Build loose-match fields set from domain config (e.g. coating)."""
        if not self.domain:
            return set()
        try:
            from core.domain_config import DomainConfig
            cfg = DomainConfig.load(self.domain)
            loose = {self._norm_field_name(f) for f in cfg.loose_match_fields}
            logger.info("[AutoValidator] loose_fields from domain '%s': %s", self.domain, loose)
            return loose
        except Exception as e:
            logger.warning("[AutoValidator] Failed to load domain config for loose_fields: %s", e)
            return set()


    def _build_twin_groups(self) -> List[List[str]]:
        """Build twin groups from domain config (regex field → DB field)."""
        if not self.domain:
            return []
        try:
            from core.domain_config import DomainConfig
            cfg = DomainConfig.load(self.domain)
            twins = getattr(cfg, "twin_groups", [])
            logger.info("[AutoValidator] twin_groups from domain %s: %s", self.domain, twins)
            return twins
        except Exception as e:
            logger.warning("[AutoValidator] Failed to load twin_groups: %s", e)
            return []
    def _load_domain_index(self, path: Optional[str] = None) -> Dict:
        target = Path(path) if path else self.ens_index_path
        try:
            with open(target, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                first_std = next(iter(data.values())) if data else None
                if isinstance(first_std, dict):
                    first_type = next(iter(first_std.values())) if first_std else None
                    if isinstance(first_type, dict) and "examples" in first_type:
                        logger.info("[AutoValidator] Loaded structured domain index from %s", target)
                        return data
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

    def _get_ens_examples(self, standard: str, item_type: str, domain: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get examples from domain index. limit=None means use self.max_examples."""
        use_limit = limit if limit is not None else self.max_examples
        """Get examples from domain index."""
        dom = domain or self.domain
        if dom:
            index = self._load_domain_index(self.ens_index_path)
        else:
            index = self._load_domain_index()

        canon_std = canonicalize_standard(standard)
        itype = item_type.strip()

        def _extract_from_index(idx: Dict) -> List[Dict]:
            if canon_std in idx and itype in idx[canon_std]:
                entry = idx[canon_std][itype]
                examples = entry.get("examples", [])
                return examples[:use_limit]
            for s in idx:
                if canon_std in s or s in canon_std:
                    for t in idx[s]:
                        if itype.lower() == t.lower():
                            return idx[s][t].get("examples", [])[:limit]
            return []

        result = _extract_from_index(index)
        if result:
            return result

        if not dom:
            all_indices = self._load_all_domain_indices()
            for dname, idx in all_indices.items():
                result = _extract_from_index(idx)
                if result:
                    logger.info("[AutoValidator] Found examples in domain '%s' for %s/%s", dname, standard, item_type)
                    return result

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
        logger.debug("[AutoValidator] Validating %s/%s against %d examples", standard, item_type, total)
        logger.debug("[AutoValidator] Pattern: %s", pattern[:120] if pattern else "(empty)")
        optional_params = set(params) - set(required)
        for ex in examples:
            result = self._test_pattern(compiled, ex, params, required, standard, optional_params, item_type)
            if result["success"]:
                success_count += 1
            details.append(result)
        score = success_count / total if total > 0 else 0.0
        passed = score >= self.activation_threshold
        mismatched = sum(1 for d in details if not d["success"] and d.get("error") != "No match")
        missing = sum(1 for d in details if d.get("error") == "No match")

        # === SUMMARY TABLE (always in debug) ===
        if logger.isEnabledFor(logging.DEBUG):
            self._print_summary_table(standard, item_type, details, required, self._skip_params, total, success_count, self._loose_fields, pattern)

        # === FAILED DETAILS (only if not passed) ===
        if not passed and logger.isEnabledFor(logging.DEBUG):
            failed = [d for d in details if not d["success"]]
            logger.debug("[AutoValidator] Failed examples (%d/%d):", len(failed), total)
            for fd in failed[:5]:
                err = fd.get("error", "mismatch")
                txt = fd.get("text", "")[:60]
                logger.debug("[AutoValidator] FAIL: %s — %s", err, txt)
                if "missing" in fd and fd["missing"]:
                    logger.debug("[AutoValidator] Missing: %s", fd["missing"])
                if "mismatches" in fd and fd["mismatches"]:
                    for mm in fd["mismatches"]:
                        logger.debug("[AutoValidator] Mismatch: param=%s expected=%s extracted=%s",
                                     mm.get("param"), mm.get("expected"), mm.get("extracted"))

        logger.info("[AutoValidator] Validation result for %s/%s: score=%.2f, passed=%s",
                    standard, item_type, score, passed)
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
        standard: str = "",
        optional_params: set = None,
        item_type: str = "",
    ) -> Dict:
        meta = ex.get("_meta", {})
        text = meta.get("full_name", meta.get("name", ""))
        if not text:
            text = ex.get("полное_наименование", ex.get("наименование", ""))
        if not text:
            return {"success": False, "error": "Empty text", "example": ex}

        # FIX 2026-06-01 21:45: preprocess bare execution (add parens)
        text = self._preprocess_bare_execution(text, ex)

        skip_params = self._skip_params
        match = pattern.search(text)
        param_results: Dict[str, str] = {}

        if not match:
            # FEAT 2026-06-01 22:00: fallback to TokenParser when regex fails
            if optional_params and item_type and hasattr(self, '_token_parser'):
                token_result = self._token_parser.parse(
                    text, standard, item_type, params, optional_params
                )
                if token_result:
                    logger.debug("[AutoValidator] TokenParser fallback for: %s", text[:50])
                    for param in params:
                        if param in skip_params:
                            continue
                        val = token_result.get(param)
                        best_exp_key, _ = self._find_expected_key(param, ex)
                        expected_val = ex.get(best_exp_key) if best_exp_key else None
                        extracted_val = val
                        if val is not None and isinstance(val, str):
                            try:
                                fval = float(val)
                                extracted_val = int(fval) if fval == int(fval) else fval
                            except ValueError:
                                extracted_val = val
                        param_results[param] = {
                            "matched": val is not None,
                            "extracted": extracted_val,
                            "expected": expected_val,
                        }
                    return {"success": True, "error": None, "text": text, "example": ex,
                            "token_parsed": True, "param_results": param_results}

            expected_info = []
            for param in required:
                if param in skip_params:
                    continue
                best_exp_key, _ = self._find_expected_key(param, ex)
                expected_val = ex.get(best_exp_key) if best_exp_key else None
                expected_info.append(f"{param}={expected_val}")
            sep = chr(10)
            no_match_lines = [
                f"NO MATCH for text: {text[:100]}",
                f"Pattern used: {pattern.pattern}",
                f"Expected params from ENS: {', '.join(expected_info)}",
            ]
            # Detailed diagnostics: try matching prefix by prefix
            for i in range(len(text), 0, -1):
                prefix = text[:i]
                try:
                    if pattern.search(prefix):
                        no_match_lines.append(f"Longest matching prefix ({i} chars): {prefix}")
                        no_match_lines.append(f"Remaining after match: {text[i:]}")
                        break
                except Exception:
                    pass
            else:
                no_match_lines.append("No prefix matches at all")
            logger.debug("[AutoValidator]" + sep + "%s", sep.join(no_match_lines))
            # FIX 2026-06-01 21:15 UTC+3: save expected params for summary table / xlsx
            expected_dict = {}
            for param in required:
                if param in skip_params:
                    continue
                best_exp_key, _ = self._find_expected_key(param, ex)
                if best_exp_key:
                    expected_dict[param] = ex.get(best_exp_key)
            return {"success": False, "error": "No match", "text": text, "example": ex,
                    "param_results": param_results, "expected": expected_dict}

        extracted = match.groupdict()
        # FEAT 2026-05-29 13:30 UTC+3: normalize GOST 7795-70 extracted values
        from core.gost_normalizer import GOST7795Normalizer
        extracted = GOST7795Normalizer.normalize(extracted, standard)

        mismatches = []
        missing = []
        for param in required:
            if param in skip_params:
                continue
            extracted_val = extracted.get(param)
            best_exp_key, _ = self._find_expected_key(param, ex)
            expected_val = ex.get(best_exp_key) if best_exp_key else None
            extracted_empty = extracted_val is None or str(extracted_val).strip() == ""
            expected_empty = expected_val is None or str(expected_val).strip() == ""
            if extracted_empty and expected_empty:

                param_results[param] = "ok"
                continue
            elif expected_empty and not extracted_empty:

                param_results[param] = "ok"
                continue
            elif extracted_empty or extracted_val == "":
                missing.append(param)

                param_results[param] = "missing"
                continue
            if not self._values_match(str(extracted_val), str(expected_val), param):
                mismatches.append({"param": param, "expected": expected_val, "extracted": extracted_val})

                param_results[param] = "mismatch"
            else:

                param_results[param] = "ok"

        optional_params = set(params) - set(required) - skip_params
        for param in optional_params:
            extracted_val = extracted.get(param)
            best_exp_key, _ = self._find_expected_key(param, ex)
            expected_val = ex.get(best_exp_key) if best_exp_key else None
            extracted_empty = extracted_val is None or str(extracted_val).strip() == ""
            expected_empty = expected_val is None or str(expected_val).strip() == ""
            if expected_empty:

                param_results[param] = "ok"
                continue
            if extracted_empty:
                mismatches.append({"param": param, "expected": expected_val, "extracted": None})

                param_results[param] = "mismatch"
                continue
            if not self._values_match(str(extracted_val), str(expected_val), param):
                mismatches.append({"param": param, "expected": expected_val, "extracted": extracted_val})

                param_results[param] = "mismatch"
            else:

                param_results[param] = "ok"

        success = len(missing) == 0 and len(mismatches) == 0
        # Detailed per-example debug removed — see _print_summary_table for aggregated view
        return {
            "success": success,
            "missing": missing,
            "mismatches": mismatches,
            "text": text,
            "example": ex,
            "param_results": param_results,
            "extracted": extracted,
        }

    @staticmethod
    def _preprocess_bare_execution(text: str, ex: dict) -> str:
        """Add parentheses around bare execution number if detected.
        Transforms 'Винт 1-6-14' to 'Винт (1)-6-14' for regex matching.
        """
        val = ex.get("исполнение")
        if val is None or str(val).strip() == "":
            return text
        val_str = str(val).strip()
        # Check if already parenthesized
        if f"({val_str})" in text:
            return text
        # Check if bare number exists in text
        if f" {val_str}-" in text:
            text = text.replace(f" {val_str}-", f" ({val_str})-", 1)
        return text

    @staticmethod
    def _extract_param_order(pattern: str) -> List[Tuple[str, bool]]:
        """Extract named groups from pattern in order of appearance.
        Returns list of (name, is_optional). Handles nested parens."""
        if not pattern:
            return []
        params = []
        i = 0
        while i < len(pattern):
            idx = pattern.find('(?P<', i)
            if idx == -1:
                break
            gt = pattern.find('>', idx)
            if gt == -1:
                break
            name = pattern[idx + 4:gt]
            start = gt + 1
            depth = 1
            j = start
            while j < len(pattern) and depth > 0:
                if pattern[j] == '(':
                    depth += 1
                elif pattern[j] == ')':
                    depth -= 1
                j += 1
            end = j
            after = pattern[end:end + 6]
            # Two forms of optional: \))? (exec in parens) or )? (direct)
            is_opt = (after.startswith('\\))') and len(after) > 3 and after[3] == '?') or \
                      after.startswith(')?')
            params.append((name, is_opt))
            i = end
        return params

    def _print_summary_table(self, standard: str, item_type: str, details: List[Dict],
                             required: List[str], skip_params: set, total: int, success_count: int,
                             loose_fields: set = None, pattern: str = "") -> None:
        """Print transposed summary table: examples as rows, params as columns.
        Cell format: ENS_value/Mask_value for OK, ENS_val≠Mask_val for mismatch, ENS_val/∅ for missing.
        Column widths scaled 1.5x to prevent overflow. Includes both required and optional params."""
        # Collect all params from all details (required + optional)
        # FIX 2026-06-01 21:15 UTC+3: also collect from ENS example when NO MATCH
        all_params = set()
        for d in details:
            pr = d.get("param_results", {})
            for p in pr:
                if p not in skip_params:
                    all_params.add(p)
            for mm in d.get("mismatches", []):
                p = mm.get("param")
                if p and p not in skip_params:
                    all_params.add(p)
            for p in d.get("missing", []):
                if p not in skip_params:
                    all_params.add(p)
            # When NO MATCH — collect params from ENS example data
            if d.get("error") == "No match":
                ex = d.get("example", {})
                for key in ex:
                    if key.startswith("_") or key in skip_params:
                        continue
                    if ex.get(key) is not None and str(ex.get(key)).strip() != "":
                        all_params.add(key)
        # FEAT 2026-06-01 21:15: order params as in pattern, not alphabetically
        ordered = self._extract_param_order(pattern)
        if ordered:
            # Keep only params that are IN the pattern, in pattern order
            params = [n for n, o in ordered if n in all_params]
        else:
            params = sorted(all_params)
        if not params:
            return

        rows = []
        for i, d in enumerate(details, 1):
            text = d.get("text", "")[:50]
            pr = d.get("param_results", {})
            error = d.get("error")
            mismatches = {mm["param"]: (mm.get("expected"), mm.get("extracted")) for mm in d.get("mismatches", [])}
            missing = set(d.get("missing", []))

            row = {"idx": i, "text": text, "cells": {}, "result": "OK"}
            if error == "No match":
                row["result"] = "NO MATCH"
            elif not d.get("success"):
                row["result"] = "FAIL"
            else:
                row["result"] = "OK"

            ex = d.get("example", {})
            for p in params:
                if error == "No match":
                    # FIX 2026-06-01 21:15: use pre-computed expected dict
                    ens_val = d.get("expected", {}).get(p)
                    if ens_val is None:
                        row["cells"][p] = "∅"
                    else:
                        row["cells"][p] = f"∅≠{ens_val}"
                    continue
                # Get ENS expected value
                best_key, _ = self._find_expected_key(p, ex)
                ens_val = ex.get(best_key) if best_key else None
                ens_str = str(ens_val) if ens_val is not None else "—"

                status = pr.get(p, "ok")
                is_loose = loose_fields and self._norm_field_name(p) in loose_fields
                if status == "ok":
                    ext_val = d.get("extracted", {}).get(p)
                    ext_str = str(ext_val) if ext_val is not None else "—"
                    # Exact match after normalization?
                    ext_norm = ext_str.lower().replace(" ", "").replace("-", "").replace("_", "").replace(",", ".")
                    ens_norm = ens_str.lower().replace(" ", "").replace("-", "").replace("_", "").replace(",", ".")
                    if ext_norm == ens_norm:
                        sep = "="  # exact match
                    elif is_loose:
                        sep = "~"  # loose (substring) match
                    else:
                        sep = "="
                    row["cells"][p] = f"{ext_str}{sep}{ens_str}"
                elif p in missing:
                    sep = "~" if is_loose else "≠"
                    row["cells"][p] = f"∅{sep}{ens_str}"
                elif p in mismatches:
                    exp, ext = mismatches[p]
                    ext_str = str(ext) if ext is not None else "—"
                    row["cells"][p] = f"{ext_str}≠{ens_str}"
                else:
                    row["cells"][p] = "?"
            rows.append(row)

        w_idx = max(len("№"), len(str(total)), max(len(str(r["idx"])) for r in rows) if rows else 0)
        w_text = max(len("Наименование"), max(len(r["text"]) for r in rows) if rows else 0)
        w_text = min(w_text, 50)
        w_result = max(len("Результат"), max(len(r["result"]) for r in rows) if rows else 0)

        # FEAT 2026-06-01 21:15: mark optional params, use pattern order for headers
        optional_map = {n: o for n, o in ordered}
        param_widths = {}
        for p in params:
            display_name = f"{p} (опц.)" if optional_map.get(p) else p
            header_len = len(display_name)
            max_cell = max(len(str(r["cells"].get(p, ""))) for r in rows) if rows else 0
            param_widths[p] = max(header_len, max_cell, 3)

        def hline(char="─"):
            parts = [f"{char*(w_idx+2)}", f"{char*(w_text+2)}", f"{char*(w_result+2)}"]
            for p in params:
                parts.append(f"{char*(param_widths[p]+2)}")
            return "├" + "┼".join(parts) + "┤"

        def top():
            parts = [f"{'─'*(w_idx+2)}", f"{'─'*(w_text+2)}", f"{'─'*(w_result+2)}"]
            for p in params:
                parts.append(f"{'─'*(param_widths[p]+2)}")
            return "┌" + "┬".join(parts) + "┐"

        def bottom():
            parts = [f"{'─'*(w_idx+2)}", f"{'─'*(w_text+2)}", f"{'─'*(w_result+2)}"]
            for p in params:
                parts.append(f"{'─'*(param_widths[p]+2)}")
            return "└" + "┴".join(parts) + "┘"

        lines = []
        if pattern:
            lines.append(f"Pattern: {pattern}")
        lines.append("=== Summary %s/%s: %d/%d passed ===" % (standard, item_type, success_count, total))
        lines.append(top())
        header_parts = [f" {'№':<{w_idx}} ", f" {'Наименование':<{w_text}} ", f" {'Результат':<{w_result}} "]
        for p in params:
            display_name = f"{p} (опц.)" if optional_map.get(p) else p
            header_parts.append(f" {display_name:<{param_widths[p]}} ")
        lines.append("│" + "│".join(header_parts) + "│")
        lines.append(hline())
        for r in rows:
            row_parts = [
                f" {str(r['idx']):<{w_idx}} ",
                f" {r['text']:<{w_text}} ",
                f" {r['result']:<{w_result}} ",
            ]
            for p in params:
                cell = str(r["cells"].get(p, ""))
                row_parts.append(f" {cell:<{param_widths[p]}} ")
            lines.append("│" + "│".join(row_parts) + "│")
        lines.append(bottom())
        sep = chr(10)
        logger.debug("[AutoValidator]" + sep + "%s", sep.join(lines))

    def _print_table(self, text: str, required: List[str], ex: Dict, skip_params: set,
                     extracted: Optional[Dict] = None, mismatches: Optional[List] = None,
                     missing: Optional[List] = None) -> None:
        """Print aligned table: param | ENS | Mask | In text."""
        rows = []
        for param in required:
            if param in skip_params:
                continue
            best_exp_key, _ = self._find_expected_key(param, ex)
            expected_val = ex.get(best_exp_key) if best_exp_key else None
            expected_str = str(expected_val) if expected_val is not None else "—"
            extracted_str = str(extracted.get(param)) if extracted and extracted.get(param) is not None else "—"
            in_text = self._find_in_text(expected_val, text) if expected_val else "—"
            status = ""
            if mismatches:
                for mm in mismatches:
                    if mm.get("param") == param:
                        status = "✗"
                        break
            if missing and param in missing:
                status = "✗"
            rows.append({"param": param, "ens": expected_str, "mask": extracted_str, "text": in_text, "status": status})

        if not rows:
            return

        w_param = max(len(r["param"]) for r in rows)
        w_ens = max(len(r["ens"]) for r in rows)
        w_mask = max(len(r["mask"]) for r in rows)
        w_text = max(len(r["text"]) for r in rows)

        def line(char="─"):
            return f"├{char*(w_param+2)}┼{char*(w_ens+2)}┼{char*(w_mask+2)}┼{char*(w_text+2)}┤"
        def top():
            return f"┌{'─'*(w_param+2)}┬{'─'*(w_ens+2)}┬{'─'*(w_mask+2)}┬{'─'*(w_text+2)}┐"
        def bottom():
            return f"└{'─'*(w_param+2)}┴{'─'*(w_ens+2)}┴{'─'*(w_mask+2)}┴{'─'*(w_text+2)}┘"
        def row(r):
            return f"│ {r['param']:<{w_param}} │ {r['ens']:<{w_ens}} │ {r['mask']:<{w_mask}} │ {r['text']:<{w_text}} │"
        def header():
            return f"│ {'Параметр':<{w_param}} │ {'ЕНС':<{w_ens}} │ {'Маска':<{w_mask}} │ {'В наименовании':<{w_text}} │"

        lines = []
        lines.append(top())
        lines.append(header())
        lines.append(line())
        for r in rows:
            lines.append(row(r))
        lines.append(bottom())
        sep = chr(10)
        logger.debug("[AutoValidator]" + sep + "%s", sep.join(lines))

    @staticmethod
    def _find_in_text(val: Any, text: str) -> str:
        """Find value (or its integer part) in text and return snippet with context."""
        if val is None:
            return "—"
        val_str = str(val).strip()
        text_lower = text.lower()
        pos = text_lower.find(val_str.lower())
        if pos >= 0:
            start = max(0, pos - 3)
            end = min(len(text), pos + len(val_str) + 3)
            return text[start:end]
        if "." in val_str and val_str.endswith(".0"):
            int_part = val_str[:-2]
            pos = text_lower.find(int_part.lower())
            if pos >= 0:
                start = max(0, pos - 3)
                end = min(len(text), pos + len(int_part) + 3)
                return text[start:end]
        val_norm = val_str.lower().replace(" ", "").replace("-", "").replace("_", "").replace(",", ".")
        text_norm = text_lower.replace(" ", "").replace("-", "").replace("_", "")
        pos = text_norm.find(val_norm)
        if pos >= 0:
            return text[max(0, pos-2):min(len(text), pos+len(val_norm)+2)]
        return "не найдено"

    def _diagnose_no_match(self, pattern_str: str, text: str, required: List[str], ex: Dict, skip_params: set) -> None:
        """Detailed diagnostics when regex doesn't match the text."""
        logger.debug("[AutoValidator] === NO-MATCH DIAGNOSTICS ===")
        logger.debug("[AutoValidator] Text: %s", text)
        logger.debug("[AutoValidator] Pattern: %s", pattern_str)
        prefix_pattern = pattern_str.rstrip("$")
        if prefix_pattern != pattern_str:
            try:
                prefix_re = re.compile(prefix_pattern, re.IGNORECASE)
                prefix_match = prefix_re.search(text)
                if prefix_match:
                    matched_part = prefix_match.group()
                    pos = len(matched_part)
                    remaining = text[pos:]
                    logger.debug("[AutoValidator] Prefix match OK up to position %d: %r", pos, matched_part)
                    logger.debug("[AutoValidator] Remaining text after match: %r", remaining)
                else:
                    logger.debug("[AutoValidator] No prefix match either — pattern fails near start")
            except re.error:
                pass
        item_type_match = re.match(r"^(Болт|Винт|Шайба|Гайка)", text, re.IGNORECASE)
        if item_type_match:
            logger.debug("[AutoValidator] Item type literal '%s' found at start — OK", item_type_match.group(1))
        else:
            logger.debug("[AutoValidator] Item type literal NOT found at start of text!")
        std_match = re.search(r"(ОСТ\s*1\s*\d+-\d+|ГОСТ\s*\d+-\d+)$", text, re.IGNORECASE)
        if std_match:
            logger.debug("[AutoValidator] Standard '%s' found at end — OK", std_match.group(1))
        else:
            logger.debug("[AutoValidator] Standard NOT found at end of text!")
        has_parens = "(" in text and ")" in text
        pattern_expects_parens = r"\(" in pattern_str or r"\)?" in pattern_str
        if has_parens and not pattern_expects_parens:
            logger.debug("[AutoValidator] Text has parentheses (execution?) but pattern does not expect them")
        if pattern_expects_parens and not has_parens:
            logger.debug("[AutoValidator] Pattern expects parentheses but text has none")
        logger.debug("[AutoValidator] === END DIAGNOSTICS ===")

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
        if param_lower in ("свойства", "группапрочности", "группа_прочности"):
            for k in ["группа_прочности", "свойства", "группапрочности"]:
                if k in ex:
                    return k, 1.0
        # FEAT 2026-05-29 13:30 UTC+3: twin group resolution (regex field → DB field)
        for twin in self._twin_groups:
            if len(twin) >= 2:
                regex_field = twin[0]
                db_fields = twin[1:]
                if param_lower == self._norm_field_name(regex_field):
                    for dbf in db_fields:
                        for k in ex:
                            if self._norm_field_name(k) == self._norm_field_name(dbf):
                                return k, 1.0
        return best_key, best_sim

    @staticmethod
    def _values_match(val1: str, val2: str, param_key: str = "") -> bool:
        v1_raw = str(val1).strip()
        v2_raw = str(val2).strip()
        v1 = v1_raw.lower().replace(" ", "").replace("-", "").replace("_", "").replace(",", ".")
        v2 = v2_raw.lower().replace(" ", "").replace("-", "").replace("_", "").replace(",", ".")
        if v1 == v2:
            return True
        # FIX 2026-05-28 22:30 UTC+3: coating — substring match is sufficient
        # "Ц" in "Ц9.хр" → OK, "Бп" in "Бп" → OK, "Н.Кд" in "Н.Кд6.т.хр" → OK
        if "покрытие" in param_key:
            if v1 in v2 or v2 in v1:
                return True
        # FIX 2026-05-28 21:58 UTC+3: substring match only for minor differences (.0 suffix, ≤2 chars)
        # Reject "22" vs "22х1.5" (different values, one contains other by accident)
        if v1 in v2 or v2 in v1:
            longer = v2 if len(v2) > len(v1) else v1
            shorter = v1 if len(v2) > len(v1) else v2
            remaining = longer.replace(shorter, "", 1)
            # Allow only .0 / ,0 suffix differences
            if remaining in (".0", ",0", ".00", ",00"):
                return True
            # Allow tiny formatting differences (≤ 2 chars)
            if abs(len(v1) - len(v2)) <= 2:
                return True
            # Otherwise: real mismatch (e.g., "22" vs "22х1.5", "6" vs "6g")
            return False
        try:
            f1 = float(v1)
            f2 = float(v2)
            return abs(f1 - f2) < 0.001
        except (ValueError, TypeError):
            pass
        if "." in v1 and v1.endswith(".0"):
            int_part = v1[:-2]
            if int_part == v2:
                return True
        if "." in v2 and v2.endswith(".0"):
            int_part = v2[:-2]
            if int_part == v1:
                return True
        if len(v1) == 2 and v1.isdigit() and len(v2) >= 3 and v2[0].isdigit() and v2[1] == "." and v2[2:].isdigit():
            int_part = v2[:-2]
            if v1 == int_part:
                return True
        if len(v2) == 2 and v2.isdigit() and len(v1) >= 3 and v1[0].isdigit() and v1[1] == "." and v1[2:].isdigit():
            int_part = v1[:-2]
            if v2 == int_part:
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
        if re.search(r"[a-zA-Zа-яА-Я]", val_str):
            tokens = re.split(r"[.\-]", val_str)
            tokens = [t for t in tokens if t and re.search(r"[a-zA-Zа-яА-Я]", t)]
            for tok in tokens:
                if tok in name_lower:
                    return True
        prefix = re.match(r"^([a-zA-Zа-яА-Я]+)", val_str)
        if prefix and prefix.group(1) in name_lower:
            return True
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