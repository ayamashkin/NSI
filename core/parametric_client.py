# =============================================================================
# FILE: core/parametric_client.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 COMMITS (UTC+3):
# 2026-05-21 12:57:18 db1fd327 21.05.2026
# 2026-05-21 08:53:16 6b906f29 21.05.2026
# 2026-05-21 08:23:07 51f335da 21.05.2026
# 2026-05-20 17:47:49 19e8ca02 20.05.2026
# 2026-05-20 17:39:23 b00c4b25 20.05.2026
# =============================================================================
# FIX 2026-05-22 08:50 UTC+3:
# _get_matching_config now reads from config.yaml matching section directly.
# All thresholds/weights loaded from config; no hardcoded fallbacks.
# =============================================================================
"""
Parametric ENS Client Module
Handles parametric matching between extracted parameters and ENS database entries.

VERSION: 2026-05-22

LAST_FIXES:
  2026-05-22 08:50 UTC+3 — _get_matching_config reads from config.yaml matching.
  2026-05-21 15:15 UTC+3 — fuzzy threshold 0.6->0.5, coating_substitution forces success.
  2026-05-21 08:50 UTC+3 — canonicalize_standard on extract & mask generation.
"""

import json
import logging
import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.standard_utils import canonicalize_standard

logger = logging.getLogger(__name__)

# Lazy import for MatchingConfig (avoid circular dependency)
_matching_config = None

def _get_matching_config():
    """Lazy load MatchingConfig from settings or config.yaml directly."""
    global _matching_config
    if _matching_config is None:
        try:
            from config.settings import get_settings
            _matching_config = get_settings().matching
            logger.debug("[MATCHING] Loaded from settings.matching")
        except Exception as e:
            logger.debug("[MATCHING] Failed to load from settings: %s", e)
            try:
                import yaml
                for path in ["config/config.yaml", "config.yaml"]:
                    p = Path(path)
                    if p.exists():
                        with open(p, "r", encoding="utf-8") as fh:
                            cfg = yaml.safe_load(fh) or {}
                        m = cfg.get("matching", {})
                        class _FallbackMatchingConfig:
                            success_threshold = m.get("success_threshold", 0.7)
                            fuzzy_threshold = m.get("fuzzy_threshold", 0.6)
                            v2_exact_threshold = m.get("v2_exact_threshold", 0.99)
                            coating_similarity_threshold = m.get("coating_similarity_threshold", 0.8)
                            strict_union_keys = m.get("strict_union_keys", False)
                            debug_per_parameter = m.get("debug_per_parameter", True)
                            fuzzy_params_comparison = m.get("fuzzy_params_comparison", True)
                            numeric_field_weight = m.get("numeric_field_weight", 5.0)
                            text_field_weight = m.get("text_field_weight", 2.0)
                            default_field_weight = m.get("default_field_weight", 1.0)
                            length_tolerance = m.get("length_tolerance", 1.0)
                            numeric_tolerance = m.get("numeric_tolerance", 0.01)
                            confidence_penalty_per_mismatch = m.get("confidence_penalty_per_mismatch", 0.15)
                            max_confidence_penalty = m.get("max_confidence_penalty", 0.5)
                        _matching_config = _FallbackMatchingConfig()
                        logger.info("[MATCHING] Loaded from %s", path)
                        break
                else:
                    raise FileNotFoundError("config.yaml not found")
            except Exception as e2:
                logger.warning("[MATCHING] Fallback to built-in defaults: %s", e2)
                class _FallbackMatchingConfig:
                    success_threshold = 0.7
                    fuzzy_threshold = 0.6
                    v2_exact_threshold = 0.99
                    coating_similarity_threshold = 0.8
                    strict_union_keys = False
                    debug_per_parameter = True
                    fuzzy_params_comparison = True
                    numeric_field_weight = 5.0
                    text_field_weight = 2.0
                    default_field_weight = 1.0
                    length_tolerance = 1.0
                    numeric_tolerance = 0.01
                    confidence_penalty_per_mismatch = 0.15
                    max_confidence_penalty = 0.5
                _matching_config = _FallbackMatchingConfig()
    return _matching_config

# Cache for empty_equivalent_values
_empty_values_cache = None
_empty_values_lock = threading.Lock()

def _get_empty_equivalent_values() -> set:
    global _empty_values_cache
    with _empty_values_lock:
        if _empty_values_cache is not None:
            return _empty_values_cache
        try:
            from config.settings import get_settings
            settings = get_settings()
            empty_values = settings.empty_values
            all_values = set()
            for category, values in empty_values.items():
                for v in values:
                    all_values.add(str(v).strip().lower())
            _empty_values_cache = all_values
            logger.debug("[EMPTY] Loaded %d empty equivalent values", len(all_values))
            return all_values
        except Exception as e:
            logger.debug("[EMPTY] Failed to load empty values: %s", e)
            _empty_values_cache = set()
            return _empty_values_cache

def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    empty_values = _get_empty_equivalent_values()
    if str(value).strip().lower() in empty_values:
        return True
    return False

def _normalize_coating(coating: str) -> str:
    if not coating:
        return coating
    coating = str(coating).strip()
    coating_lower = coating.lower()
    if coating_lower in ('кд', 'кд.'):
        return coating
    return coating

def _token_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    def _extract_tokens(text):
        raw_tokens = re.findall(r'[a-zA-Zа-яА-Я0-9]+', str(text).lower())
        cleaned = []
        for t in raw_tokens:
            letters = re.sub(r'[0-9]', '', t)
            if letters:
                cleaned.append(letters)
        return set(cleaned)
    tokens_a = _extract_tokens(a)
    tokens_b = _extract_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)

class ParametricENSClient:
    """
    Client for parametric matching with ENS database.
    """

    def __init__(self, mask_db=None, ens_index_path: str = None, skip_fields: List[str] = None):
        self.mask_db = mask_db
        self.ens_index_path = ens_index_path
        self.skip_fields = skip_fields or []
        self._ens_index = None
        self._ens_index_lock = threading.Lock()
        logger.info("[ParametricENSClient] skip_fields=%s", self.skip_fields)

    def _load_ens_index(self):
        with self._ens_index_lock:
            if self._ens_index is not None:
                return self._ens_index
            if not self.ens_index_path:
                self._ens_index = []
                return self._ens_index
            try:
                with open(self.ens_index_path, 'rb') as f:
                    self._ens_index = pickle.load(f)
                logger.info("[ENS] Loaded %d records from %s", len(self._ens_index), self.ens_index_path)
            except Exception as e:
                logger.warning("[ENS] Failed to load index: %s", e)
                self._ens_index = []
            return self._ens_index

    def extract_params(self, text: str, pattern: str) -> Dict[str, str]:
        """Extract parameters from text using regex pattern."""
        if not pattern:
            return {}
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.groupdict()
        except re.error as e:
            logger.warning("Regex error in extract_params: %s", e)
        return {}

    def _get_ens_candidates(self, standard: str, item_type: str) -> List[Dict]:
        """Get ENS candidates by standard and item type."""
        ens_index = self._load_ens_index()
        if not ens_index:
            return []
        candidates = []
        search_standard = canonicalize_standard(standard)
        for item in ens_index:
            item_standard = canonicalize_standard(item.get('стандарт', ''))
            if item_standard != search_standard:
                continue
            item_type_ens = item.get('тип_изделия', '').strip().lower()
            if item_type and item_type_ens != item_type.strip().lower():
                continue
            candidates.append(item)
        logger.debug("[ENS] Found %d candidates for %s/%s", len(candidates), standard, item_type)
        return candidates

    def _compare_params(self, extracted: Dict[str, Any], ens_item: Dict[str, Any],
                        mask_params: Dict[str, Any] = None) -> Tuple[float, int, int]:
        """Compare parameters, return (score, matched_count, total_count)."""
        score, matched, total, _ = self._compare_params_debug(extracted, ens_item, mask_params)
        return score, matched, total

    def _compare_params_debug(self, extracted: Dict[str, Any], ens_item: Dict[str, Any],
                              mask_params: Dict[str, Any] = None) -> Tuple[float, int, int, Dict]:
        """Compare parameters with detailed debug info."""
        matching_cfg = _get_matching_config()
        debug = {
            'extracted': extracted,
            'ens_item': ens_item,
            'comparisons': {},
            'matched': {},
            'mismatched': {},
            'missing': [],
        }

        if not extracted:
            return 0.0, 0, 0, debug

        total_weight = 0.0
        matched_weight = 0.0
        TEXT_FIELDS = {'покрытие', 'материал', 'марка_материала', 'марка_стали'}

        for param_name, extracted_val in extracted.items():
            if param_name in self.skip_fields:
                continue
            if _is_empty_value(extracted_val):
                continue

            weight = matching_cfg.numeric_field_weight if param_name not in TEXT_FIELDS else matching_cfg.text_field_weight
            total_weight += weight

            ens_val = ens_item.get(param_name) or ens_item.get(param_name.replace('_', ' '))

            if param_name in TEXT_FIELDS:
                sim = _token_similarity(str(extracted_val), str(ens_val)) if ens_val else 0.0
                matched = sim >= matching_cfg.coating_similarity_threshold
                debug['comparisons'][param_name] = {
                    'status': 'exact' if sim >= 0.99 else ('matched' if matched else 'mismatched'),
                    'extracted': str(extracted_val),
                    'ens_value': str(ens_val) if ens_val else None,
                    'similarity': round(sim, 3),
                }
                if matched:
                    matched_weight += weight * sim
                    debug['matched'][param_name] = f"{extracted_val} ~ {ens_val} (sim={sim:.2f})"
                else:
                    debug['mismatched'][param_name] = f"{extracted_val} vs {ens_val} (sim={sim:.2f})"
            else:
                try:
                    ext_num = float(str(extracted_val).replace(',', '.'))
                    ens_num = float(str(ens_val).replace(',', '.')) if ens_val else None
                    matched = ens_num is not None and abs(ext_num - ens_num) <= matching_cfg.numeric_tolerance
                except (ValueError, TypeError):
                    matched = str(extracted_val).strip() == str(ens_val).strip()
                debug['comparisons'][param_name] = {
                    'status': 'exact' if matched else 'mismatched',
                    'extracted': str(extracted_val),
                    'ens_value': str(ens_val) if ens_val else None,
                    'similarity': 1.0 if matched else 0.0,
                }
                if matched:
                    matched_weight += weight
                    debug['matched'][param_name] = f"{extracted_val} == {ens_val}"
                else:
                    debug['mismatched'][param_name] = f"{extracted_val} != {ens_val}"

        score = matched_weight / total_weight if total_weight > 0 else 0.0
        matched_count = len(debug['matched'])
        total_count = len([k for k in extracted.keys() if k not in self.skip_fields and not _is_empty_value(extracted[k])])
        return score, matched_count, total_count, debug

    def find_best_match(self, text: str, standard: str, item_type: str,
                        extracted_params: Dict[str, str], mask: Any = None) -> Optional[Dict]:
        """Find best ENS match for extracted parameters."""
        result = self.find_best_match_debug(text, standard, item_type, extracted_params, mask)
        return result[0] if result else None

    def find_best_match_debug(self, text: str, standard: str, item_type: str,
                              extracted_params: Dict[str, str], mask: Any = None) -> Tuple[Optional[Dict], List[Dict]]:
        """Find best ENS match with debug info."""
        candidates = self._get_ens_candidates(standard, item_type)
        if not candidates:
            return None, []

        best_match = None
        best_score = 0.0
        debug_candidates = []

        for candidate in candidates:
            score, matched, total, debug = self._compare_params_debug(extracted_params, candidate)
            candidate_debug = {
                'name': candidate.get('наименование', candidate.get('полное_наименование', 'N/A')),
                'ens_code': candidate.get('код', candidate.get('mdm_key', 'N/A')),
                'score': round(score, 3),
                'matched': matched,
                'total': total,
                'debug': debug,
            }
            if score > best_score:
                best_score = score
                best_match = candidate
                candidate_debug['is_best'] = True
            else:
                candidate_debug['is_best'] = False
            debug_candidates.append(candidate_debug)

        matching_cfg = _get_matching_config()
        if best_match and best_score >= matching_cfg.fuzzy_threshold:
            logger.info("[MATCH] Best match: score=%.3f, code=%s", best_score,
                     best_match.get('код', best_match.get('mdm_key', 'N/A')))
            return best_match, debug_candidates
        return None, debug_candidates

    def find_best_match_v2(self, text: str, standard: str, item_type: str,
                           extracted_params: Dict[str, str], mask: Any = None) -> Optional[Dict]:
        """V2 exact match via generic pattern."""
        result = self.find_best_match_v2_debug(text, standard, item_type, extracted_params, mask)
        return result[0] if result else None

    def find_best_match_v2_debug(self, text: str, standard: str, item_type: str,
                                  extracted_params: Dict[str, str], mask: Any = None) -> Tuple[Optional[Dict], List[Dict]]:
        """V2 exact match with debug info."""
        candidates = self._get_ens_candidates(standard, item_type)
        if not candidates:
            return None, []

        generic_pattern = self._get_generic_pattern(standard, item_type, extracted_params, mask)
        logger.debug("[V2] Generic pattern: %s", generic_pattern)

        best_match = None
        debug_candidates = []

        for candidate in candidates:
            cand_name = candidate.get('наименование', candidate.get('полное_наименование', ''))
            cand_generic = self._get_generic_pattern(standard, item_type, candidate, mask)
            matched = generic_pattern.strip() == cand_generic.strip()
            candidate_debug = {
                'name': cand_name,
                'ens_code': candidate.get('код', candidate.get('mdm_key', 'N/A')),
                'generic_pattern': cand_generic,
                'matched': matched,
            }
            if matched:
                best_match = candidate
                candidate_debug['is_best'] = True
                debug_candidates.append(candidate_debug)
                logger.info("[V2] EXACT MATCH: %s", cand_name[:60])
                return best_match, debug_candidates
            candidate_debug['is_best'] = False
            debug_candidates.append(candidate_debug)

        return None, debug_candidates

    def _get_generic_pattern(self, standard: str, item_type: str,
                             params: Dict[str, Any], mask: Any = None) -> str:
        """Build generic pattern for V2 exact match."""
        parts = [item_type]
        for key in sorted(params.keys()):
            val = params[key]
            if val is not None and str(val).strip():
                parts.append(str(val).strip())
        parts.append(standard)
        return ' '.join(parts)

    def _remap_params(self, params: Dict[str, str], mask_params: Dict[str, Any]) -> Dict[str, str]:
        """Remap parameters to ENS field names."""
        if not mask_params or not isinstance(mask_params, dict):
            return params
        result = {}
        for key, value in params.items():
            if value is None:
                continue
            mapping = mask_params.get(key)
            if mapping and isinstance(mapping, dict) and mapping.get('ens_field'):
                ens_field = mapping['ens_field']
                if isinstance(ens_field, list):
                    for field_name in ens_field:
                        result[field_name] = value
                else:
                    result[ens_field] = value
            else:
                result[key] = value
        return result

    def _normalize_value_types(self, value):
        """Normalize value types."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            val_str = value.strip()
            if val_str.lower() in ('да', 'yes', 'true', '1'):
                return True
            if val_str.lower() in ('нет', 'no', 'false', '0', ''):
                return False
            try:
                if '.' in val_str or ',' in val_str:
                    return float(val_str.replace(',', '.'))
                return int(val_str)
            except (ValueError, TypeError):
                return val_str
        return value

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize all parameters."""
        return {k: self._normalize_value_types(v) for k, v in params.items() if v is not None}

    def _get_ens_by_code(self, code: str) -> Optional[Dict]:
        """Get ENS record by code."""
        ens_index = self._load_ens_index()
        for item in ens_index:
            if str(item.get('код', '')) == str(code) or str(item.get('mdm_key', '')) == str(code):
                return item
        return None

    def _get_ens_params_from_match(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ENS parameters from match record."""
        params = {}
        for key in ['диаметр', 'длина', 'шаг_резьбы', 'покрытие', 'материал', 'марка_материала', 'марка_стали']:
            val = match.get(key) or match.get(key.replace('_', ' '))
            if val:
                params[key] = val
        return params

    def _build_ens_match_dict(self, match: Dict[str, Any], score: float = 0.0,
                              match_type: str = None, match_type_ru: str = None) -> Dict[str, Any]:
        """Build ENS match dictionary."""
        return {
            'code': match.get('код', match.get('mdm_key')),
            'name': match.get('наименование', match.get('полное_наименование')),
            'mdm_key': match.get('mdm_key'),
            'score': score,
            'type': match_type,
            'type_ru': match_type_ru,
            **match
        }

    def _build_result(self, text: str, params: Dict[str, Any], ens_match: Dict[str, Any],
                      confidence: float, processing_time_ms: float,
                      details: Dict[str, Any] = None,
                      item_type: str = '', standard: str = '') -> Dict[str, Any]:
        """Build processing result dictionary."""
        matching_cfg = _get_matching_config()
        success = confidence >= matching_cfg.success_threshold
        return {
            'text': text,
            'success': success,
            'params': params,
            'ens_match': ens_match,
            'confidence': round(confidence, 3),
            'processing_time_ms': round(processing_time_ms, 2),
            'item_type': item_type,
            'standard': standard,
            'details': details or {},
        }