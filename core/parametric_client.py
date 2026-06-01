#!/usr/bin/env python3
# =============================================================================
# Parametric ENS Client
# Client for parametric matching against ENS (ЕНС) reference data.
#
# LAST_FIXES:
#  2026-06-01 11:44:00 UTC+3 — FIX: extract_params() now logs pattern and re.match result for debug
#  2026-05-28 20:40:00 UTC+3 — FIX: get_mask() now normalizes standard via canonicalize_standard()
#  2026-05-28 20:30:00 UTC+3 — FIX: get_mask() fallback search by standard error fixed
#  2026-05-28 20:15:00 UTC+3 — FIX: get_mask() now uses canonicalize_standard() for standard normalization
#  2026-05-28 20:00:00 UTC+3 — FIX: get_mask() fallback search by standard error fixed
# =============================================================================

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from utils.standard_utils import canonicalize_standard
from core.domain_config import DomainConfig

logger = logging.getLogger(__name__)


@dataclass
class MaskRecord:
    """Record representing a single regex mask."""
    id: Optional[int] = None
    standard: str = ""
    item_type: str = ""
    pattern: str = ""
    params: List[str] = field(default_factory=list)
    required: List[str] = field(default_factory=list)
    source: str = "manual"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    is_active: bool = True
    validation_score: Optional[float] = None
    auto_score: Optional[float] = None
    notes: Optional[str] = None
    version: int = 1
    previous_version_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "standard": self.standard,
            "item_type": self.item_type,
            "pattern": self.pattern,
            "params": self.params,
            "required": self.required,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "validation_score": self.validation_score,
            "auto_score": self.auto_score,
            "notes": self.notes,
            "version": self.version,
            "previous_version_id": self.previous_version_id,
        }


class ParametricENSClient:
    """Client for parametric matching against ENS reference data."""

    def __init__(self, mask_db=None, ens_index_path: str = None, skip_fields: List[str] = None):
        self.mask_db = mask_db
        self.ens_index_path = ens_index_path
        self.skip_fields = set(skip_fields or [])
        logger.info("[ParametricENSClient] skip_fields=%s", list(self.skip_fields))

    def extract_params(self, text: str, pattern: str) -> Dict[str, str]:
        """Extract parameters using regex pattern.

        FIX 2026-06-01 11:44:00 UTC+3: added debug logging for pattern and match result.
        """
        logger.debug("[extract_params] text=%r pattern=%r", text, pattern)
        match = re.match(pattern, text)
        if not match:
            logger.debug("[extract_params] re.match returned None — pattern does NOT match text")
            return {}
        params = {}
        for key, value in match.groupdict().items():
            if value is not None:
                params[key] = value.strip()
        logger.debug("[extract_params] matched groups=%s", params)
        return params

    def find_best_match(self, text: str, standard: str, item_type: str,
                        extracted_params: Dict[str, str], mask: Any) -> Tuple[Optional[Dict], float, List[Dict]]:
        """Find best ENS match using fuzzy scoring."""
        from ens.indexer import ENSIndexLoader
        loader = ENSIndexLoader(self.ens_index_path)
        candidates = loader.get_candidates(standard, item_type)
        if not candidates:
            return None, 0.0, []

        TEXT_FIELDS = {'покрытие', 'материал', 'марка_материала', 'марка_стали'}
        best_match = None
        best_score = 0.0
        debug_info = []

        for candidate in candidates:
            total_weight = 0.0
            matched_weight = 0.0
            param_details = {}

            for param_name, extracted_val in extracted_params.items():
                if not extracted_val:
                    continue
                weight = 2.0 if param_name in TEXT_FIELDS else 1.0
                total_weight += weight

                candidate_val = candidate.get(param_name) or candidate.get(param_name.replace('_', ' '), '')

                if param_name in TEXT_FIELDS:
                    sim = self._token_similarity(extracted_val, candidate_val)
                    matched = sim >= 0.5
                    if matched:
                        matched_weight += weight * sim
                else:
                    try:
                        matched = float(str(extracted_val).replace(',', '.')) == float(str(candidate_val).replace(',', '.'))
                    except (ValueError, TypeError):
                        matched = str(extracted_val).strip() == str(candidate_val).strip()
                    if matched:
                        matched_weight += weight

                param_details[param_name] = {
                    'extracted': extracted_val,
                    'ens_value': candidate_val,
                    'matched': matched,
                    'similarity': sim if param_name in TEXT_FIELDS else (1.0 if matched else 0.0)
                }

            score = matched_weight / total_weight if total_weight > 0 else 0.0
            debug_info.append({
                'candidate': candidate.get('name', 'N/A'),
                'score': score,
                'params': param_details
            })

            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match, best_score, debug_info

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        """Token-based Jaccard similarity."""
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