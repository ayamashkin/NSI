"""
core/automated_processor.py
Automated Parametric Processor for ENS nomenclature matching.

FIXES (2026-05-19):
1. CRITICAL: _load_ens_index properly handles {'items': [...]} ENS index structure.
2. CRITICAL: _find_ens_match uses correct field names ('нтд', 'standard', 'тип_изделия').
3. CRITICAL: _find_ens_match normalizes standards (ОСТ 1 → ОСТ1) for proper comparison.
4. _calculate_match_score uses fuzzy key matching like parametric_client.
5. _finalize_result uses correct thresholds (success_threshold from config).
6. Restored ParametricENSClient delegation for proven matching logic.

LAST_FIX: 2026-05-19 16:45 UTC+3
"""

import logging
import re
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

# Lazy import для доступа к MatchingConfig (избегаем circular dependency)
_matching_config = None

def _get_matching_config():
    """Ленивая загрузка MatchingConfig из settings."""
    global _matching_config
    if _matching_config is None:
        try:
            from config.settings import get_settings
            _matching_config = get_settings().matching
        except Exception:
            class _FallbackMatchingConfig:
                success_threshold = 0.7
                fuzzy_threshold = 0.6
                v2_exact_threshold = 0.99
                coating_similarity_threshold = 0.8
                strict_union_keys = False
                debug_per_parameter = False
            _matching_config = _FallbackMatchingConfig()
    return _matching_config


@dataclass
class ProcessingResult:
    """Result of processing a single nomenclature item."""
    text: str
    level: str = ""
    success: bool = False
    params: Dict = field(default_factory=dict)
    ens_code: Optional[str] = None
    ens_name: Optional[str] = None
    ens_params: Dict = field(default_factory=dict)
    ens_params_mask: Dict = field(default_factory=dict)
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    item_type: str = ""
    standard: str = ""
    mask_pattern: Optional[str] = None
    match_type: Optional[str] = None
    match_type_ru: Optional[str] = None
    coating_substitution: Optional[Dict] = None
    fuzzy_mismatched_params: Optional[Dict] = None
    fuzzy_params_comparison: Optional[Dict] = None
    details: Optional[Dict] = None
    mask_id: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        return d


class StandardInfo:
    """Parsed standard information."""
    def __init__(self, standard_type: str, standard_number: str, year: str,
                 full_name: str, normalized: str):
        self.standard_type = standard_type
        self.standard_number = standard_number
        self.year = year
        self.full_name = full_name
        self.normalized = normalized

    def to_dict(self):
        return {
            'standard_type': self.standard_type,
            'standard_number': self.standard_number,
            'year': self.year,
            'full_name': self.full_name,
            'normalized': self.normalized
        }


class StandardExtractor:
    """Extracts standard and item type from nomenclature text."""

    STANDARD_PATTERNS = [
        (r'ГОСТ\s*\d+[\s\-]*\d*[-—]\d+', 'ГОСТ'),
        (r'ОСТ\s*\d+\s*\d+[-—]\d+', 'ОСТ'),
        (r'ТУ\s*\d+[-—]\d+', 'ТУ'),
    ]

    ITEM_TYPES = ['Болт', 'Винт', 'Гайка', 'Шайба', 'Шпилька', 'Штифт',
                  'Шпонка', 'Шайба', 'Гровер', 'Шайба-гровер']

    def extract_all(self, text: str) -> Dict:
        result = {}

        # Extract standard
        std_info = self._extract_standard(text)
        if std_info:
            result['standard_info'] = std_info

        # Extract item type
        item_type = self._extract_item_type(text)
        if item_type:
            result['item_type'] = item_type

        return result

    def _extract_standard(self, text: str) -> Optional[StandardInfo]:
        for pattern, std_type in self.STANDARD_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                full = match.group(0)
                normalized = re.sub(r'[-—]', '-', full)
                normalized = re.sub(r'\s+', ' ', normalized).strip()

                # Parse number and year
                parts = normalized.replace(std_type, '').strip().split('-')
                if len(parts) >= 2:
                    year = parts[-1]
                    number = '-'.join(parts[:-1]).strip()
                    return StandardInfo(std_type, number, year, full, normalized)
        return None

    def _extract_item_type(self, text: str) -> Optional[str]:
        for itype in self.ITEM_TYPES:
            if re.search(rf'\b{re.escape(itype)}\b', text, re.IGNORECASE):
                return itype.lower()
        return None


class AutomatedParametricProcessor:
    """
    Fixed version with proper ENS index handling and matching logic.
    Delegates parametric matching to ParametricENSClient for proven reliability.
    """

    # Critical parameters that must match exactly
    CRITICAL_PARAMS = [
        'длина', 'номинальный_диаметр_резьбы', 'исполнение',
        'шаг_резьбы', 'класс_поле_допуска', 'группа_класс_прочности'
    ]

    def __init__(self, mask_db, llm_clients=None, ens_index_path=None,
                 use_llm_generation=False, settings=None):
        self.mask_db = mask_db
        self.llm_clients = llm_clients or {}
        self.ens_index_path = ens_index_path
        self.use_llm_generation = use_llm_generation
        self.settings = settings or {}

        # Thresholds from config or defaults
        self.min_v2_threshold = 0.85
        self.min_fuzzy_threshold = 0.80
        self.success_threshold = 0.7
        if settings:
            if isinstance(settings, dict):
                self.min_v2_threshold = settings.get('matching', {}).get('min_v2_score', 0.85)
                self.min_fuzzy_threshold = settings.get('matching', {}).get('min_fuzzy_score', 0.80)
                self.success_threshold = settings.get('matching', {}).get('success_threshold', 0.7)
            else:
                self.min_v2_threshold = getattr(settings, 'min_v2_score', 0.85)
                self.min_fuzzy_threshold = getattr(settings, 'min_fuzzy_score', 0.80)
                self.success_threshold = getattr(settings, 'success_threshold', 0.7)

        # Load ENS index for direct access (ParametricENSClient also loads its own)
        self.ens_index = {}
        self.ens_items_by_code = {}
        if ens_index_path and Path(ens_index_path).exists():
            self._load_ens_index(ens_index_path)

        # Standard extractor
        self.standard_extractor = StandardExtractor()

        # Coating mapper (lazy init)
        self._coating_mapper = None

        # Initialize ParametricENSClient for proven matching logic
        self._init_parametric_client()

    def _init_parametric_client(self):
        """Initialize ParametricENSClient with proper skip_fields."""
        skip_fields = None
        if self.settings:
            if isinstance(self.settings, dict):
                skip_fields = self.settings.get('output', {}).get('ens_params_skip_fields')
            else:
                if hasattr(self.settings, 'output') and hasattr(self.settings.output, 'ens_params_skip_fields'):
                    skip_fields = self.settings.output.ens_params_skip_fields

        try:
            from core.parametric_client import ParametricENSClient
            self.parametric_client = ParametricENSClient(
                mask_db=self.mask_db,
                ens_index_path=self.ens_index_path,
                skip_fields=skip_fields
            )
            logger.info("[AutomatedProcessor] ParametricENSClient initialized")
        except Exception as e:
            logger.warning("[AutomatedProcessor] Failed to init ParametricENSClient: %s", e)
            self.parametric_client = None

    @property
    def coating_mapper(self):
        if self._coating_mapper is None:
            try:
                from core.coating_mapper import get_mapper
                self._coating_mapper = get_mapper()
            except Exception:
                self._coating_mapper = {}
        return self._coating_mapper

    def _load_ens_index(self, path):
        """Load ENS index from pickle file. Handles standard structures."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            items = []
            # Handle standard structure: {'items': [...], 'metadata': {...}}
            if isinstance(data, dict):
                if 'items' in data and isinstance(data['items'], list):
                    items = data['items']
                    logger.info("[ENS] Loaded %d items from 'items' key", len(items))
                else:
                    # Fallback: assume dict values are items
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            items.extend(value)
                        elif isinstance(value, dict):
                            items.append(value)
                    logger.info("[ENS] Loaded %d items from dict values", len(items))
            elif isinstance(data, list):
                items = data
                logger.info("[ENS] Loaded %d items from list", len(items))

            # Build code-indexed dict
            self.ens_index = {}
            for item in items:
                if isinstance(item, dict):
                    code = str(item.get('код', '')).strip()
                    mdm = str(item.get('mdm_key', '')).strip()
                    if code:
                        self.ens_index[code] = item
                        self.ens_items_by_code[code] = item
                    if mdm and mdm != code:
                        self.ens_index[mdm] = item
                        self.ens_items_by_code[mdm] = item
                    if not code and not mdm:
                        # Use hash of item as key
                        item_key = str(hash(str(item)))
                        self.ens_index[item_key] = item
                elif isinstance(item, list):
                    # Convert list to pseudo-dict
                    item_str = ' '.join(str(x) for x in item if x is not None)
                    item_key = str(len(self.ens_index))
                    self.ens_index[item_key] = {
                        '_raw_list': item,
                        '_str': item_str,
                        'наименование': item_str,
                        'полное_наименование': item_str,
                    }

            logger.info("[ENS] Index ready: %d unique items", len(self.ens_index))
        except Exception as e:
            logger.warning("[ENS] Failed to load ENS index: %s", e)
            self.ens_index = {}

    @staticmethod
    def _normalize_standard(std: Optional[str]) -> str:
        """Нормализация стандарта для сравнения: ОСТ 1 → ОСТ1."""
        if not std:
            return ''
        s = str(std).strip()
        s = re.sub(r'ОСТ\s*1', 'ОСТ1', s)
        s = re.sub(r'\s+', '', s)
        return s.upper()

    def process(self, text: str) -> ProcessingResult:
        """Process a single nomenclature text through parametric pipeline."""
        start_time = time.time()

        result = ProcessingResult(text=text)

        # Step 0: Extract standard and type
        extracted = self.standard_extractor.extract_all(text)
        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')

        if not standard_info or not item_type:
            result.level = 'tfidf_fallback'
            result.success = False
            result.confidence = 0.0
            result.item_type = item_type or ""
            result.standard = standard_info.normalized if standard_info else ""
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        result.item_type = item_type.upper() if item_type else ""
        result.standard = standard_info.normalized
        standard = result.standard
        search_item_type = result.item_type

        # Step 1: Mask lookup
        mask = self.mask_db.get_mask(standard, search_item_type)
        if mask is None:
            # Fallback: try without uppercasing
            mask = self.mask_db.get_mask(standard, item_type)

        if mask is None:
            result.level = 'no_mask'
            result.success = False
            result.confidence = 0.0
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        result.mask_pattern = mask.pattern if hasattr(mask, 'pattern') else str(mask)
        result.mask_id = getattr(mask, 'id', None)

        # Step 2: Extract params with mask
        try:
            params = self._extract_params_with_mask(text, mask)
            result.params = params or {}
        except Exception as e:
            logger.debug("Param extraction failed for '%s': %s", text, e)
            params = {}
            result.params = {}

        # Step 3: Parametric matching via ParametricENSClient (proven logic)
        match_info = None
        if self.parametric_client and params:
            try:
                effective_standard = getattr(mask, 'standard', None) or standard
                match_result = self.parametric_client.match(
                    text=text,
                    standard=effective_standard,
                    item_type=mask.item_type if hasattr(mask, 'item_type') else item_type
                )

                if match_result and match_result.ens_code:
                    # Build match_info from ParametricENSClient result
                    match_info = {
                        'ens_code': match_result.ens_code,
                        'ens_name': match_result.ens_name,
                        'ens_params': match_result.ens_params,
                        'ens_params_mask': match_result.ens_params_mask,
                        'score': match_result.score,
                        'v2_score': match_result.score,
                        'match_result_score': match_result.score,
                        'fuzzy_score': match_result.score,
                        'match_type': match_result.match_type,
                        'match_type_ru': self._match_type_to_ru(match_result.match_type),
                        'needs_coating_substitution': False,
                        'debug_candidates': [],
                    }

                    # Check if coating substitution is needed
                    if match_result.score < 1.0 and params.get('покрытие'):
                        match_info['needs_coating_substitution'] = True

                elif match_result and match_result.matched_params:
                    # Regex-only match
                    result.params = match_result.matched_params

            except Exception as e:
                logger.warning("[AutomatedProcessor] ParametricENSClient match failed: %s", e)

        # Step 4: Fallback direct ENS search if parametric_client didn't find match
        if not match_info and params:
            match_info = self._find_ens_match_direct(text, standard, item_type, params, mask)

        # Step 5: Apply coating substitution if needed
        if match_info and match_info.get('needs_coating_substitution'):
            match_info = self._apply_coating_substitution(text, match_info, params)

        # Step 6: Finalize result
        self._finalize_result(result, match_info, params)

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def _extract_params_with_mask(self, text: str, mask) -> Dict:
        """Extract parameters using regex mask pattern."""
        pattern = mask.pattern if hasattr(mask, 'pattern') else str(mask)
        match = re.search(pattern, text)
        if not match:
            return {}

        params = match.groupdict()
        # Clean up values
        cleaned = {}
        for key, value in params.items():
            if value is None:
                continue
            # Try to convert numbers
            try:
                if '.' in value or ',' in value:
                    cleaned[key] = float(value.replace(',', '.'))
                else:
                    cleaned[key] = int(value)
            except (ValueError, TypeError):
                cleaned[key] = value.strip() if value else None
        return cleaned

    def _find_ens_match_direct(self, text: str, standard: str, item_type: str,
                                params: Dict, mask) -> Optional[Dict]:
        """Direct ENS search as fallback when ParametricENSClient fails."""
        if not self.ens_index:
            return None

        std_norm = self._normalize_standard(standard)
        query_type = item_type.upper().strip() if item_type else None

        candidates = []
        for code, ens_item in self.ens_index.items():
            if not isinstance(ens_item, dict):
                continue

            # Check standard - try multiple field names
            item_std = (ens_item.get('нтд') or
                       ens_item.get('standard') or
                       ens_item.get('нтд_1', ''))
            item_std_norm = self._normalize_standard(item_std) if item_std else ''

            if std_norm and item_std_norm and std_norm != item_std_norm:
                continue

            # Check type
            item_type_field = str(
                ens_item.get('тип_изделия', '') or
                ens_item.get('наименование_типа', '') or
                ens_item.get('тип', '')
            ).upper().strip()

            if query_type and item_type_field and query_type != item_type_field:
                continue

            candidates.append((code, ens_item))

        if not candidates:
            return None

        # Score candidates
        best_score = 0.0
        best_candidate = None
        best_comparison = {}
        best_mismatched = {}

        for code, ens_item in candidates:
            score, comparison, mismatched = self._calculate_match_score(
                params, ens_item, text
            )
            if score > best_score:
                best_score = score
                best_candidate = (code, ens_item)
                best_comparison = comparison
                best_mismatched = mismatched

        if not best_candidate or best_score < 0.5:
            return None

        code, ens_item = best_candidate

        # Determine match type
        if best_score >= 1.0:
            match_type = 'name_exact'
            match_type_ru = 'Совпадение по наименованию'
        elif best_score >= 0.85:
            match_type = 'parametric_full'
            match_type_ru = 'Полное совпадение параметров'
        elif best_score >= 0.5:
            match_type = 'parametric_partial'
            match_type_ru = 'Частичное совпадение параметров'
        else:
            match_type = 'fuzzy_fallback'
            match_type_ru = 'Нечеткое совпадение (fuzzy matching)'

        # Build ens_params_mask
        ens_params_mask = {}
        for key in params:
            if key in ens_item:
                val = ens_item[key]
                ens_params_mask[key] = str(val) if val is not None else None

        return {
            'ens_code': code,
            'ens_name': (ens_item.get('наименование') or
                        ens_item.get('полное_наименование') or
                        f"{ens_item.get('наименование_типа', '')} {code}"),
            'ens_params': ens_params_mask,
            'ens_params_mask': ens_params_mask,
            'score': best_score,
            'v2_score': best_score,
            'match_result_score': best_score,
            'fuzzy_score': best_score,
            'match_type': match_type,
            'match_type_ru': match_type_ru,
            'fuzzy_params_comparison': best_comparison,
            'fuzzy_mismatched_params': best_mismatched,
            'needs_coating_substitution': bool(best_mismatched.get('покрытие')),
            'debug_candidates': []
        }

    def _calculate_match_score(self, params: Dict, ens_item: Dict, text: str) -> Tuple[float, Dict, Dict]:
        """Calculate match score with fuzzy key matching."""
        comparison = {}
        mismatched = {}

        if not params:
            return 0.0, {}, {}

        if isinstance(ens_item, dict) and '_raw_list' in ens_item:
            # Pseudo-dict from list
            ens_item_str = ens_item.get('_str', '')
            matched_count = 0
            total_weight = 0
            for param, extracted_val in params.items():
                if param.startswith('_'):
                    continue
                weight = 2.0
                total_weight += weight
                if str(extracted_val) in ens_item_str:
                    matched_count += weight
                    comparison[param] = {'status': 'exact', 'similarity': 1.0}
                else:
                    mismatched[param] = f"{extracted_val} not found in list"
            score = matched_count / total_weight if total_weight > 0 else 0.0
            return score, comparison, mismatched

        if not isinstance(ens_item, dict):
            return 0.0, {}, {}

        matched_count = 0.0
        total_weight = 0.0

        weights = {
            'исполнение': 5.0,
            'номинальный_диаметр_резьбы': 5.0,
            'длина': 5.0,
            'покрытие': 4.0,
            'шаг_резьбы': 3.0,
            'класс_поле_допуска': 3.0,
            'группа_класс_прочности': 3.0,
            'тип_изделия': 2.0,
            'нтд_1': 2.0,
            'нтд': 2.0,
        }

        # Build key mapping with fuzzy matching
        ens_keys = list(ens_item.keys())
        key_map = {}
        for param in params.keys():
            if param.startswith('_'):
                continue
            if param in ens_item:
                key_map[param] = param
            else:
                # Fuzzy key match
                best_key = self._fuzzy_match_key(param, ens_keys)
                if best_key:
                    key_map[param] = best_key

        for param, extracted_val in params.items():
            if param.startswith('_'):
                continue

            weight = weights.get(param, 2.0)
            total_weight += weight

            ens_key = key_map.get(param)
            ens_val = ens_item.get(ens_key) if ens_key else None

            status, similarity = self._compare_values(extracted_val, ens_val, param)

            comparison[param] = {
                'status': status,
                'extracted': str(extracted_val) if extracted_val is not None else '',
                'ens_value': str(ens_val) if ens_val is not None else '',
                'similarity': similarity
            }

            if status in ('exact', 'token_matched'):
                matched_count += weight * similarity
            else:
                mismatched[param] = f"{extracted_val} != {ens_val}"

        score = matched_count / total_weight if total_weight > 0 else 0.0
        return score, comparison, mismatched

    def _fuzzy_match_key(self, param: str, ens_keys: List[str]) -> Optional[str]:
        """Find best matching key in ens_keys for param."""
        param_lower = param.lower().replace('_', '')
        for key in ens_keys:
            key_lower = key.lower().replace('_', '')
            if param_lower == key_lower:
                return key
            # Check if one contains the other
            if param_lower in key_lower or key_lower in param_lower:
                return key
        return None

    def _compare_values(self, val1, val2, param_name: str) -> Tuple[str, float]:
        """Compare two parameter values."""
        if val1 is None and val2 is None:
            return ('exact', 1.0)
        if val1 is None or val2 is None:
            return ('mismatched', 0.0)

        # Numeric comparison
        try:
            n1 = float(str(val1).replace(',', '.'))
            n2 = float(str(val2).replace(',', '.'))
            if abs(n1 - n2) < 0.001:
                return ('exact', 1.0)
            else:
                return ('mismatched', 0.0)
        except (ValueError, TypeError):
            pass

        # String comparison
        s1 = str(val1).strip().lower()
        s2 = str(val2).strip().lower()

        if s1 == s2:
            return ('exact', 1.0)

        # Token matching for coatings
        if param_name == 'покрытие':
            tokens1 = set(s1.replace('.', ' ').split())
            tokens2 = set(s2.replace('.', ' ').split())
            if tokens1 & tokens2:
                similarity = len(tokens1 & tokens2) / max(len(tokens1 | tokens2), 1)
                return ('token_matched', round(similarity, 3))

        return ('mismatched', 0.0)

    def _apply_coating_substitution(self, text: str, match_info: Dict,
                                    params: Dict) -> Dict:
        """Apply coating substitution rules and recalculate match."""
        if not self.coating_mapper:
            return match_info

        material = match_info.get('ens_params', {}).get('марка_материала', '')
        current_coating = params.get('покрытие', '')

        # Find applicable rule
        for rule in self.coating_mapper.get('rules', []):
            material_pattern = rule.get('material_pattern', '')
            wrong_coating = rule.get('wrong_coating', '')
            correct_coating = rule.get('correct_coating', '')

            if (re.search(material_pattern, str(material)) and
                    current_coating == wrong_coating):

                # Create substituted params
                substituted_params = dict(params)
                substituted_params['покрытие'] = correct_coating

                # Recalculate score
                # Find the ENS item
                ens_code = match_info.get('ens_code')
                ens_item = self.ens_index.get(ens_code) if ens_code else None
                if ens_item:
                    new_score, new_comparison, new_mismatched = self._calculate_match_score(
                        substituted_params, ens_item, text
                    )
                else:
                    new_score = match_info.get('score', 0.0)
                    new_mismatched = {}

                match_info['coating_substitution'] = {
                    'original': current_coating,
                    'corrected': correct_coating,
                    'material': material,
                    'reason': rule.get('reason', ''),
                    'rule': rule
                }
                match_info['v2_score'] = new_score
                match_info['match_result_score'] = new_score
                match_info['match_type'] = 'coating_substituted'
                match_info['match_type_ru'] = 'Совпадение после подбора правильного покрытия'
                match_info['needs_coating_substitution'] = False
                match_info['fuzzy_mismatched_params'] = new_mismatched
                return match_info

        return match_info

    def _finalize_result(self, result: ProcessingResult, match_info: Optional[Dict],
                         extracted_params: Dict):
        """Finalize result with proper success logic."""
        if not match_info:
            result.success = False
            result.confidence = 0.0
            result.level = 'no_match'
            result.match_type = 'failed'
            result.match_type_ru = 'Не определено'
            return

        score = match_info.get('score', 0.0)
        v2_score = match_info.get('v2_score', 0.0)
        effective_score = max(score, v2_score)

        match_type = match_info.get('match_type', '')
        result.match_type = match_type
        result.match_type_ru = match_info.get('match_type_ru', '')

        # Set ENS data
        result.ens_code = match_info.get('ens_code')
        result.ens_name = match_info.get('ens_name')
        result.ens_params = match_info.get('ens_params', {})
        result.ens_params_mask = match_info.get('ens_params_mask', {})
        result.coating_substitution = match_info.get('coating_substitution')
        result.fuzzy_mismatched_params = match_info.get('fuzzy_mismatched_params')
        result.fuzzy_params_comparison = match_info.get('fuzzy_params_comparison')

        # Build details
        result.details = {
            'mask_id': result.mask_id,
            'mask_pattern': result.mask_pattern,
            'match_type': match_type,
            'match_type_ru': result.match_type_ru,
            'extracted_standard': self.standard_extractor.extract_all(result.text).get('standard_info', {}).to_dict() if self.standard_extractor.extract_all(result.text).get('standard_info') else None,
            'extracted_type': result.item_type.lower() if result.item_type else None,
            'fuzzy_used': match_type == 'fuzzy_fallback',
            'fuzzy_score': match_info.get('fuzzy_score', 0.0),
            'match_result_score': match_info.get('match_result_score', 0.0),
            'v2_score': v2_score,
            'coating_substitution': result.coating_substitution,
            'fuzzy_mismatched_params': result.fuzzy_mismatched_params,
        }

        # Success logic: use success_threshold from config (default 0.7)
        if match_type in ('name_exact', 'parametric_full', 'params_ens_exact', 'params_mask_exact'):
            if effective_score >= self.min_v2_threshold:
                result.success = True
                result.confidence = effective_score
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'parametric_match'

        elif match_type == 'coating_substituted':
            mismatched = match_info.get('fuzzy_mismatched_params', {})
            has_critical_mismatch = any(p in mismatched for p in self.CRITICAL_PARAMS)

            if has_critical_mismatch:
                result.success = False
                result.confidence = 0.0
                result.match_type_ru = 'Подмена покрытия отклонена (несовпадение параметров)'
                result.level = 'parametric_match'
            elif effective_score >= self.min_v2_threshold:
                result.success = True
                result.confidence = min(effective_score, 0.95)
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'parametric_match'

        elif match_type in ('fuzzy_fallback', 'parametric_partial'):
            mismatched = match_info.get('fuzzy_mismatched_params', {})
            has_critical_mismatch = any(p in mismatched for p in self.CRITICAL_PARAMS)

            if has_critical_mismatch:
                result.success = False
                result.confidence = 0.0
                result.match_type_ru = 'Нечеткое совпадение отклонено (критичные параметры)'
                result.level = 'parametric_match'
            elif effective_score >= self.min_fuzzy_threshold:
                result.success = True
                result.confidence = effective_score
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'parametric_match'

        else:
            if effective_score >= self.success_threshold:
                result.success = True
                result.confidence = effective_score
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'no_match'

    @staticmethod
    def _match_type_to_ru(match_type: str) -> str:
        """Convert match type to Russian description."""
        mapping = {
            'name_exact': 'Совпадение по наименованию',
            'params_ens_exact': 'Полное совпадение параметров с индексом',
            'params_mask_exact': 'Полное совпадение параметров с маской ENS',
            'v2_exact': 'Полное совпадение V2',
            'parametric_full': 'Полное совпадение параметров',
            'parametric_partial': 'Частичное совпадение параметров',
            'fuzzy_fallback': 'Нечеткое совпадение (fuzzy matching)',
            'coating_substituted': 'Совпадение после подбора правильного покрытия',
            'regex_only': 'Только regex (без ENS)',
            'failed': 'Не определено',
        }
        return mapping.get(match_type, match_type)

    def batch_process(self, texts: List[str]) -> List[ProcessingResult]:
        """Пакетная обработка."""
        return [self.process(text) for text in texts]

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика процессора."""
        return {
            'mask_db_stats': self.mask_db.get_statistics() if hasattr(self.mask_db, 'get_statistics') else {},
            'llm_generation_enabled': self.use_llm_generation,
            'min_v2_threshold': self.min_v2_threshold,
            'min_fuzzy_threshold': self.min_fuzzy_threshold,
            'success_threshold': self.success_threshold,
        }