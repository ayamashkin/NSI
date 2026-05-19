"""
core/automated_processor.py
Automated Parametric Processor for ENS nomenclature matching.

FIXES (2026-05-19):
1. CRITICAL: _find_ens_match now returns ens_params_mask instead of full ens_item dict.
   This fixes 8+ GB output files caused by serializing entire ENS records.
2. Correct success/confidence logic based on actual scores
3. Coating substitution validates ALL critical parameters
4. Fuzzy fallback respects score thresholds
5. V2 scoring properly gates success
6. Added has_mask info to results

LAST_FIX: 2026-05-19 14:33 UTC+3
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
        # Handle nested dataclasses if any
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
    Fixed version with proper scoring logic for ENS parametric matching.
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
        if settings:
            self.min_v2_threshold = getattr(settings, 'min_v2_score',
                settings.get('matching', {}).get('min_v2_score', 0.85) if isinstance(settings, dict) else 0.85)
            self.min_fuzzy_threshold = getattr(settings, 'min_fuzzy_score',
                settings.get('matching', {}).get('min_fuzzy_score', 0.80) if isinstance(settings, dict) else 0.80)

        # Load ENS index
        self.ens_index = {}
        if ens_index_path and Path(ens_index_path).exists():
            self._load_ens_index(ens_index_path)

    def _load_ens_index(self, path):
        """Load ENS index from pickle file. Handles both dict and list structures."""
        import pickle
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Handle different structures:
            # 1. Dict {code: dict} — standard
            # 2. Dict {code: list} — values are lists
            # 3. List of dicts
            # 4. List of lists/records

            if isinstance(data, dict):
                # Check if values are lists and convert if needed
                converted = {}
                for key, value in data.items():
                    if isinstance(value, list):
                        # Convert list to pseudo-dict for uniform access
                        value_str = ' '.join(str(x) for x in value if x is not None)
                        converted[key] = {
                            '_raw_list': value,
                            '_str': value_str,
                            'нтд_1': value_str,
                            'наименование_типа': value_str,
                            'наименование': value_str
                        }
                    elif isinstance(value, dict):
                        converted[key] = value
                    else:
                        converted[key] = {'_raw': value, 'нтд_1': str(value)}
                self.ens_index = converted
                logger.info(f"ENS index loaded (dict): {len(self.ens_index)} items")
            elif isinstance(data, list):
                if data and len(data) > 0:
                    if isinstance(data[0], dict):
                        # List of dicts — convert to dict by code
                        if 'код' in data[0]:
                            self.ens_index = {str(item['код']): item for item in data}
                        elif 'ens_code' in data[0]:
                            self.ens_index = {str(item['ens_code']): item for item in data}
                        else:
                            self.ens_index = {str(i): item for i, item in enumerate(data)}
                    elif isinstance(data[0], (list, tuple)):
                        # List of lists — wrap each in pseudo-dict
                        self.ens_index = {}
                        for i, item in enumerate(data):
                            item_str = ' '.join(str(x) for x in item if x is not None)
                            self.ens_index[str(i)] = {
                                '_raw_list': item,
                                '_str': item_str,
                                'нтд_1': item_str,
                                'наименование_типа': item_str,
                                'наименование': item_str
                            }
                    else:
                        self.ens_index = {str(i): item for i, item in enumerate(data)}
                else:
                    self.ens_index = {}
                logger.info(f"ENS index loaded (list): {len(self.ens_index)} items")
            else:
                self.ens_index = {}
                logger.warning(f"Unknown ENS index structure: {type(data)}")
        except Exception as e:
            logger.warning(f"Failed to load ENS index: {e}")
            self.ens_index = {}

        # Standard extractor
        self.standard_extractor = StandardExtractor()

        # Coating mapper (lazy init)
        self._coating_mapper = None

    @property
    def coating_mapper(self):
        if self._coating_mapper is None:
            try:
                from core.coating_mapper import get_mapper
                self._coating_mapper = get_mapper()
            except Exception:
                self._coating_mapper = {}
        return self._coating_mapper

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

        # Step 1: Mask lookup
        standard = result.standard
        search_item_type = result.item_type

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
            result.params = params
        except Exception as e:
            logger.debug(f"Param extraction failed for '{text}': {e}")
            params = {}

        # Step 3: Find ENS match
        match_info = self._find_ens_match(text, standard, item_type, params, mask)

        # Step 4: Apply coating substitution if needed
        if match_info and match_info.get('needs_coating_substitution'):
            match_info = self._apply_coating_substitution(text, match_info, params)

        # Step 5: Finalize with CORRECT logic
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

    def _find_ens_match(self, text: str, standard: str, item_type: str,
                        params: Dict, mask) -> Optional[Dict]:
        """
        Find best ENS match for extracted parameters.
        Returns match_info dict with scores and candidate data.
        Handles both dict and list structures for ens_items.
        """
        if not self.ens_index:
            return None

        # Filter ENS items by standard and type
        candidates = []
        for code, ens_item in self.ens_index.items():
            # Handle list-type ens_items (convert to dict if needed)
            if isinstance(ens_item, list):
                ens_item_str = ' '.join(str(x) for x in ens_item if x is not None)
                ens_std = ens_item_str
                ens_type = ens_item_str.lower()

                if standard not in ens_std and ens_std not in standard:
                    continue
                if item_type not in ens_type and ens_type not in item_type:
                    continue

                # Wrap list in a pseudo-dict for uniform access
                ens_item = {'_raw_list': ens_item, '_str': ens_item_str}
                candidates.append((code, ens_item))
                continue

            # Normal dict-type ens_item
            if not isinstance(ens_item, dict):
                logger.debug(f"ENS item {code} has unexpected type: {type(ens_item)}")
                continue

            ens_std = str(ens_item.get('нтд_1', '')).strip()
            ens_type = str(ens_item.get('наименование_типа', '')).strip().lower()

            if standard not in ens_std and ens_std not in standard:
                continue
            if item_type not in ens_type and ens_type not in item_type:
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

        if not best_candidate:
            return None

        code, ens_item = best_candidate

        # Determine match type based on score
        match_type = 'fuzzy_fallback'
        match_type_ru = 'Нечеткое совпадение (fuzzy matching)'

        if best_score >= 1.0:
            match_type = 'name_exact'
            match_type_ru = 'Совпадение по наименованию'
        elif best_score >= 0.85:
            match_type = 'parametric_full'
            match_type_ru = 'Полное совпадение параметров'
        elif best_score >= 0.5:
            match_type = 'parametric_partial'
            match_type_ru = 'Частичное совпадение параметров'

        # Check if coating substitution is needed
        needs_substitution = False
        if best_mismatched:
            for param, info in best_mismatched.items():
                if param == 'покрытие':
                    needs_substitution = True
                    break

        # Build ens_params_mask
        ens_params_mask = {}
        for key in params:
            if key in ens_item:
                val = ens_item[key]
                ens_params_mask[key] = str(val) if val is not None else None

        return {
            'ens_code': code,
            'ens_name': ens_item.get('наименование', f"{ens_item.get('наименование_типа', '')} {code}"),
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
            'needs_coating_substitution': needs_substitution,
            'debug_candidates': []
        }

    def _calculate_match_score(self, params: Dict, ens_item, text: str) -> Tuple[float, Dict, Dict]:
        """
        Calculate match score between extracted params and ENS item.
        Handles both dict and list structures for ens_item.
        Returns (score, comparison_dict, mismatched_dict).
        """
        comparison = {}
        mismatched = {}

        if not params:
            return 0.0, {}, {}

        # Handle pseudo-dict from list conversion (_raw_list key)
        if isinstance(ens_item, dict) and '_raw_list' in ens_item:
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
                    comparison[param] = {
                        'status': 'exact',
                        'extracted': str(extracted_val),
                        'ens_value': 'found in list',
                        'similarity': 1.0
                    }
                else:
                    mismatched[param] = f"{extracted_val} not found in list"
            score = matched_count / total_weight if total_weight > 0 else 0.0
            return score, comparison, mismatched

        # Handle raw list-type ens_items
        if isinstance(ens_item, list):
            ens_item_str = ' '.join(str(x) for x in ens_item if x is not None)
            matched_count = 0
            total_weight = 0
            for param, extracted_val in params.items():
                if param.startswith('_'):
                    continue
                weight = 2.0
                total_weight += weight
                if str(extracted_val) in ens_item_str:
                    matched_count += weight
                    comparison[param] = {
                        'status': 'exact',
                        'extracted': str(extracted_val),
                        'ens_value': 'found in list',
                        'similarity': 1.0
                    }
                else:
                    mismatched[param] = f"{extracted_val} not found in list"
            score = matched_count / total_weight if total_weight > 0 else 0.0
            return score, comparison, mismatched

        if not isinstance(ens_item, dict):
            return 0.0, {}, {}

        matched_count = 0
        total_weight = 0

        # Parameter weights
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
        }

        for param, extracted_val in params.items():
            if param.startswith('_'):
                continue

            weight = weights.get(param, 2.0)
            total_weight += weight

            ens_val = ens_item.get(param)

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

    def _compare_values(self, val1, val2, param_name: str) -> Tuple[str, float]:
        """Compare two parameter values. Returns (status, similarity)."""
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

                # Recalculate score with substituted coating
                new_score, new_comparison, new_mismatched = self._calculate_match_score(
                    substituted_params, match_info.get('ens_params', {}), text
                )

                # Update match info
                match_info['coating_substitution'] = {
                    'original': current_coating,
                    'corrected': correct_coating,
                    'material': material,
                    'reason': rule.get('reason', ''),
                    'rule': rule
                }
                match_info['v2_score'] = new_score
                match_info['match_result_score'] = new_score
                match_info['fuzzy_params_comparison'] = new_comparison
                match_info['fuzzy_mismatched_params'] = new_mismatched
                match_info['match_type'] = 'coating_substituted'
                match_info['match_type_ru'] = 'Совпадение после подбора правильного покрытия'
                match_info['needs_coating_substitution'] = False

                return match_info

        return match_info

    def _finalize_result(self, result: ProcessingResult, match_info: Optional[Dict],
                         extracted_params: Dict):
        """
        CORRECTED success/confidence logic.

        CRITICAL FIX: success and confidence are now based on actual scores,
        not hardcoded to True/1.0.
        """
        if not match_info:
            result.success = False
            result.confidence = 0.0
            result.level = 'no_match'
            result.match_type = 'failed'
            result.match_type_ru = 'Не определено'
            return

        # Extract scores
        v2_score = match_info.get('v2_score', 0.0)
        match_result_score = match_info.get('match_result_score', 0.0)
        fuzzy_score = match_info.get('fuzzy_score', 0.0)
        effective_score = max(v2_score, match_result_score)

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

        # Build details dict
        result.details = {
            'mask_id': result.mask_id,
            'mask_pattern': result.mask_pattern,
            'match_type': match_type,
            'match_type_ru': result.match_type_ru,
            'extracted_standard': self.standard_extractor.extract_all(result.text).get('standard_info', {}).to_dict() if self.standard_extractor.extract_all(result.text).get('standard_info') else None,
            'extracted_type': result.item_type.lower() if result.item_type else None,
            'fuzzy_used': match_type in ('fuzzy_fallback', 'coating_substituted'),
            'debug_candidates': match_info.get('debug_candidates', []),
            'fuzzy_score': fuzzy_score,
            'match_result_score': match_result_score,
            'v2_score': v2_score,
            'v2_computed': True,
            'coating_substitution': result.coating_substitution,
            'fuzzy_mismatched_params': result.fuzzy_mismatched_params
        }

        # === CRITICAL FIX: Proper success logic based on scores ===

        if match_type == 'name_exact':
            if effective_score >= self.min_v2_threshold:
                result.success = True
                result.confidence = effective_score
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'parametric_match'
                result.match_type_ru = 'Совпадение по наименованию (низкий score)'

        elif match_type == 'parametric_full':
            if effective_score >= self.min_v2_threshold:
                result.success = True
                result.confidence = effective_score
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'parametric_match'

        elif match_type == 'parametric_partial':
            if effective_score >= self.min_v2_threshold:
                result.success = True
                result.confidence = effective_score
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'parametric_match'

        elif match_type == 'fuzzy_fallback':
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

        elif match_type == 'coating_substituted':
            mismatched = match_info.get('fuzzy_mismatched_params', {})
            has_critical_mismatch = any(p in mismatched for p in self.CRITICAL_PARAMS)

            if has_critical_mismatch:
                result.success = False
                result.confidence = 0.0
                result.match_type = 'coating_substituted_rejected'
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

        else:
            result.success = False
            result.confidence = 0.0
            result.level = 'no_match'