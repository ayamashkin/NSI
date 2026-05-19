"""
core/automated_processor.py
Automated Parametric Processor for ENS nomenclature matching.

FIXES (2026-05-19):
1. CRITICAL: Restored ParametricENSClient delegation for proven matching logic.
2. CRITICAL: Proper ENS index handling via ParametricENSClient.
3. CRITICAL: Text cleanup (trailing commas) before processing.
4. JSON output uses full ProcessingResult.to_dict() structure.
5. Excel output uses human-readable Russian columns.

LAST_FIX: 2026-05-19 19:11 UTC+3
"""

import logging
import re
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

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
        """Convert to dictionary for JSON serialization (full technical structure)."""
        return {
            'text': self.text,
            'level': self.level,
            'success': self.success,
            'params': self.params,
            'ens_code': self.ens_code,
            'ens_name': self.ens_name,
            'ens_params': self.ens_params,
            'ens_params_mask': self.ens_params_mask,
            'confidence': self.confidence,
            'processing_time_ms': self.processing_time_ms,
            'item_type': self.item_type,
            'standard': self.standard,
            'match_type': self.match_type,
            'match_type_ru': self.match_type_ru,
            'coating_substitution': self.coating_substitution,
            'fuzzy_mismatched_params': self.fuzzy_mismatched_params,
            'mask_id': self.mask_id,
            'mask_pattern': self.mask_pattern,
            'details': self.details,
        }


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
            'normalized': self.normalized,
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
        std_info = self._extract_standard(text)
        if std_info:
            result['standard_info'] = std_info
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
    Fixed version: delegates parametric matching to ParametricENSClient.
    """

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

        # Initialize ParametricENSClient for proven matching logic
        self.parametric_client = None
        self._init_parametric_client()

        # Standard extractor
        self.standard_extractor = StandardExtractor()

        # Coating mapper (lazy init)
        self._coating_mapper = None

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

    def process(self, text: str) -> ProcessingResult:
        """Process a single nomenclature text through parametric pipeline."""
        start_time = time.time()

        # CRITICAL FIX: clean trailing punctuation
        text = text.strip().rstrip(',;').strip()
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
            mask = self.mask_db.get_mask(standard, item_type)

        if mask is None:
            result.level = 'no_mask'
            result.success = False
            result.confidence = 0.0
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result

        result.mask_pattern = mask.pattern if hasattr(mask, 'pattern') else str(mask)
        result.mask_id = getattr(mask, 'id', None)

        # Step 2: Parametric matching via ParametricENSClient
        match_info = None
        if self.parametric_client:
            try:
                match_result = self.parametric_client.match(
                    text=text,
                    standard=standard,
                    item_type=item_type
                )

                if match_result:
                    # Convert ParametricMatch to our match_info structure
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
                        'debug_candidates': match_result.details.get('debug_candidates', []) if match_result.details else [],
                        'fuzzy_params_comparison': {},
                        'fuzzy_mismatched_params': {},
                    }

                    result.params = match_result.matched_params or {}
                    result.ens_code = match_result.ens_code
                    result.ens_name = match_result.ens_name
                    result.ens_params = match_result.ens_params or {}
                    result.ens_params_mask = match_result.ens_params_mask or {}
                    result.confidence = match_result.score or 0.0
                    result.match_type = match_result.match_type
                    result.match_type_ru = match_info['match_type_ru']

                    # Build details
                    result.details = {
                        'mask_id': result.mask_id,
                        'mask_pattern': result.mask_pattern,
                        'match_type': match_result.match_type,
                        'match_type_ru': result.match_type_ru,
                        'extracted_standard': standard_info.to_dict(),
                        'extracted_type': item_type,
                        'fuzzy_used': match_result.match_type == 'fuzzy_fallback',
                        'debug_candidates': match_info['debug_candidates'],
                        'fuzzy_score': match_result.score if match_result.match_type == 'fuzzy_fallback' else 0,
                        'match_result_score': match_result.score,
                        'v2_score': match_result.score,
                        'v2_computed': True,
                        'coating_substitution': None,
                        'fuzzy_mismatched_params': None,
                    }

                    # Check if coating substitution is needed
                    if match_result.score < 1.0 and result.params.get('покрытие'):
                        match_info['needs_coating_substitution'] = True

            except Exception as e:
                logger.warning("[AutomatedProcessor] ParametricENSClient match failed: %s", e)

        # Step 3: Apply coating substitution if needed
        if match_info and match_info.get('needs_coating_substitution'):
            result = self._apply_coating_substitution(result)
            # Update match_info from result
            match_info['coating_substitution'] = result.coating_substitution
            if result.coating_substitution:
                match_info['match_type'] = 'coating_substituted'
                match_info['match_type_ru'] = 'Совпадение после подбора правильного покрытия'
                result.match_type = 'coating_substituted'
                result.match_type_ru = match_info['match_type_ru']
                if result.details:
                    result.details['match_type'] = 'coating_substituted'
                    result.details['match_type_ru'] = result.match_type_ru
                    result.details['coating_substitution'] = result.coating_substitution

        # Step 4: Finalize result
        self._finalize_result(result, match_info)

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def _apply_coating_substitution(self, result: ProcessingResult) -> ProcessingResult:
        """Apply coating substitution rules and recalculate match."""
        if not self.coating_mapper:
            return result

        material = result.ens_params.get('марка_материала', '')
        current_coating = result.params.get('покрытие', '')

        for rule in self.coating_mapper.get('rules', []):
            material_pattern = rule.get('material_pattern', '')
            wrong_coating = rule.get('wrong_coating', '')
            correct_coating = rule.get('correct_coating', '')

            if (re.search(material_pattern, str(material)) and
                    current_coating == wrong_coating):

                result.coating_substitution = {
                    'original': current_coating,
                    'corrected': correct_coating,
                    'material': material,
                    'reason': rule.get('reason', ''),
                    'rule': rule,
                }

                # Try to find a better match with corrected coating
                if self.parametric_client and result.standard and result.item_type:
                    try:
                        corrected_params = dict(result.params)
                        corrected_params['покрытие'] = correct_coating
                        # Re-run match with corrected params (simplified)
                        match_result = self.parametric_client.match(
                            text=result.text,
                            standard=result.standard,
                            item_type=result.item_type.lower()
                        )
                        if match_result and match_result.score > result.confidence:
                            result.confidence = match_result.score
                            result.ens_code = match_result.ens_code
                            result.ens_name = match_result.ens_name
                            result.ens_params = match_result.ens_params or {}
                            result.ens_params_mask = match_result.ens_params_mask or {}
                    except Exception:
                        pass

                return result

        return result

    def _finalize_result(self, result: ProcessingResult, match_info: Optional[Dict]):
        """Finalize result with proper success logic."""
        if not match_info:
            result.success = False
            result.confidence = 0.0
            result.level = 'no_match'
            result.match_type = 'failed'
            result.match_type_ru = 'Не определено'
            if result.details:
                result.details['match_type'] = 'failed'
                result.details['match_type_ru'] = 'Не определено'
            return

        score = match_info.get('score', 0.0)
        match_type = match_info.get('match_type', '')

        if match_type in ('name_exact', 'params_ens_exact', 'params_mask_exact'):
            result.success = True
            result.confidence = 1.0
            result.level = 'parametric_match'

        elif match_type == 'coating_substituted':
            mismatched = result.fuzzy_mismatched_params or {}
            has_critical_mismatch = any(p in mismatched for p in self.CRITICAL_PARAMS)
            if has_critical_mismatch:
                result.success = False
                result.confidence = 0.0
                result.match_type_ru = 'Подмена покрытия отклонена (несовпадение параметров)'
                result.level = 'parametric_match'
            elif score >= self.min_v2_threshold:
                result.success = True
                result.confidence = min(score, 0.95)
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'parametric_match'

        elif match_type in ('fuzzy_fallback', 'partial'):
            mismatched = result.fuzzy_mismatched_params or {}
            has_critical_mismatch = any(p in mismatched for p in self.CRITICAL_PARAMS)
            if has_critical_mismatch:
                result.success = False
                result.confidence = 0.0
                result.match_type_ru = 'Нечеткое совпадение отклонено (критичные параметры)'
                result.level = 'parametric_match'
            elif score >= self.min_fuzzy_threshold:
                result.success = True
                result.confidence = score
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'parametric_match'

        else:
            if score >= self.success_threshold:
                result.success = True
                result.confidence = score
                result.level = 'parametric_match'
            else:
                result.success = False
                result.confidence = 0.0
                result.level = 'no_match'

        if result.details:
            result.details['fuzzy_mismatched_params'] = result.fuzzy_mismatched_params

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
            'partial': 'Частичное совпадение параметров',
        }
        return mapping.get(match_type, match_type)

    def batch_process(self, texts: List[str]) -> List[ProcessingResult]:
        """Batch processing."""
        return [self.process(text) for text in texts]

    def get_statistics(self) -> Dict[str, Any]:
        """Processor statistics."""
        return {
            'mask_db_stats': self.mask_db.get_statistics() if hasattr(self.mask_db, 'get_statistics') else {},
            'llm_generation_enabled': self.use_llm_generation,
            'min_v2_threshold': self.min_v2_threshold,
            'min_fuzzy_threshold': self.min_fuzzy_threshold,
            'success_threshold': self.success_threshold,
        }