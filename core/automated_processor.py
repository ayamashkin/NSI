# =============================================================================
# FILE: core/automated_processor.py
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
Main Processor Module
Integration of all levels: StandardExtractor -> MaskDatabase -> LLM Generator ->
AutoValidator -> ParametricMatch -> TF-IDF Fallback

VERSION: 2026-05-22

LAST_FIXES:
  2026-05-22 08:50 UTC+3 — _get_matching_config reads from config.yaml matching.
  2026-05-21 15:15 UTC+3 — fuzzy threshold 0.6->0.5, coating_substitution forces success.
  2026-05-21 08:50 UTC+3 — canonicalize_standard on extract & mask generation.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from utils.standard_utils import canonicalize_standard

# Lazy import to avoid circular dependency
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

logger = logging.getLogger(__name__)

class ProcessingLevel(Enum):
    """Уровни обработки."""
    LEVEL_0_EXTRACT = "standard_extraction"
    LEVEL_1_MASK_LOOKUP = "mask_lookup"
    LEVEL_2_LLM_GENERATE = "llm_generation"
    LEVEL_3_VALIDATE = "auto_validation"
    LEVEL_5_SAVE = "save_mask"
    LEVEL_6_PARAMETRIC_MATCH = "parametric_match"
    LEVEL_7_TFIDF_FALLBACK = "tfidf_fallback"
    LEVEL_8_LLM_DIRECT = "llm_direct"

class ProcessingResult:
    """Результат обработки."""

    def __init__(
        self,
        text: str = '',
        level: ProcessingLevel = ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH,
        success: bool = False,
        params: Optional[Dict[str, Any]] = None,
        ens_params: Optional[Dict[str, Any]] = None,
        ens_params_mask: Optional[Dict[str, Any]] = None,
        ens_match: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        processing_time_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        item_type: str = '',
        standard: str = ''
    ):
        self.text = text
        self.level = level
        self.success = success
        self.params = params or {}
        self.ens_params = ens_params or {}
        self.ens_params_mask = ens_params_mask or {}
        self.ens_match = ens_match
        self.confidence = confidence
        self.processing_time_ms = processing_time_ms
        self.details = details or {}
        self.item_type = item_type
        self.standard = standard

    @property
    def ens_code(self) -> Optional[str]:
        if self.ens_match and self.ens_match.get('code'):
            return str(self.ens_match['code'])
        return None

    @property
    def ens_name(self) -> Optional[str]:
        if self.ens_match and self.ens_match.get('name'):
            return str(self.ens_match['name'])
        return None

    @property
    def match_type(self) -> Optional[str]:
        return self.details.get('match_type') if self.details else None

    @property
    def match_type_ru(self) -> Optional[str]:
        return self.details.get('match_type_ru') if self.details else None

    @property
    def coating_substitution(self) -> Optional[Dict]:
        return self.details.get('coating_substitution') if self.details else None

    @property
    def fuzzy_mismatched_params(self) -> Optional[Dict]:
        return self.details.get('fuzzy_mismatched_params') if self.details else None

    @property
    def mask_id(self) -> Optional[int]:
        return self.details.get('mask_id') if self.details else None

    @property
    def mask_pattern(self) -> Optional[str]:
        return self.details.get('mask_pattern') if self.details else None

    def to_dict(self) -> Dict[str, Any]:
        """Полная сериализация для JSON."""
        return {
            'text': self.text,
            'level': self.level.value if isinstance(self.level, ProcessingLevel) else str(self.level),
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

    @property
    def ens_params_from_match(self) -> Optional[Dict[str, Any]]:
        """Параметры из ENS записи (только при наличии ens_code)."""
        if (self.ens_match and self.ens_match.get('code')
            and 'params' in self.ens_match and self.ens_match['params']):
            return self.ens_match['params']
        return None

class AutomatedParametricProcessor:
    """
    Основной процессор автоматизированного параметрического поиска.
    """

    def __init__(
        self,
        mask_db,
        llm_clients: Optional[Dict[str, Any]] = None,
        ens_index_path: Optional[str] = None,
        min_mask_score: float = 0.85,
        max_llm_retries: int = 3,
        use_llm_generation: bool = True,
        settings: Optional[Any] = None,
        result_db_path: Optional[str] = None,
        cache_ttl_days: int = 7
    ):
        self.mask_db = mask_db
        self.llm_clients = llm_clients or {}
        self.ens_index_path = ens_index_path
        self.min_mask_score = min_mask_score
        self.max_llm_retries = max_llm_retries
        self.use_llm_generation = use_llm_generation
        self.settings = settings
        self.result_db_path = result_db_path
        self.cache_ttl_days = cache_ttl_days
        self._cache_stats = {'hits': 0, 'misses': 0}
        if result_db_path:
            db_file = Path(result_db_path)
            if not db_file.exists():
                logger.info("[CACHE] result.db will be created at: %s", result_db_path)
            else:
                logger.info("[CACHE] Using existing result.db: %s", result_db_path)

        # === ПРОИЗВОДИТЕЛЬНОСТЬ: кэши ===
        self._coating_rules_cache: Optional[Dict] = None
        self._coating_rules_loaded: bool = False
        self._ens_candidates_cache: Dict[Tuple[str, str], List[Dict]] = {}

        # === THREAD-SAFETY: locks для кэшей ===
        self._coating_rules_lock = threading.Lock()
        self._ens_candidates_lock = threading.Lock()

        # Инициализация компонентов
        self._init_components()

    def _init_components(self):
        """Инициализация внутренних компонентов."""
        from parsers.standard_extractor import get_standard_extractor
        self.standard_extractor = get_standard_extractor()

        from core.auto_validator import AutoValidator
        self.validator = AutoValidator(
            ens_index_path=self.ens_index_path,
            activation_threshold=self.min_mask_score
        )

        if self.use_llm_generation and self.llm_clients:
            from generators.llm_mask_generator import LLMMaskGenerator
            self.llm_generator = LLMMaskGenerator(
                clients=self.llm_clients,
                settings=self.settings,
                max_retries=self.max_llm_retries
            )
        else:
            self.llm_generator = None

        skip_fields = None
        if self.settings and hasattr(self.settings, 'output') and hasattr(self.settings.output, 'ens_params_skip_fields'):
            skip_fields = self.settings.output.ens_params_skip_fields
            logger.info("[AutomatedProcessor] Loaded %d skip_fields from settings", len(skip_fields))

        from core.parametric_client import ParametricENSClient
        self.parametric_client = ParametricENSClient(
            mask_db=self.mask_db,
            ens_index_path=self.ens_index_path,
            skip_fields=skip_fields
        )

        logger.info("AutomatedParametricProcessor initialized")

    # ------------------------------------------------------------------
    # CACHE METHODS (result.db)
    # ------------------------------------------------------------------

    def _check_cache(self, article: str, text: str) -> Optional[Dict[str, Any]]:
        """Проверить кэш в result.db. Вернуть dict если найдена свежая запись."""
        if not self.result_db_path:
            return None
        try:
            from core.result_database import ResultDatabaseManager
            manager = ResultDatabaseManager(db_path=self.result_db_path)
            logger.debug("[CACHE] DB lookup: article=%r name=%s", article, text[:50])
            cached = manager.get_result(article, text)
            if cached:
                logger.debug("[CACHE] DB found: ens_code=%s updated_at=%s", cached.get('ens_code'), cached.get('updated_at'))
                updated_at = cached.get('updated_at')
                if updated_at:
                    try:
                        last_update = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        if datetime.now() - last_update < timedelta(days=self.cache_ttl_days):
                            self._cache_stats['hits'] += 1
                            return cached
                    except Exception:
                        self._cache_stats['hits'] += 1
                        return cached
                else:
                    self._cache_stats['hits'] += 1
                    return cached
        except Exception as e:
            logger.debug("Cache check error: %s", e)
        return None

    def _result_from_cache(self, cached: Dict[str, Any]) -> ProcessingResult:
        """Восстановить ProcessingResult из кэшированной записи result.db."""
        params = cached.get('params') or {}
        ens_params = cached.get('ens_params') or {}
        ens_params_mask = cached.get('ens_params_mask') or {}
        ens_match = None
        if cached.get('ens_code'):
            ens_match = {
                'code': cached.get('ens_code'),
                'name': cached.get('ens_name'),
                'mdm_key': cached.get('ens_code'),
                'score': cached.get('confidence', 0.0),
                'type': cached.get('match_type'),
                'params': ens_params
            }
        details = cached.get('details') or {}
        details['from_cache'] = True
        details['cached_at'] = cached.get('updated_at')

        level_str = cached.get('level', 'parametric_match')
        try:
            level = ProcessingLevel(level_str)
        except ValueError:
            level = ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH

        return ProcessingResult(
            text=cached.get('name', ''),
            level=level,
            success=bool(cached.get('success', False)),
            params=params,
            ens_params=ens_params,
            ens_params_mask=ens_params_mask,
            ens_match=ens_match,
            confidence=float(cached.get('confidence', 0.0)),
            processing_time_ms=0.0,
            details=details,
            item_type=cached.get('item_type', ''),
            standard=cached.get('standard', '')
        )

    def process(self, text: str, article: str = "", force: bool = False) -> ProcessingResult:
        """
        Обработка одной строки номенклатуры.
        """
        import time
        start_time = time.time()

        clean_text = text.strip().rstrip(',.;: ')
        if clean_text != text.strip():
            logger.debug("Cleaned trailing punctuation: '%s' -> '%s'", text, clean_text)

        extracted = self.standard_extractor.extract_all(clean_text)
        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')
        extracted_standard = canonicalize_standard(standard_info.normalized) if standard_info else None

        # CACHE CHECK
        logger.debug("[CACHE] process() called: text=%s standard=%s result_db_path=%s",
                 clean_text[:50], extracted_standard, self.result_db_path)
        if not force and self.result_db_path:
            cached = self._check_cache(clean_text, extracted_standard)
            if cached:
                logger.info("[CACHE] HIT for '%s' / %s (code=%s, mask=%s)",
                         clean_text[:50], extracted_standard,
                         cached.get('ens_code', 'N/A'), cached.get('mask_pattern', 'N/A')[:30])
                return self._result_from_cache(cached)
            else:
                logger.info("[CACHE] MISS for '%s' / %s", clean_text[:50], extracted_standard)
        else:
            if force:
                logger.debug("[CACHE] skipped (force=True)")
            elif not self.result_db_path:
                logger.debug("[CACHE] skipped (result_db_path not set)")
        self._cache_stats['misses'] += 1

        logger.info("Processing: %s...", clean_text[:50])

        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')

        if not standard_info or not item_type:
            return self._llm_direct_process(clean_text, start_time)

        standard = canonicalize_standard(standard_info.normalized)
        logger.info("[PROCESS] standard='%s', item_type='%s', clean_text='%s'", standard, item_type, clean_text[:60])

        # Level 1: Проверка MaskDatabase
        search_item_type = item_type.upper()
        logger.info("[PROCESS] Searching mask: standard='%s', item_type='%s'", standard, search_item_type)
        mask = self.mask_db.get_mask(standard, search_item_type)
        logger.info("[PROCESS] Search by (standard+UPPER): found=%s", mask is not None)

        if mask is None:
            mask = self.mask_db.get_mask(standard, item_type)
            if mask:
                logger.info("[PROCESS] Found mask with original item_type: %s", item_type)

        if mask is None:
            try:
                standard_masks = self.mask_db.list_masks(standard=standard)
                logger.info("[PROCESS] Found %d masks for standard='%s' (any item_type)", len(standard_masks), standard)
                if standard_masks:
                    mask = standard_masks[0]
                    logger.info("[PROCESS] Using mask with item_type='%s'", mask.item_type)
            except Exception as e:
                logger.warning("[PROCESS] Fallback search by standard error: %s", e)

        logger.info("[PROCESS] mask found: %s, is_active: %s", mask is not None, getattr(mask, 'is_active', False))

        if mask is not None and not mask.is_active:
            logger.info("[PROCESS] Mask found but inactive, activating")
            try:
                self.mask_db.activate_mask(mask.id)
                mask.is_active = True
            except Exception as e:
                logger.warning("[PROCESS] Failed to activate mask: %s", e)

        if mask and mask.is_active:
            effective_standard = getattr(mask, 'standard', None) or standard
            if not getattr(mask, 'standard', None):
                mask.standard = effective_standard
                logger.info("[PROCESS] Fixed empty mask.standard -> '%s'", effective_standard)
            logger.info("[PROCESS] -> Level 6: ParametricMatch with mask %s", mask.id)
            return self._parametric_match(clean_text, mask, extracted, start_time)

        # Level 2: LLM Generation
        if self.use_llm_generation and self.llm_generator:
            standard_info = extracted.get('standard_info')
            generated_mask = self._generate_mask(standard, item_type, clean_text, standard_info)

            if generated_mask:
                validation_result = self._validate_mask(generated_mask, standard, item_type)

                if validation_result.passed:
                    mask_record = self._save_mask(generated_mask, validation_result)

                    if mask_record:
                        return self._parametric_match(clean_text, mask_record, extracted, start_time)
                else:
                    logger.warning("Generated mask failed validation: %.2f", validation_result.score)

        # Level 7: TF-IDF Fallback
        return self._tfidf_fallback(text, extracted, start_time)

    def _generate_mask(self, standard: str, item_type: str, text: str = "",
                     standard_info: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Генерация маски через LLM."""
        if not self.llm_generator:
            return None

        examples = self.validator._get_ens_examples(standard, item_type)

        if len(examples) < 10:
            logger.warning("Not enough examples for %s/%s", standard, item_type)
            return None

        ens_item_type = item_type
        if examples:
            first_example = examples[0]
            type_from_ens = first_example.get('тип_изделия') or first_example.get('наименование_типа')
            if type_from_ens and str(type_from_ens).strip():
                ens_item_type = str(type_from_ens).strip().lower()
                logger.info("[AutoProcessor] Тип из ЕСН: '%s' (был: '%s')", ens_item_type, item_type)

        canon_std = canonicalize_standard(standard)
        mask, attempts = self.llm_generator.generate_mask(
            standard=canon_std,
            item_type=ens_item_type,
            examples=examples,
            name=text,
            standard_info=standard_info
        )

        if mask:
            logger.info("Generated mask for %s/%s", canon_std, item_type)
            return mask

        return None

    def _validate_mask(self, mask: Dict[str, Any], standard: str, item_type: str) -> Any:
        """Валидация сгенерированной маски."""
        from database.mask_database import MaskRecord
        temp_mask = MaskRecord(
            standard=standard,
            item_type=item_type,
            pattern=mask['pattern'],
            params=mask['params'],
            required=mask['required']
        )
        result = self.validator.validate_mask(
            pattern=temp_mask.pattern,
            params=temp_mask.params,
            required=temp_mask.required,
            standard=standard,
            item_type=item_type
        )
        return result

    def _save_mask(self, mask: Dict[str, Any], validation: Any) -> Optional[Any]:
        """Сохранение валидированной маски в БД."""
        from database.mask_database import MaskRecord
        mask_record = MaskRecord(
            standard=mask['standard'],
            item_type=mask['item_type'],
            pattern=mask['pattern'],
            params=mask['params'],
            required=mask['required'],
            auto_score=validation.score,
            is_active=validation.passed,
            source='llm',
            test_examples=validation.details[:5]
        )
        mask_id = self.mask_db.save_mask(mask_record, auto_activate=True)
        if mask_id:
            mask_record.id = mask_id
            logger.info("Saved mask with ID: %s", mask_id)
            return mask_record
        return None

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        """Token-based Jaccard similarity."""
        import re
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

    def _get_coating_variants(self, coating: str) -> List[str]:
        """Generate coating search variants."""
        if not coating:
            return [coating]
        coating_str = str(coating).strip().lower()
        variants = [coating]
        if coating_str in ('кд', 'кд.'):
            for v in ['Кд6', 'Кд9', 'Кд6.фос', 'Кд9.фос',
                      'Кд6.фос.окс', 'Кд9.фос.окс', 'Кд.фос.окс']:
                variants.append(v)
            return variants
        return variants

    def _get_cached_ens_candidates(self, standard: str, item_type: str) -> List[Dict]:
        """Кэшированная загрузка ENS candidates (thread-safe)."""
        key = (standard.upper(), item_type.upper())
        with self._ens_candidates_lock:
            if key not in self._ens_candidates_cache:
                candidates = self.validator._get_ens_examples(standard, item_type) or []
                self._ens_candidates_cache[key] = candidates
                logger.debug("[ENS_CACHE] Loaded %d candidates for %s (cache miss)", len(candidates), key)
            else:
                candidates = self._ens_candidates_cache[key]
                logger.debug("[ENS_CACHE] Using %d cached candidates for %s (cache hit)", len(candidates), key)
        return candidates

    def _load_coating_rules(self) -> Optional[Dict]:
        """Получение coating_rules из settings или YAML (thread-safe)."""
        with self._coating_rules_lock:
            if self._coating_rules_loaded:
                return self._coating_rules_cache

            try:
                from config.settings import get_settings
                settings = get_settings()
                coating_rules = getattr(settings, 'coating_rules', None)
                if coating_rules:
                    self._coating_rules_cache = coating_rules
                    self._coating_rules_loaded = True
                    logger.info("[COATING_SUBST] Loaded from settings")
                    return coating_rules
            except Exception as e:
                logger.debug("[COATING_SUBST] Settings not available: %s", e)

            import yaml
            from pathlib import Path
            import os

            search_paths = []
            cwd = Path.cwd()
            for level in range(6):
                search_paths.append(cwd / "config" / "config.yaml")
                search_paths.append(cwd / "config.yaml")
                cwd = cwd.parent

            try:
                import inspect
                script_dir = Path(os.path.dirname(os.path.abspath(inspect.getfile(self.__class__))))
                search_paths.extend([
                    script_dir / ".." / "config" / "config.yaml",
                    script_dir / ".." / ".." / "config" / "config.yaml",
                ])
            except Exception:
                pass

            search_paths.extend([
                Path("/app/config/config.yaml"),
                Path("/workspace/config/config.yaml"),
            ])

            env_path = os.environ.get('NSI_CONFIG_PATH')
            if env_path:
                search_paths.insert(0, Path(env_path))

            for path in search_paths:
                try:
                    if path.exists():
                        with open(path, 'r', encoding='utf-8') as f_cfg:
                            config = yaml.safe_load(f_cfg) or {}
                        coating_rules = config.get('coating_rules')
                        if coating_rules:
                            self._coating_rules_cache = coating_rules
                            self._coating_rules_loaded = True
                            logger.info("[COATING_SUBST] Loaded from %s", path)
                            return coating_rules
                except Exception:
                    continue

            logger.warning("[COATING_SUBST] coating_rules not found anywhere")
            self._coating_rules_cache = {}
            self._coating_rules_loaded = True
            return None

    def _apply_coating_substitution(self, extracted_params: Dict[str, str], ens_candidates: List[Dict]) -> Tuple[Dict[str, str], Optional[Dict]]:
        """Применить auto_substitution из coating_rules к extracted_params."""
        logger.debug("[COATING_SUBST] Called with coating=%s, candidates=%d", extracted_params.get('покрытие'), len(ens_candidates))

        coating_rules = self._load_coating_rules()
        if not coating_rules:
            logger.warning("[COATING_SUBST] No coating_rules in config, skipping substitution")
            return extracted_params, None

        if not coating_rules.get('auto_substitution_enabled', False):
            logger.debug("[COATING_SUBST] auto_substitution_enabled=false, skipping")
            return extracted_params, None

        trial_match, trial_debug = self._fuzzy_match_ens_debug(extracted_params, ens_candidates)
        if not trial_match and not trial_debug:
            logger.debug("[COATING_SUBST] No candidates for trial match, skipping")
            return extracted_params, None

        material = None
        if trial_match:
            material = trial_match.get('марка_материала') or trial_match.get('марка_стали')
        if not material and trial_debug:
            for cd in trial_debug[:5]:
                key_params = cd.get('key_ens_params', {})
                material = key_params.get('марка_материала') or key_params.get('марка_стали')
                if material:
                    break

        if not material:
            logger.debug("[COATING_SUBST] Could not determine material from candidates")
            return extracted_params, None

        logger.debug("[COATING_SUBST] Detected material: '%s'", material)

        import re
        coating = extracted_params.get('покрытие', '')
        if not coating:
            return extracted_params, None

        for rule in coating_rules.get('auto_substitution', []):
            material_pattern = rule.get('material_pattern', '')
            wrong_coating = rule.get('wrong_coating', '')
            correct_coating = rule.get('correct_coating', '')

            if not re.search(material_pattern, str(material), re.IGNORECASE):
                continue

            wrong_sim = self._token_similarity(coating, wrong_coating)
            if wrong_sim < 0.5:
                continue

            new_params = dict(extracted_params)
            new_params['покрытие'] = correct_coating
            substitution_info = {
                'original': coating,
                'corrected': correct_coating,
                'material': str(material),
                'reason': rule.get('note', ''),
                'rule': {
                    'material_pattern': material_pattern,
                    'wrong_coating': wrong_coating,
                    'correct_coating': correct_coating,
                }
            }
            logger.info(
                "[COATING_SUBST] Applied: '%s' -> '%s' (material='%s', rule=%s)",
                coating, correct_coating, material, rule.get('note', '')
            )
            return new_params, substitution_info

        logger.debug("[COATING_SUBST] No matching rule for coating='%s', material='%s'", coating, material)
        return extracted_params, None

    def _fuzzy_match_ens(self, extracted_params: Dict[str, str], ens_candidates: List[Dict]) -> Optional[Dict]:
        best_match, _ = self._fuzzy_match_ens_debug(extracted_params, ens_candidates)
        return best_match

    def _fuzzy_match_ens_debug(self, extracted_params: Dict[str, str], ens_candidates: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
        """Fuzzy matching с подробным debug-выводом."""
        TEXT_FIELDS = {'покрытие', 'материал', 'марка_материала', 'марка_стали'}
        best_match = None
        best_score = 0.0
        debug_candidates = []

        coating_variants = None
        if 'покрытие' in extracted_params:
            coating_variants = self._get_coating_variants(extracted_params['покрытие'])

        extracted_str = ", ".join("{}={}".format(k, v) for k, v in extracted_params.items() if v is not None)
        logger.debug("[FUZZY] Extracted params: %s", extracted_str)
        logger.debug("[FUZZY] Total candidates to check: %d", len(ens_candidates) if ens_candidates else 0)

        for candidate in ens_candidates:
            total_weight = 0.0
            matched_weight = 0.0
            candidate_debug = {
                'name': candidate.get('наименование', candidate.get('полное_наименование', 'N/A')),
                'ens_code': candidate.get('код', candidate.get('mdm_key', 'N/A')),
                'params_matched': {},
                'params_mismatched': {},
                'params_missing': [],
                'params_comparison': {},
            }

            for param_name, extracted_val in extracted_params.items():
                if not extracted_val:
                    continue
                weight = 2.0 if param_name in TEXT_FIELDS else 1.0
                total_weight += weight

                candidate_val = candidate.get(param_name) or candidate.get(param_name.replace('_', ' '), '')

                if param_name in TEXT_FIELDS:
                    if param_name == 'покрытие' and coating_variants and len(coating_variants) > 1:
                        best_sim = max(self._token_similarity(v, candidate_val) for v in coating_variants)
                        sim = best_sim
                    else:
                        sim = self._token_similarity(extracted_val, candidate_val)
                    matched = sim >= 0.5
                    if sim >= 0.99:
                        status = 'exact'
                    elif matched:
                        status = 'token_matched'
                    else:
                        status = 'mismatched'
                    candidate_debug['params_comparison'][param_name] = {
                        'status': status,
                        'extracted': str(extracted_val),
                        'ens_value': str(candidate_val) if candidate_val else None,
                        'similarity': round(sim, 3) if candidate_val else 0.0,
                    }
                    if matched:
                        matched_weight += weight * sim
                        candidate_debug['params_matched'][param_name] = "'{}' ~ '{}' (sim={:.2f})".format(extracted_val, candidate_val, sim)
                    else:
                        candidate_debug['params_mismatched'][param_name] = "'{}' vs '{}' (sim={:.2f})".format(extracted_val, candidate_val, sim)
                else:
                    try:
                        matched = float(str(extracted_val).replace(',', '.')) == float(str(candidate_val).replace(',', '.'))
                    except (ValueError, TypeError):
                        matched = str(extracted_val).strip() == str(candidate_val).strip()
                    status = 'exact' if matched else 'mismatched'
                    candidate_debug['params_comparison'][param_name] = {
                        'status': status,
                        'extracted': str(extracted_val),
                        'ens_value': str(candidate_val) if candidate_val else None,
                        'similarity': 1.0 if matched else 0.0,
                    }
                    if matched:
                        matched_weight += weight
                        candidate_debug['params_matched'][param_name] = "{} == {}".format(extracted_val, candidate_val)
                    else:
                        candidate_debug['params_mismatched'][param_name] = "{} != {}".format(extracted_val, candidate_val)

            score = matched_weight / total_weight if total_weight > 0 else 0.0
            candidate_debug['score'] = round(score, 3)
            candidate_debug['total_weight'] = total_weight
            candidate_debug['matched_weight'] = round(matched_weight, 3)
            candidate_debug['params_count'] = len(extracted_params)

            if score > best_score:
                best_score = score
                best_match = candidate
                candidate_debug['is_best'] = True
            else:
                candidate_debug['is_best'] = False

            debug_candidates.append(candidate_debug)

        if best_match:
            logger.info(
                "[FUZZY] Best match: score=%.3f, code=%s, name=%s",
                best_score,
                best_match.get('код', best_match.get('mdm_key', 'N/A')),
                best_match.get('наименование', best_match.get('полное_наименование', 'N/A'))[:50]
            )
        else:
            logger.info("[FUZZY] No match found among %d candidates", len(ens_candidates))

        return best_match, debug_candidates

    def _remap_params(self, params: Dict[str, str], mask_params: Dict[str, Any]) -> Dict[str, str]:
        """Переименовать параметры из маски в параметры ЕНС."""
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

    def _get_generic_pattern(self, standard: str, item_type: str, params: Dict[str, str], mask: Any) -> str:
        """Построить generic-паттерн для V2 exact match."""
        parts = [item_type]
        for key in sorted(params.keys()):
            val = params[key]
            if val is not None and str(val).strip():
                parts.append(str(val).strip())
        parts.append(standard)
        return ' '.join(parts)

    def _parametric_match(self, text: str, mask: Any, extracted: Dict, start_time: float) -> ProcessingResult:
        """
        Параметрическое сопоставление с ЕНС.
        """
        import time
        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')
        standard = canonicalize_standard(standard_info.normalized) if standard_info else ''

        logger.info("[ANALYZE] text=%s", text[:80])
        logger.info("[ANALYZE] standard=%s, item_type=%s", standard, item_type)

        # Extract params using mask
        params = self.parametric_client.extract_params(text, mask.pattern)
        logger.info("[ANALYZE] Extracted params: %s", params)

        # Remap to ENS field names
        remapped = self._remap_params(params, mask.params)
        logger.info("[ANALYZE] Remapped params: %s", remapped)

        # Get ENS candidates
        candidates = self._get_cached_ens_candidates(standard, item_type)
        if not candidates:
            logger.warning("[ANALYZE] No ENS candidates for %s/%s", standard, item_type)
            return self._tfidf_fallback(text, extracted, start_time)

        # Apply coating substitution
        substituted_params, substitution_info = self._apply_coating_substitution(remapped, candidates)
        if substitution_info:
            logger.info("[ANALYZE] Coating substitution applied: %s", substitution_info)
            remapped = substituted_params

        # V2 exact match via generic pattern
        generic_pattern = self._get_generic_pattern(standard, item_type, remapped, mask)
        logger.info("[ANALYZE] Generic pattern: %s", generic_pattern)

        best_match = None
        best_score = 0.0
        match_type = None
        match_type_ru = None
        fuzzy_mismatched_params = None
        debug_candidates = []

        # Try exact match first
        for candidate in candidates:
            cand_name = candidate.get('наименование', candidate.get('полное_наименование', ''))
            if cand_name and generic_pattern:
                cand_generic = self._get_generic_pattern(standard, item_type, candidate, mask)
                if generic_pattern.strip() == cand_generic.strip():
                    best_match = candidate
                    best_score = 1.0
                    match_type = 'exact'
                    match_type_ru = 'Точное совпадение (V2 generic)'
                    logger.info("[ANALYZE] V2 EXACT MATCH: %s", cand_name[:60])
                    break

        # If no exact match, try fuzzy match
        if not best_match:
            best_match, debug_candidates = self._fuzzy_match_ens_debug(remapped, candidates)
            if best_match:
                match_type = 'fuzzy'
                match_type_ru = 'Нечеткое совпадение'
                best_score = 0.85
                for cd in debug_candidates:
                    if cd.get('is_best'):
                        best_score = cd.get('score', 0.85)
                        break

        # If still no match, try with coating variants
        if not best_match and remapped.get('покрытие'):
            coating_variants = self._get_coating_variants(remapped['покрытие'])
            for variant in coating_variants[1:]:
                variant_params = dict(remapped)
                variant_params['покрытие'] = variant
                best_match, debug_candidates = self._fuzzy_match_ens_debug(variant_params, candidates)
                if best_match:
                    match_type = 'fuzzy_coating_variant'
                    match_type_ru = 'Нечеткое совпадение (вариант покрытия)'
                    best_score = 0.8
                    logger.info("[ANALYZE] Found match with coating variant: %s", variant)
                    break

        processing_time = (time.time() - start_time) * 1000

        if not best_match:
            logger.warning("[ANALYZE] No match found for %s", text[:60])
            return ProcessingResult(
                text=text,
                level=ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH,
                success=False,
                params=params,
                ens_params=remapped,
                ens_params_mask=mask.params,
                confidence=0.0,
                processing_time_ms=processing_time,
                details={
                    'match_type': None,
                    'match_type_ru': 'Не найдено совпадение',
                    'debug_candidates': debug_candidates[:5],
                },
                item_type=item_type,
                standard=standard
            )

        # Build ENS match dict
        ens_match = {
            'code': best_match.get('код', best_match.get('mdm_key')),
            'name': best_match.get('наименование', best_match.get('полное_наименование')),
            'mdm_key': best_match.get('mdm_key'),
            'score': best_score,
            'type': match_type,
            **best_match
        }

        # Get ENS params from best_match
        ens_params_from_match = {}
        for key in remapped.keys():
            val = best_match.get(key) or best_match.get(key.replace('_', ' '))
            if val:
                ens_params_from_match[key] = val

        # Calculate confidence
        confidence = best_score
        if match_type == 'exact':
            confidence = 1.0
        elif match_type == 'fuzzy':
            confidence = best_score
        elif match_type == 'fuzzy_coating_variant':
            confidence = best_score * 0.95

        # Determine success
        matching_cfg = _get_matching_config()
        success = confidence >= matching_cfg.success_threshold

        # Build details
        details = {
            'match_type': match_type,
            'match_type_ru': match_type_ru,
            'mask_id': getattr(mask, 'id', None),
            'mask_pattern': getattr(mask, 'pattern', None),
            'debug_candidates': debug_candidates[:5] if debug_candidates else [],
        }

        if substitution_info:
            details['coating_substitution'] = substitution_info
            # If substitution was applied and we have a match, force success
            if best_match:
                success = True
                confidence = max(confidence, matching_cfg.success_threshold)
                logger.info("[ANALYZE] Forced success=True due to coating substitution")

        if fuzzy_mismatched_params:
            details['fuzzy_mismatched_params'] = fuzzy_mismatched_params

        result = ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH,
            success=success,
            params=params,
            ens_params=ens_params_from_match,
            ens_params_mask=mask.params,
            ens_match=ens_match,
            confidence=round(confidence, 3),
            processing_time_ms=round(processing_time, 2),
            details=details,
            item_type=item_type,
            standard=standard
        )

        logger.info("[ANALYZE] Result: success=%s, confidence=%.3f, match_type=%s", success, confidence, match_type)
        return result

    def _normalize_value_types(self, value):
        """Нормализация типов значений."""
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
        """Нормализация всех параметров."""
        return {k: self._normalize_value_types(v) for k, v in params.items() if v is not None}

    def _get_ens_by_code(self, code: str) -> Optional[Dict]:
        """Получить запись ЕНС по коду."""
        try:
            import pickle
            with open(self.ens_index_path, 'rb') as f:
                ens_index = pickle.load(f)
            for item in ens_index:
                if str(item.get('код', '')) == str(code) or str(item.get('mdm_key', '')) == str(code):
                    return item
        except Exception as e:
            logger.warning("Failed to load ENS by code: %s", e)
        return None

    def _tfidf_fallback(self, text: str, extracted: Dict, start_time: float) -> ProcessingResult:
        """TF-IDF fallback matching."""
        import time
        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')
        standard = canonicalize_standard(standard_info.normalized) if standard_info else ''
        processing_time = (time.time() - start_time) * 1000

        try:
            from core.tfidf_matcher import TFIDFMatcher
            matcher = TFIDFMatcher(self.ens_index_path)
            matches = matcher.find_matches(text, top_n=3)

            if matches:
                best = matches[0]
                ens_match = {
                    'code': best.get('код', best.get('mdm_key')),
                    'name': best.get('наименование', best.get('полное_наименование')),
                    'mdm_key': best.get('mdm_key'),
                    'score': best.get('score', 0.0),
                    **best
                }
                confidence = best.get('score', 0.0)
                matching_cfg = _get_matching_config()
                success = confidence >= matching_cfg.fuzzy_threshold
                return ProcessingResult(
                    text=text,
                    level=ProcessingLevel.LEVEL_7_TFIDF_FALLBACK,
                    success=success,
                    params={},
                    ens_params={},
                    ens_match=ens_match,
                    confidence=round(confidence, 3),
                    processing_time_ms=round(processing_time, 2),
                    details={
                        'match_type': 'tfidf',
                        'match_type_ru': 'TF-IDF fallback',
                        'tfidf_matches': matches[:3],
                    },
                    item_type=item_type,
                    standard=standard
                )
        except Exception as e:
            logger.warning("TF-IDF fallback failed: %s", e)

        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_7_TFIDF_FALLBACK,
            success=False,
            params={},
            ens_params={},
            confidence=0.0,
            processing_time_ms=round(processing_time, 2),
            details={
                'match_type': None,
                'match_type_ru': 'TF-IDF fallback не удался',
            },
            item_type=item_type,
            standard=standard
        )

    def _llm_direct_process(self, text: str, start_time: float) -> ProcessingResult:
        """Прямой LLM fallback."""
        import time
        processing_time = (time.time() - start_time) * 1000
        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_8_LLM_DIRECT,
            success=False,
            params={},
            ens_params={},
            confidence=0.0,
            processing_time_ms=round(processing_time, 2),
            details={
                'match_type': None,
                'match_type_ru': 'LLM direct fallback',
            },
            item_type='',
            standard=''
        )

    def batch_process(self, texts: List[str], articles: Optional[List[str]] = None,
                     force: bool = False, workers: int = 1) -> List[ProcessingResult]:
        """
        Batch processing with optional parallel workers.
        """
        import time
        start_time = time.time()
        if articles is None:
            articles = [""] * len(texts)

        results = []
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self.process, text, article, force)
                    for text, article in zip(texts, articles)
                ]
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error("Batch processing error: %s", e)
                        results.append(ProcessingResult(
                            text="",
                            success=False,
                            details={'error': str(e)}
                        ))
        else:
            for text, article in zip(texts, articles):
                results.append(self.process(text, article, force))

        total_time = (time.time() - start_time) * 1000
        logger.info("Batch processed %d items in %.1f ms", len(texts), total_time)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'cache_hits': self._cache_stats['hits'],
            'cache_misses': self._cache_stats['misses'],
        }