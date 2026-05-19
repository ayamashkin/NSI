#!/usr/bin/env python3
"""
Main Processor Module
Интеграция всех уровней: StandardExtractor -> MaskDatabase -> LLM Generator ->
AutoValidator -> ParametricMatch -> TF-IDF Fallback

VERSION: 2026-05-19

LAST_FIXES:
 2026-05-19 21:45 UTC+3 — RESTORED from b5ae1c32 (safe commit).
   + ProcessingResult.to_dict() with ens_code/ens_name/level string/match_type_ru
   + result.db caching restored
   + ThreadPoolExecutor support preserved
   + JSON/Excel dual output structure
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

# Lazy import для избежания circular dependency
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
        """Полная сериализация для JSON (как в эталонном results — копия.json)."""
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

        # Загружаем skip_fields из settings и передаём в ParametricENSClient
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
            if cached:
                # Проверяем TTL
                updated_at = cached.get('updated_at')
                if updated_at:
                    try:
                        last_update = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                        if datetime.now() - last_update < timedelta(days=self.cache_ttl_days):
                            self._cache_stats['hits'] += 1
                            return cached
                    except Exception:
                        # Если не удалось распарсить дату, всё равно используем кэш
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
        # Десериализация JSON-полей
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
        # Добавляем информацию о кэше
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
        Поддерживает кэширование через result.db.

        Args:
            text: Строка номенклатуры
            article: Артикул (для ключа кэша)
            force: Принудительный пересчёт (игнорировать кэш)

        Returns:
            ProcessingResult
        """
        import time
        start_time = time.time()

        # === CACHE CHECK ===
        logger.debug("[CACHE] process() called: article=%r text=%s result_db_path=%s", article, text[:50], self.result_db_path)
        if not force and self.result_db_path:
            cached = self._check_cache(article, text)
            if cached:
                logger.info("[CACHE] HIT for '%s' (code=%s, mask=%s)",
                            text[:50], cached.get('ens_code', 'N/A'), cached.get('mask_pattern', 'N/A')[:30])
                return self._result_from_cache(cached)
            else:
                logger.debug("[CACHE] MISS for article=%r text=%s", article, text[:50])
        else:
            if force:
                logger.debug("[CACHE] skipped (force=True)")
            elif not self.result_db_path:
                logger.debug("[CACHE] skipped (result_db_path not set)")
        self._cache_stats['misses'] += 1

        clean_text = text.strip().rstrip(',.;: ')
        if clean_text != text.strip():
            logger.debug("Cleaned trailing punctuation: '%s' -> '%s'", text, clean_text)

        logger.info("Processing: %s...", text[:50])

        # Level 0: Извлечение стандарта
        extracted = self.standard_extractor.extract_all(clean_text)
        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')

        if not standard_info or not item_type:
            return self._llm_direct_process(clean_text, start_time)

        standard = standard_info.normalized
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

        mask, attempts = self.llm_generator.generate_mask(
            standard=standard,
            item_type=ens_item_type,
            examples=examples,
            name=text,
            standard_info=standard_info
        )

        if mask:
            logger.info("Generated mask for %s/%s", standard, item_type)
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
                        with open(path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f) or {}
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

            key_ens_params = {}
            for key_field in ['длина', 'номинальный_диаметр_резьбы', 'исполнение', 'покрытие', 'марка_материала', 'шаг_резьбы', 'тип_резьбы']:
                if key_field in candidate and candidate[key_field] is not None and str(candidate[key_field]).strip():
                    key_ens_params[key_field] = candidate[key_field]
            candidate_debug['key_ens_params'] = key_ens_params

            if total_weight > 0:
                score = matched_weight / total_weight
                candidate_debug['score'] = round(score, 3)
                candidate_debug['weight'] = round(total_weight, 1)
                candidate_debug['matched_weight'] = round(matched_weight, 1)
                logger.debug("[FUZZY] Candidate '%s': score=%.3f, weight=%.1f, matched=%.1f", candidate_debug['name'][:50], score, total_weight, matched_weight)
                if _get_matching_config().debug_per_parameter:
                    ens_params_str = ", ".join("{}={}".format(k, v) for k, v in key_ens_params.items())
                    logger.debug("[FUZZY] ENS params: %s", ens_params_str)
                    if candidate_debug['params_matched']:
                        matched_str = ", ".join("{}: {}".format(k, v) for k, v in candidate_debug['params_matched'].items())
                        logger.debug("[FUZZY] MATCHED: %s", matched_str)
                    if candidate_debug['params_mismatched']:
                        mismatched_str = ", ".join("{}: {}".format(k, v) for k, v in candidate_debug['params_mismatched'].items())
                        logger.debug("[FUZZY] MISMATCHED: %s", mismatched_str)
                if score > best_score:
                    best_score = score
                    best_match = {**candidate, '_fuzzy_score': best_score}
            else:
                candidate_debug['score'] = 0.0
                candidate_debug['reason'] = 'no comparable params (weight=0)'
                logger.debug("[FUZZY] Candidate '%s': no comparable params (weight=0)", candidate_debug['name'][:50])

            debug_candidates.append(candidate_debug)

        debug_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

        if debug_candidates and _get_matching_config().debug_per_parameter:
            top_n = min(5, len(debug_candidates))
            logger.debug("[FUZZY] Top %d candidates:", top_n)
            for i, cd in enumerate(debug_candidates[:top_n], 1):
                logger.debug("[FUZZY] #%d: '%s' score=%s, code=%s", i, cd.get('name', '')[:50], cd.get('score', 0), cd.get('ens_code', 'N/A'))
                if cd.get('params_mismatched'):
                    for pk, pv in cd['params_mismatched'].items():
                        logger.debug("[FUZZY] mismatch: %s: %s", pk, pv)
            logger.debug("[FUZZY] Best score: %.3f, threshold: 0.6, matched: %s", best_score, best_match is not None and best_score >= 0.6)
        return (best_match if best_score >= 0.6 else None), debug_candidates

    def _remap_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Переименование параметров — теперь pass-through."""
        return dict(params) if params else params

    def _get_generic_pattern(self, item_type: str, standard: str = None) -> Optional[str]:
        """Генерация generic паттерна для item_type."""
        _ru_t = chr(0x0442)
        _ru_b = chr(0x0431)
        _ru_v = 'В'

        item_upper = (item_type or '').upper()

        if item_upper in ('БОЛТ', 'БОЛ' + _ru_t, 'BOLT'):
            if standard and '7795' in standard:
                return (
                    r'^Болт\s*(?:(?P<исполнение>\d+)\s*)?'
                    r'(?:M(?P<номинальный_диаметр_резьбы>\d+)'
                    r'(?:[xX×](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?'
                    r'[-\s]*(?P<класс_поле_допуска>\d+[a-zA-Z])'
                    r'[xX×](?P<длина>\d+)\.(?P<группа_класс_прочности>\d+(?:\.\d+)?))'
                    r'(?:[-\s]*(?P<покрытие>[\w.]+))?'
                    r'\s*ГОСТ\s*7795-70\s*$'
                )
            pattern = (
                r'^Болт\s*(?:\((?P<исполнение>\d+)\)[-\s]*)?'
                r'(?P<номинальный_диаметр_резьбы>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<длина>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<покрытие>[\w.]+)'
            )
            if standard:
                pattern += r'(?:[-\s]*.*)?'
            else:
                pattern += r'(?:\s+.*)?'
            return pattern

        if item_upper in ('ВИНТ', 'ВИН' + _ru_t, 'SCREW'):
            pattern = (
                r'^Винт\s*(?:\((?P<исполнение>\d+)\)[-\s]*)?'
                r'(?P<номинальный_диаметр_резьбы>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<длина>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<покрытие>[\w.]+)'
            )
            if standard:
                pattern += r'(?:[-\s]*.*)?'
            else:
                pattern += r'(?:\s+.*)?'
            return pattern

        if item_upper in ('ШАЙБА', 'ШАЙ' + _ru_b + chr(0x0430), 'WASHER'):
            pattern = (
                r'^Шайба\s*(?:(?P<тип>[A-Z])\s*)?'
                r'(?P<диаметр>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<наружный_диаметр>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<толщина>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<покрытие>[\w.]+)'
            )
            if standard:
                pattern += r'(?:[-\s]*.*)?'
            else:
                pattern += r'(?:\s+.*)?'
            return pattern

        if item_upper in ('ГАЙКА', 'ГАЙКА', 'NUT'):
            pattern = (
                r'^Гайка\s*(?:(?P<исполнение>\d+)\s*)?'
                r'(?P<номинальный_диаметр_резьбы>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<покрытие>[\w.]+)'
            )
            if standard:
                pattern += r'(?:[-\s]*.*)?'
            else:
                pattern += r'(?:\s+.*)?'
            return pattern

        return None

    def _parametric_match(
        self,
        text: str,
        mask,
        extracted: Dict[str, Any],
        start_time: float
    ) -> ProcessingResult:
        """Параметрическое сопоставление."""
        import time

        substitution_info: Optional[Dict] = None

        extracted_std = None
        if extracted.get('standard_info'):
            extracted_std = extracted['standard_info'].normalized
        effective_standard = getattr(mask, 'standard', None) or extracted_std or ''
        if not getattr(mask, 'standard', None) and extracted_std:
            mask.standard = extracted_std
            logger.info("[PARAM_MATCH] Fixed mask.standard from extracted: '%s'", extracted_std)

        logger.debug("[PARAM_MATCH] text='%s', mask.pattern='%s...', mask.standard='%s', mask.item_type='%s'",
                     text[:50], mask.pattern[:50], effective_standard, mask.item_type)

        match_result = self.parametric_client.match(
            text=text,
            standard=effective_standard,
            item_type=mask.item_type
        )

        logger.debug("[PARAM_MATCH] score=%s, matched_params=%s, ens_code=%s",
                     match_result.score, match_result.matched_params, match_result.ens_code)

        fallback_params = None
        if not match_result.matched_params and match_result.score > 0:
            import re
            try:
                relaxed_for_fallback = self.parametric_client._relax_pattern(mask.pattern, standard=effective_standard)
                logger.debug("[PARAM_MATCH] Fallback pattern: %s", relaxed_for_fallback[:200])
                m = re.search(relaxed_for_fallback, text, re.IGNORECASE)
                if m:
                    fallback_params = {k: v for k, v in m.groupdict().items() if v is not None}
                    if fallback_params:
                        logger.info("[PARAM_MATCH] Fallback extraction: %s", fallback_params)
                    else:
                        logger.warning("[PARAM_MATCH] Fallback regex did NOT match. Pattern: %s", relaxed_for_fallback[:120])
                else:
                    logger.warning("[PARAM_MATCH] Fallback regex did NOT match. Pattern: %s", relaxed_for_fallback[:120])
            except Exception as e:
                logger.warning("[PARAM_MATCH] Fallback extraction error: %s", e)

        if not fallback_params:
            import re
            try:
                generic_pattern = self._get_generic_pattern(mask.item_type, effective_standard)
                if generic_pattern:
                    relaxed_generic = self.parametric_client._relax_pattern(generic_pattern, standard=effective_standard)
                    logger.debug("[PARAM_MATCH] Generic pattern: %s", relaxed_generic[:200])
                    m = re.search(relaxed_generic, text, re.IGNORECASE)
                    if m:
                        fallback_params = {k: v for k, v in m.groupdict().items() if v is not None}
                        if fallback_params:
                            logger.info("[PARAM_MATCH] Generic extraction: %s", fallback_params)
                        else:
                            logger.debug("[PARAM_MATCH] Generic pattern did NOT match")
                    else:
                        logger.debug("[PARAM_MATCH] Generic pattern did NOT match")
            except Exception as e:
                logger.warning("[PARAM_MATCH] Generic fallback error: %s", e)

        relaxed_result = None
        if match_result.score == 0 and not match_result.matched_params:
            try:
                relaxed_pattern = self.parametric_client._relax_pattern(mask.pattern, standard=effective_standard)
                if relaxed_pattern != mask.pattern:
                    relaxed_mask = type(mask)(
                        id=getattr(mask, 'id', -1),
                        standard=effective_standard,
                        item_type=mask.item_type,
                        pattern=relaxed_pattern,
                        params=mask.params,
                        required=mask.required,
                        auto_score=getattr(mask, 'auto_score', 0),
                        is_active=True,
                        source='relaxed',
                        usage_count=0,
                        test_examples='[]',
                        created_at='',
                        last_used='',
                        pattern_hash=''
                    )
                    relaxed_result = self.parametric_client.match(
                        text=text,
                        standard=effective_standard,
                        item_type=mask.item_type,
                        pattern=relaxed_pattern
                    )
                    if relaxed_result.score > 0:
                        logger.info("[PARAM_MATCH] Relaxed pattern matched: score=%s", relaxed_result.score)
                        match_result = relaxed_result
            except Exception as e:
                logger.debug("[PARAM_MATCH] Relaxed pattern error: %s", e)

        fuzzy_ens_code = None
        fuzzy_score = 0.0
        fuzzy_mismatched_params = None
        fuzzy_debug = []

        final_matched_params = fallback_params if fallback_params else match_result.matched_params

        if final_matched_params:
            final_matched_params = self._remap_params(final_matched_params)
            logger.debug("[PARAM_MATCH] Remapped params: %s", final_matched_params)

        raw_params = fallback_params if fallback_params else match_result.matched_params
        logger.debug("[PARAM_MATCH] Coating substitution check: raw_params=%s", raw_params)
        if raw_params and raw_params.get('покрытие'):
            logger.debug("[PARAM_MATCH] Coating substitution: has покрытие='%s', calling _apply_coating_substitution", raw_params.get('покрытие'))
            try:
                ens_candidates_sub = self._get_cached_ens_candidates(effective_standard, mask.item_type)
                logger.debug("[PARAM_MATCH] Coating substitution: got %d candidates", len(ens_candidates_sub))
                if ens_candidates_sub:
                    _, substitution_info = self._apply_coating_substitution(raw_params, ens_candidates_sub)
                    if substitution_info:
                        logger.info("[PARAM_MATCH] Coating substitution result: %s", substitution_info)
                    else:
                        logger.debug("[PARAM_MATCH] Coating substitution: no substitution applied")
                else:
                    logger.debug("[PARAM_MATCH] Coating substitution: no candidates, skipping")
            except Exception as e:
                logger.warning("[PARAM_MATCH] Coating substitution error: %s", e, exc_info=True)
        else:
            logger.debug("[PARAM_MATCH] Coating substitution: no покрытие in raw_params, skipping")

        debug_candidates = []

        if match_result.score < 0.7 or not match_result.ens_code:
            try:
                ens_candidates = self._get_cached_ens_candidates(effective_standard, mask.item_type)
                logger.info("[PARAM_MATCH] ENS candidates: count=%d, standard=%s, item_type=%s",
                            len(ens_candidates), effective_standard, mask.item_type)

                search_params = final_matched_params
                if substitution_info and ens_candidates:
                    search_params = dict(final_matched_params)
                    search_params['покрытие'] = substitution_info['corrected']
                    logger.info("[PARAM_MATCH] Using substituted params for search: %s", search_params)

                if search_params:
                    clean_params = {k: v for k, v in search_params.items() if v is not None}
                    manual_ens, find_debug = self.parametric_client._find_in_ens_debug(
                        clean_params,
                        list(clean_params.keys()),
                        standard=effective_standard,
                        text=text,
                        item_type=mask.item_type if mask else None
                    )
                    if find_debug:
                        debug_candidates.extend(find_debug[:10])
                    ens_code_field = manual_ens.get('код') or manual_ens.get('code') if manual_ens else None
                    if manual_ens and ens_code_field:
                        fuzzy_ens_code = ens_code_field
                        fuzzy_score = manual_ens.get('_match_score', 0.8)
                        if not match_result.ens_name:
                            match_result.ens_name = manual_ens.get('наименование') or manual_ens.get('name')
                        logger.info("[PARAM_MATCH] Exact ENS search matched: ens_code=%s, score=%.2f", fuzzy_ens_code, fuzzy_score)

                if not fuzzy_ens_code and ens_candidates and search_params:
                    logger.info("[PARAM_MATCH] Trying fuzzy match with params: %s", search_params)

                    fuzzy_match, fuzzy_debug = self._fuzzy_match_ens_debug(search_params, ens_candidates)
                    if fuzzy_debug:
                        debug_candidates.extend(fuzzy_debug[:10])

                    if not fuzzy_match and search_params != final_matched_params:
                        logger.info("[PARAM_MATCH] Substituted params didn't match, trying original: %s", final_matched_params)
                        fuzzy_match, fuzzy_debug = self._fuzzy_match_ens_debug(final_matched_params, ens_candidates)
                        if fuzzy_debug:
                            debug_candidates.extend(fuzzy_debug[:10])

                    if fuzzy_match:
                        fuzzy_ens_code = fuzzy_match.get('код') or fuzzy_match.get('mdm_key')
                        fuzzy_score = fuzzy_match.get('_fuzzy_score', 0.0)
                        logger.info("[PARAM_MATCH] Fuzzy fallback matched: score=%.2f, ens_code=%s", fuzzy_score, fuzzy_ens_code)
                    else:
                        logger.warning("[PARAM_MATCH] Fuzzy fallback: no match above threshold 0.6")

                    if fuzzy_debug:
                        best_candidate = fuzzy_debug[0]
                        if best_candidate.get('params_mismatched'):
                            fuzzy_mismatched_params = dict(best_candidate['params_mismatched'])
                            logger.info("[PARAM_MATCH] Fuzzy mismatched params: %s", fuzzy_mismatched_params)
                        else:
                            fuzzy_mismatched_params = {}
                elif not ens_candidates:
                    logger.warning("[PARAM_MATCH] Fuzzy fallback: no ENS candidates found")
                elif not search_params:
                    logger.warning("[PARAM_MATCH] Fuzzy fallback: empty search_params")
            except Exception as e:
                logger.warning("[PARAM_MATCH] Fuzzy fallback error: %s", e)

        final_ens_code = match_result.ens_code or fuzzy_ens_code
        final_ens_name = match_result.ens_name
        final_mdm_key = match_result.mdm_key if match_result.ens_code else None

        if final_ens_code and not final_ens_name:
            try:
                ens_item = self._get_ens_by_code(final_ens_code)
                if ens_item:
                    final_ens_name = ens_item.get('наименование') or ens_item.get('name')
                    if not final_mdm_key:
                        final_mdm_key = ens_item.get('mdm_key')
                    logger.info("[PARAM_MATCH] ENS name resolved: '%s'", final_ens_name)
            except Exception as e:
                logger.info("[PARAM_MATCH] Failed to resolve ENS name: %s", e)

        generic_params = None
        if mask:
            try:
                import re
                generic = self._get_generic_pattern(mask.item_type, effective_standard)
                if generic:
                    m = re.search(generic, text, re.IGNORECASE)
                    if m:
                        generic_params = {k: v for k, v in m.groupdict().items() if v is not None}
                        generic_params = self._normalize_params(generic_params)
                        logger.info("[PARAM_MATCH] Params from generic on text: %s", generic_params)
                        if not final_matched_params:
                            final_matched_params = generic_params
            except Exception as e:
                logger.debug("[PARAM_MATCH] Generic parse on text error: %s", e)

        v2_score = 0.0
        v2_match_type = None
        v2_computed = False
        params_for_v2 = final_matched_params or generic_params
        if params_for_v2 and (final_ens_name or final_ens_code):
            if not final_ens_name and final_ens_code:
                try:
                    ens_item = self._get_ens_by_code(final_ens_code)
                    if ens_item:
                        final_ens_name = ens_item.get('наименование') or ens_item.get('name')
                        logger.info("[PARAM_MATCH] ENS name resolved (V2): '%s'", final_ens_name)
                except Exception as e:
                    logger.info("[PARAM_MATCH] ENS name resolution failed (V2): %s", e)
            try:
                import re
                generic = self._get_generic_pattern(mask.item_type, effective_standard)
                if generic and final_ens_name:
                    ens_name_str = str(final_ens_name)
                    logger.info("[PARAM_MATCH] V2 trying generic on ENS name: '%s', pattern: %s...", ens_name_str[:60], generic[:80])
                    m = re.search(generic, ens_name_str, re.IGNORECASE)
                    if m:
                        generic_ens_mask = {k: v for k, v in m.groupdict().items() if v is not None}
                        logger.info("[PARAM_MATCH] Generic parsed ENS name: %s", generic_ens_mask)
                        v2_score, v2_match_type, v2_details = self.parametric_client._calculate_match_score_v2(
                            text=text,
                            ens_name=final_ens_name,
                            params=params_for_v2,
                            ens_params={},
                            ens_params_mask=generic_ens_mask,
                            required=list(params_for_v2.keys())
                        )
                        v2_computed = True
                        logger.info("[PARAM_MATCH] V2 score: %s, type: %s", v2_score, v2_match_type)
                    else:
                        logger.info("[PARAM_MATCH] V2: generic pattern did NOT match ENS name")
                elif not final_ens_name:
                    logger.info("[PARAM_MATCH] V2 skipped: no ENS name for code %s", final_ens_code)
                elif not generic:
                    logger.info("[PARAM_MATCH] V2 skipped: no generic pattern for %s", mask.item_type)
            except Exception as e:
                logger.info("[PARAM_MATCH] V2 scoring error: %s", e)

        if v2_computed:
            if v2_score >= _get_matching_config().v2_exact_threshold:
                final_score = v2_score
                if v2_match_type:
                    match_result.match_type = v2_match_type
            else:
                final_score = max(match_result.score, fuzzy_score)
        else:
            final_score = max(match_result.score, fuzzy_score)

        # Получаем параметры из ENS записи — используем skip_fields из parametric_client
        ens_params_from_index = None
        if final_ens_code:
            try:
                ens_item = self._get_ens_by_code(final_ens_code)
                if ens_item:
                    import math
                    # Используем skip_fields из parametric_client (загружено из конфига)
                    skip_fields = getattr(self.parametric_client, '_skip_fields', set())
                    base_skip = {'_match_score', '_match_type', 'item_type', 'standard'}
                    skip_fields = skip_fields | base_skip

                    ens_params_from_index = {}
                    for k, v in ens_item.items():
                        if k.startswith('_'):
                            continue
                        if k in skip_fields:
                            continue
                        if v is None or v == '':
                            continue
                        if isinstance(v, float) and math.isnan(v):
                            continue
                        if isinstance(v, list):
                            continue
                        ens_params_from_index[k] = v
                    logger.debug("[PARAM_MATCH] ENS params from index: %s", ens_params_from_index)
            except Exception as e:
                logger.warning("[PARAM_MATCH] Failed to get ENS params: %s", e)

        if final_matched_params:
            final_matched_params = self._normalize_params(final_matched_params)
            logger.debug("[PARAM_MATCH] Normalized params: %s", final_matched_params)
        if ens_params_from_index:
            ens_params_from_index = self._normalize_params(ens_params_from_index)
            logger.debug("[PARAM_MATCH] Normalized ens_params: %s", ens_params_from_index)

        ens_params_mask = match_result.ens_params_mask if hasattr(match_result, "ens_params_mask") else {}
        if not ens_params_mask and final_ens_name:
            if mask and mask.pattern:
                try:
                    import re
                    relaxed_for_ens = self.parametric_client._relax_pattern(mask.pattern, standard=effective_standard)
                    m = re.search(relaxed_for_ens, str(final_ens_name), re.IGNORECASE)
                    if m:
                        ens_params_mask = {k: v for k, v in m.groupdict().items() if v is not None}
                        logger.debug("[PARAM_MATCH] ENS params mask (from mask): %s", ens_params_mask)
                except Exception as e:
                    logger.debug("[PARAM_MATCH] Failed to parse ens_name with mask: %s", e)
            if not ens_params_mask and mask:
                try:
                    generic = self._get_generic_pattern(mask.item_type, effective_standard)
                    if generic:
                        m = re.search(generic, str(final_ens_name), re.IGNORECASE)
                        if m:
                            ens_params_mask = {k: v for k, v in m.groupdict().items() if v is not None}
                            logger.debug("[PARAM_MATCH] ENS params mask (from generic): %s", ens_params_mask)
                except Exception as e:
                    logger.debug("[PARAM_MATCH] Generic pattern parse error: %s", e)

        display_params = self._remap_params(final_matched_params) if final_matched_params else {}
        display_ens_params_mask = self._remap_params(ens_params_mask) if ens_params_mask else {}

        processing_time = (time.time() - start_time) * 1000

        text_norm = self.parametric_client._normalize_name(text) if text else ''
        ens_name_norm = self.parametric_client._normalize_name(final_ens_name) if final_ens_name else ''
        is_name_exact = text_norm and ens_name_norm and text_norm == ens_name_norm

        if substitution_info and final_ens_code:
            match_type_out = 'coating_substituted'
            match_type_ru = 'Совпадение после подбора правильного покрытия'
        elif is_name_exact and final_ens_code:
            match_type_out = 'name_exact'
            match_type_ru = 'Совпадение по наименованию'
        elif match_result.ens_code:
            # Map all parametric_client match_types explicitly
            pc_type = match_result.match_type
            if pc_type == 'name_exact' or is_name_exact:
                match_type_out = 'name_exact'
                match_type_ru = 'Совпадение по наименованию'
            elif pc_type in ('params_ens_exact', 'params_mask_exact', 'exact'):
                match_type_out = 'parametric_full'
                match_type_ru = 'Полное совпадение параметров'
            elif pc_type == 'v2_exact':
                match_type_out = 'v2_exact'
                match_type_ru = 'Полное совпадение V2'
            elif pc_type == 'partial':
                match_type_out = 'parametric_partial'
                match_type_ru = 'Частичное совпадение параметров'
            elif pc_type == 'fuzzy_fallback':
                match_type_out = 'fuzzy_fallback'
                match_type_ru = 'Нечеткое совпадение (fuzzy matching)'
            else:
                match_type_out = pc_type or 'unknown'
                match_type_ru = 'Сопоставление по параметрам'
        elif fuzzy_ens_code:
            # Если fuzzy нашел, но V2 подтвердил exact — это parametric_full
            if v2_computed and v2_score >= 0.99:
                match_type_out = 'parametric_full'
                match_type_ru = 'Полное совпадение параметров (V2 + fuzzy)'
            else:
                match_type_out = 'fuzzy_fallback'
                match_type_ru = 'Нечеткое совпадение (fuzzy matching)'
        else:
            match_type_out = match_result.match_type or None
            match_type_ru = 'Не определено'

        logger.info("[PARAM_MATCH] RETURN substitution_info=%s: %s", 'SET' if substitution_info else 'NONE', substitution_info)
        logger.info("[PARAM_MATCH] RETURN match_type=%s", match_type_out)

        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH,
            success=final_score >= _get_matching_config().success_threshold,
            params=display_params,
            ens_match={
                'code': final_ens_code,
                'name': final_ens_name,
                'mdm_key': final_mdm_key or final_ens_code,
                'score': final_score,
                'type': match_type_out,
                'params': ens_params_from_index
            } if final_ens_code else None,
            confidence=final_score,
            ens_params=ens_params_from_index,
            ens_params_mask=display_ens_params_mask,
            processing_time_ms=processing_time,
            details={
                'mask_id': mask.id,
                'mask_pattern': mask.pattern,
                'match_type': match_type_out,
                'match_type_ru': match_type_ru,
                'extracted_standard': extracted.get('standard_info'),
                'extracted_type': extracted.get('item_type'),
                'fuzzy_used': fuzzy_ens_code is not None and not match_result.ens_code,
                'debug_candidates': debug_candidates[:15] if debug_candidates else [],
                'fuzzy_score': round(fuzzy_score, 3) if fuzzy_score else 0,
                'match_result_score': round(match_result.score, 3) if match_result else 0,
                'v2_score': round(v2_score, 3) if 'v2_score' in locals() else None,
                'v2_computed': v2_computed if 'v2_computed' in locals() else False,
                'coating_substitution': substitution_info,
                'fuzzy_mismatched_params': fuzzy_mismatched_params if fuzzy_mismatched_params is not None else None,
                **({'fuzzy_params_comparison': fuzzy_debug[0].get('params_comparison')} if _get_matching_config().debug_per_parameter and fuzzy_debug else {}),
            },
            item_type=mask.item_type,
            standard=mask.standard
        )

    def _normalize_value_types(self, value: Any) -> Any:
        """Нормализация типов значений параметров."""
        if isinstance(value, float):
            if value == int(value):
                return int(value)
            return value
        if isinstance(value, str):
            try:
                f = float(value)
                if f == int(f):
                    return int(f)
                return f
            except ValueError:
                return value
        return value

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализация типов для всех значений в словаре параметров."""
        if not params:
            return params
        return {k: self._normalize_value_types(v) for k, v in params.items()}

    def _get_ens_by_code(self, ens_code: str) -> Optional[Dict[str, Any]]:
        """Поиск записи ЕСН по коду через индекс (O(1))."""
        if not self.parametric_client:
            return None
        return self.parametric_client._get_ens_by_code(ens_code)

    def _tfidf_fallback(
        self,
        text: str,
        extracted: Dict[str, Any],
        start_time: float
    ) -> ProcessingResult:
        """TF-IDF fallback — всегда success=False."""
        import time

        match_result = self.parametric_client._tfidf_fallback(text)
        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_7_TFIDF_FALLBACK,
            success=False,
            params={},
            ens_match=None,
            confidence=0.0,
            processing_time_ms=processing_time,
            details={
                'fallback': True,
                'tfidf_score': match_result.score,
                'tfidf_ens_candidate': match_result.ens_code,
                'extracted': extracted
            },
            item_type=extracted.get('item_type', ''),
            standard=extracted.get('standard_info', {}).get('standard', '') if isinstance(extracted.get('standard_info'), dict) else ''
        )

    def _llm_direct_process(self, text: str, start_time: float) -> ProcessingResult:
        """Прямая обработка через LLM (без маски)."""
        import time

        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_0_EXTRACT,
            success=False,
            params={},
            ens_match=None,
            confidence=0.0,
            processing_time_ms=processing_time,
            details={'error': 'Could not extract standard or type'},
            item_type='',
            standard=''
        )

    def batch_process(self, texts: List[str]) -> List[ProcessingResult]:
        """Пакетная обработка."""
        return [self.process(text) for text in texts]

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика процессора."""
        return {
            'mask_db_stats': self.mask_db.get_statistics(),
            'parametric_client_stats': self.parametric_client.get_stats(),
            'llm_generation_enabled': self.use_llm_generation,
            'min_mask_score': self.min_mask_score,
            'max_llm_retries': self.max_llm_retries
        }