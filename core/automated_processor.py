# =============================================================================
# ФАЙЛ: core/automated_processor.py
# ПОСЛЕДНИЕ 5 ИЗМЕНЕНИЙ (МСК, UTC+3), от новых к старым:
# 2026-06-08 14:30:00 — FIX 23: исполнение без скобок — \( и \) опциональны.
#   Болт 1-6-20-Бп = исп(1)+диам(6)+дл(20)+покр(Бп). Было: диам=1,дл=6,покр=20-Бп.
# 2026-06-08 14:15:00 — CRITICAL FIX: parametric_client.py re.match → re.IGNORECASE.
#   Все КАПС-наименования ('БОЛТ', 'ШАЙБА') теперь ловятся. ~+50% improvement.
# 2026-06-08 03:30:00 — FIX 21: разделитель [-.\s]+ перед ОСТ/ГОСТ.
#   Исправлены опечатки: (?P< не (?!<, ОСТ кирилл не OCT латин. Ловит "Хим.Н-ОСТ".
# 2026-06-08 03:25:00 — FIX 10v2: ГОСТ 1491/17475 шаг резьбы опционально. +~221.
# 2026-06-08 03:20:00 — FIX 18/19: покрытие [\w.\-]+, пробел после М [mмMМ]\s*.
# =============================================================================
"""
Main Processor Module
Integration of all levels: StandardExtractor -> MaskDatabase -> LLM Generator ->
AutoValidator -> ParametricMatch -> TF-IDF Fallback
"""

import json
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from utils.standard_utils import canonicalize_standard


# FEAT 2026-06-02: канонические ключи в _meta индекса
# _meta.code / _meta.name / _meta.standard / _meta.item_type
# Загружаются из ens_field_mapping в domain config; fallback — _get_ci() по record


def _get_ci(record: dict, key: str) -> Optional[Any]:
    """Case-insensitive dict.get. Returns value if key found (any case), else None.

    FIX 2026-06-02: ENS exports may have 'Код' instead of 'код', 'Наименование' vs 'наименование'.
    """
    # Fast path: exact match
    if key in record:
        return record[key]
    # Case-insensitive search
    key_lower = key.lower()
    for k, v in record.items():
        if isinstance(k, str) and k.lower() == key_lower:
            return v
    return None


def _get_meta_value(candidate: dict, canonical_key: str,
                    field_mapping: Optional[Dict] = None) -> Optional[Any]:
    """Extract value from ENS candidate by canonical key.

    Priority:
      1. candidate['_meta'][canonical_key] — новый формат индекса (e.g. _meta.code)
      2. candidate[field_name] — прямой доступ через field_mapping (e.g. candidate['Код'])

    FEAT 2026-06-02: unified extractor. DEFAULT_FIELD_MAPPING удалён — используем _meta.
    """
    # 1. _meta with canonical key (новый формат индекса после перестройки)
    meta = candidate.get('_meta', {}) or {}
    val = meta.get(canonical_key)
    if val is not None and str(val).strip() not in ('', 'None', 'null'):
        return val

    # 2. Direct access via field_mapping (e.g. field_mapping['code'] = 'Код')
    fm = field_mapping or {}
    field_name = fm.get(canonical_key)
    if field_name:
        val = _get_ci(candidate, field_name)
        if val is not None and str(val).strip() not in ('', 'None', 'null'):
            return val

    return None


def _find_field_value(record: dict, field_name: str) -> Optional[Any]:
    """Find field value in ENS record by exact, normalized, or partial match.

    FIX 2026-06-02: replaces broken candidate.get(param.replace('_',' '), '')
    which returned '' (empty string) when key not found, breaking numeric comparison.
    """
    # 1. Exact match
    if field_name in record and record[field_name] is not None:
        return record[field_name]

    # 2. Case-insensitive exact match (e.g. 'Код' vs 'код')
    val = _get_ci(record, field_name)
    if val is not None:
        return val

    # 3. Normalize: remove underscores
    fn_norm = field_name.replace('_', '')
    for key, val in record.items():
        if key.startswith('_'):
            continue
        if val is not None and key.replace('_', '') == fn_norm:
            return val

    # 4. Partial: field_name is substring of key (e.g. 'длина' in 'длина_резьбы')
    for key, val in record.items():
        if key.startswith('_'):
            continue
        if val is not None and field_name in key:
            return val

    # 4. Partial reverse: key is substring of field_name
    for key, val in record.items():
        if key.startswith('_'):
            continue
        if val is not None and key in field_name:
            return val

    return None


# Lazy import to avoid circular dependency
_matching_config = None

def _get_matching_config():
    """Lazy load MatchingConfig from settings or config.yaml directly."""
    global _matching_config
    if _matching_config is None:
        try:
            from core.settings import get_settings
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
            'ens_params_mask': (self.ens_params_mask if isinstance(self.ens_params_mask, dict) else {}),
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
        cache_ttl_days: int = 7,
        no_cache: bool = False,
        domain: Optional[str] = None,
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
        self.no_cache = no_cache
        self.domain = domain
        self._cache_stats = {'hits': 0, 'misses': 0}

        # FEAT 2026-06-02: load field mapping + extraction rules from domain config
        self._field_mapping = self._load_domain_config(domain, 'ens_field_mapping', {})
        self._item_types = self._load_domain_config(domain, 'item_types', ['Болт', 'Винт', 'Гайка', 'Шайба'])
        self._standard_patterns = self._load_domain_config(domain, 'standard_patterns',
            [r'ОСТ\s*\d+\s*\d+-\d+', r'ГОСТ\s*\d+-\d+', r'ТУ\s*\d+-\d+'])
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

    @staticmethod
    def _load_domain_config(domain: Optional[str], key: str, default: Any) -> Any:
        """Load a config key from domain config YAML.

        FEAT 2026-06-02: generic loader for ens_field_mapping, item_types, standard_patterns, etc.
        Falls back to default if domain not found or key not present.
        """
        if not domain:
            return default
        try:
            from core.domain_config import DomainConfig
            cfg = DomainConfig.load(domain)
            val = getattr(cfg, key, None)
            if val is not None:
                logger.info("[DOMAIN_CFG] %s from '%s': %s", key, domain, val)
                return val
        except Exception as e:
            logger.debug("[DOMAIN_CFG] Failed to load %s for domain '%s': %s", key, domain, e)
        return default

    def _validate_extraction(self, item_type: Optional[str], standard_info: Any) -> Tuple[Optional[str], Optional[str]]:
        """Validate and normalize extracted item_type and standard against domain config.

        FEAT 2026-06-02: uses self._item_types and self._standard_patterns from domain config.
        """
        validated_type = None
        if item_type:
            it_clean = str(item_type).strip()
            for allowed in self._item_types:
                if it_clean.lower() == allowed.lower():
                    validated_type = allowed  # use canonical form from config
                    break
            if not validated_type:
                # 2026-06-03 15:00:00 (МСК, UTC+3): не отбрасываем тип — используем как есть
                # Иначе кандидаты ENS не ищутся (Заклепка не в allowed list, но в индексе есть)
                validated_type = it_clean.capitalize()
                logger.debug("[EXTRACT] item_type '%s' not in allowed list, using as '%s'",
                             it_clean, validated_type)

        validated_standard = None
        if standard_info and getattr(standard_info, 'normalized', None):
            std = str(standard_info.normalized).strip()
            import re
            for pat in self._standard_patterns:
                if re.search(pat, std, re.IGNORECASE):
                    validated_standard = std
                    break
            if not validated_standard:
                logger.debug("[EXTRACT] standard '%s' did not match any pattern: %s", std, self._standard_patterns)

        return validated_type, validated_standard

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
            from core.llm_mask_generator import LLMMaskGenerator
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
            text_safe = text or ""
            logger.debug("[CACHE] DB lookup: article=%r name=%s", article, text_safe[:50])
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
            else:
                logger.debug("[CACHE] DB miss for '%s...'", text_safe[:50])
        except Exception as e:
            logger.debug("Cache check error: %s", e)
        return None

    @staticmethod
    def _ensure_dict(val):
        """Преобразует значение из БД в dict. Защита от JSON-строк и None.
        2026-06-03 15:05 (МСК, UTC+3): добавлено после ошибки 'NoneType' not subscriptable.
        """
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    def _result_from_cache(self, cached: Dict[str, Any]) -> ProcessingResult:
        """Восстановить ProcessingResult из кэшированной записи result.db."""
        # 2026-06-03 15:05 (МСК, UTC+3): защита от None/invalid cached records
        if not cached or not isinstance(cached, dict):
            logger.warning("[CACHE] Invalid cached record: %s", type(cached).__name__)
            return ProcessingResult(success=False)

        params = self._ensure_dict(cached.get('params'))
        ens_params = self._ensure_dict(cached.get('ens_params'))
        ens_params_mask = self._ensure_dict(cached.get('ens_params_mask'))
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
        details = self._ensure_dict(cached.get('details'))
        details['from_cache'] = True
        details['cached_at'] = cached.get('updated_at')

        level_str = cached.get('level', 'parametric_match') or 'parametric_match'
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
            confidence=min(float(cached.get('confidence', 0.0)), 1.0),
            processing_time_ms=0.0,
            details=details,
            item_type=cached.get('item_type', ''),
            standard=cached.get('standard', '')
        )

    def process(self, text: str, article: str = "", force: bool = False) -> ProcessingResult:
        """
        Обработка одной строки номенклатуры.
        """
        try:
            result = self._process_impl(text, article, force)
            return result
        except Exception:
            import traceback
            logger.error("[PROCESS_ERROR] %s: %s", text[:50], traceback.format_exc())
            return ProcessingResult(success=False)


    def _process_impl(self, text: str, article: str = "", force: bool = False) -> ProcessingResult:
        """
        Обработка одной строки номенклатуры (impl).
        """
        import time
        start_time = time.time()

        clean_text = text.strip().rstrip(',.;: ')
        if clean_text != text.strip():
            logger.debug("Cleaned trailing punctuation: '%s' -> '%s'", text, clean_text)

        # 2026-06-03 14:50:00 (МСК, UTC+3): Заклёпка → Заклепка (ё→е нормализация)
        # Иначе standard_extractor не распознаёт тип с кириллической ё
        clean_text = clean_text.replace("ё", "е").replace("Ё", "Е")
        # FIX 2026-06-05 08:35 (МСК, UTC+3): Кириллическая М (U+041C) → латинская M (U+004D)
        # перед числом — метрическая резьба М10, М12 и т.д.
        clean_text = re.sub(r'(?<![a-zA-Zа-яА-Я])М(?=\d)', 'M', clean_text)

        extracted = self.standard_extractor.extract_all(clean_text)
        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')

        # FEAT 2026-06-02: validate against domain config (item_types, standard_patterns)
        validated_type, validated_standard = self._validate_extraction(
            item_type, standard_info
        )
        item_type = validated_type or item_type
        extracted_standard = validated_standard or (
            canonicalize_standard(standard_info.normalized) if standard_info else None
        )

        # CACHE CHECK
        logger.debug("[CACHE] process() called: text=%s standard=%s result_db_path=%s",
                     clean_text[:50], extracted_standard, self.result_db_path)
        if not force and self.result_db_path and not self.no_cache:
            cached = self._check_cache(clean_text, extracted_standard)
            if cached:
                logger.info("[CACHE] HIT for '%s' / %s (code=%s, mask=%s)",
                            clean_text[:50], extracted_standard,
                            cached.get('ens_code', 'N/A'), (cached.get('mask_pattern') or 'N/A')[:30])
                return self._result_from_cache(cached)
            else:
                logger.info("[CACHE] MISS for '%s' / %s", clean_text[:50], extracted_standard)
        else:
            if force:
                logger.debug("[CACHE] skipped (force=True)")
            elif not self.result_db_path:
                logger.debug("[CACHE] skipped (result_db_path not set)")
            self._cache_stats['misses'] += 1

        logger.info("=" * 70)
        logger.info("Обработка номенклатуры: %s", clean_text[:60])
        logger.info("=" * 70)

        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')

        if not standard_info or not item_type:
            logger.info("[ЭТАП 1] Поиск по наименованию: стандарт/тип не определены")
            return self._llm_direct_process(clean_text, start_time)

        standard = canonicalize_standard(standard_info.normalized)
        logger.info("[ЭТАП 1] Поиск по наименованию: стандарт=%s, тип=%s", standard, item_type)

        # Level 1: Проверка MaskDatabase
        search_item_type = item_type.upper()
        mask = self.mask_db.get_mask(standard, search_item_type)
        if mask is None:
            mask = self.mask_db.get_mask(standard, item_type)
        if mask is None:
            try:
                standard_masks = self.mask_db.get_all_masks(standard=standard)
                if standard_masks:
                    mask = standard_masks[0]
            except Exception:
                pass

        if mask and mask.is_active:
            logger.info("[ЭТАП 1] ✓ Маска найдена: id=%s, стандарт=%s, тип=%s",
                        mask.id, getattr(mask, 'standard', standard), mask.item_type)
            mask = self._fix_loaded_mask(mask)
            return self._parametric_match(clean_text, mask, extracted, start_time)
        elif mask:
            logger.info("[ЭТАП 1] ⚠ Маска найдена, но неактивна")
            mask.is_active = True
            mask = self._fix_loaded_mask(mask)
            return self._parametric_match(clean_text, mask, extracted, start_time)
        else:
            logger.info("[ЭТАП 1] ✗ Маска не найдена")

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
        mask, meta = self.llm_generator.generate_mask(
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
        from core.mask_database import MaskRecord
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
        from core.mask_database import MaskRecord
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
                from core.settings import get_settings
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
        best_exact_count = 0
        prev_best_idx = None  # index of previous best in debug_candidates
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
            candidate_name = _get_meta_value(candidate, 'name', self._field_mapping) or ''
            candidate_debug = {
                'name': candidate_name,
                'ens_code': _get_meta_value(candidate, 'code', self._field_mapping),
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

                candidate_val = _find_field_value(candidate, param_name)

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

                    # FIX 2026-06-02: if field doesn't match, check if value is in candidate NAME
                    # ENS data may have stale field values but correct name
                    # FIX 2026-06-09: exclude matches inside standard codes (ГОСТ XXXX-XX, ОСТ XXX-XX)
                    in_name = False
                    in_name_weight_factor = 0.5  # reduced weight for in-name matches
                    if not matched and candidate_name:
                        # Strict: value must appear as token in name (e.g. "12" in "Болт (2)-12-44-...")
                        name_lower = candidate_name.lower().replace(',', '.')
                        val_str = str(extracted_val).lower().strip().replace(',', '.')
                        # Token boundary check: preceded by non-digit, followed by non-digit
                        # Token boundary: preceded by non-digit, followed by non-digit
                        # Allow '.' after number as separator (e.g. "50.Хим" → "50" is a token)
                        in_name_raw = bool(re.search(r'(?<![\d.])' + re.escape(val_str) + r'(?![\d])', name_lower))
                        if in_name_raw:
                            # Exclude matches inside standard code portion of name
                            # e.g. "70" in "ГОСТ 3129-70", "1" in "ОСТ1 10569-72"
                            std_code_pattern = r'(?:гост|ост|ту)\s*\d[\d\s\.\-]*(?:\-\d+)?'
                            name_without_std = re.sub(std_code_pattern, '', name_lower)
                            in_name = bool(re.search(r'(?<![\d.])' + re.escape(val_str) + r'(?![\d])', name_without_std))

                    status = 'exact' if matched else ('exact (in name)' if in_name else 'mismatched')
                    candidate_debug['params_comparison'][param_name] = {
                        'status': status,
                        'extracted': str(extracted_val),
                        'ens_value': str(candidate_val) if candidate_val else None,
                        'similarity': 1.0 if matched else (0.5 if in_name else 0.0),
                    }
                    if matched:
                        matched_weight += weight
                        candidate_debug['params_matched'][param_name] = "{} == {}".format(extracted_val, candidate_val)
                    elif in_name:
                        matched_weight += weight * in_name_weight_factor
                        candidate_debug['params_matched'][param_name] = "{} ≈ {} (in name)".format(extracted_val, candidate_val)
                    else:
                        candidate_debug['params_mismatched'][param_name] = "{} != {}".format(extracted_val, candidate_val)

            # 2026-06-04 13:20 (МСК, UTC+3): бонус за exact coating match
            # Кд=Кд должен выигрывать у Кд~Ц9.фос.окс при равном score
            # FIX 2026-06-04: бонус за token_matched + доп.бонус если покрытие кандидата
            # начинается с извлеченного покрытия (Кд9.хр начинается с Кд, Ц9.фос.окс — нет)
            coating_comparison = candidate_debug.get('params_comparison', {}).get('покрытие', {})
            coating_status = coating_comparison.get('status', '')
            if coating_status in ('exact', 'token_matched'):
                matched_weight += 0.01
                # Дополнительный бонус за prefix match: Кд9.хр начинается с Кд
                extracted_coating = str(extracted_params.get('покрытие', '')).lower().strip()
                candidate_coating = str(coating_comparison.get('ens_value', '')).lower().strip()
                if candidate_coating.startswith(extracted_coating):
                    matched_weight += 0.01

            score = matched_weight / total_weight if total_weight > 0 else 0.0
            exact_count = sum(1 for p in candidate_debug['params_comparison'].values() if p['status'] == 'exact')
            candidate_debug['score'] = round(score, 3)
            candidate_debug['exact_count'] = exact_count
            candidate_debug['total_weight'] = total_weight
            candidate_debug['matched_weight'] = round(matched_weight, 3)
            candidate_debug['params_count'] = len(extracted_params)

            # FEAT 2026-06-02: tie-breaker — prefer candidate with more exact matches
            is_better = score > best_score
            if score == best_score and exact_count > best_exact_count:
                is_better = True
            # 2026-06-04 13:20 (МСК, UTC+3): exact coating match даёт +0.01 бонус
            # в matched_weight — tie-breaker при равном score/exact_count
            if is_better:
                # FIX 2026-06-02: clear is_best from previous best candidate
                if prev_best_idx is not None and prev_best_idx < len(debug_candidates):
                    debug_candidates[prev_best_idx]['is_best'] = False
                prev_best_idx = len(debug_candidates)
                best_score = score
                best_exact_count = exact_count
                best_match = candidate
                candidate_debug['is_best'] = True
            else:
                candidate_debug['is_best'] = False

            debug_candidates.append(candidate_debug)

        # FEAT 2026-06-02: structured table logging
        self._log_candidates_table(extracted_params, debug_candidates, best_match, best_score)

        if not best_match:
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

    def _log_candidates_table(self, extracted_params: Dict[str, str],
                               debug_candidates: List[Dict],
                               best_match: Optional[Dict],
                               best_score: float) -> None:
        """Log TOP-5 candidates in structured table format.

        FEAT 2026-06-02: readable table with param comparisons per candidate.
        """
        if not debug_candidates:
            logger.info("[ЭТАП 2] Кандидаты не найдены")
            return

        param_names = list(extracted_params.keys())
        sorted_cds = sorted(debug_candidates, key=lambda x: (-x['score'], -x.get('exact_count', 0)))

        # Header
        logger.info("[ЭТАП 2] Топ-%d кандидатов по параметрам:", min(5, len(sorted_cds)))
        header = "│ {:<12} │ {:<36} │ {:>5} ".format(
            "Код ЕНС", "Наименование ЕНС", "Score")
        for pn in param_names:
            header += "│ {:>14} ".format(pn[:14])
        logger.info(header)
        logger.info("├" + "─" * 14 + "┼" + "─" * 38 + "┼" + "─" * 7 +
                    "".join("┼" + "─" * 16 for _ in param_names) + "┤")

        # Rows
        for cd in sorted_cds[:5]:
            is_best = cd.get('is_best', False)
            code = (cd.get('ens_code') or 'N/A')[:12]
            name = (cd.get('name') or 'N/A')[:34] + (' ★' if is_best else '')
            score = cd.get('score', 0)
            row = "│ {:<12} │ {:<36} │ {:>5.3f} ".format(code, name, score)
            for pn in param_names:
                pcmp = cd.get('params_comparison', {}).get(pn, {})
                status = pcmp.get('status', '?')
                extracted = pcmp.get('extracted', '?')
                ens_val = pcmp.get('ens_value', '?')
                if status == 'exact':
                    cell = "{}={}".format(extracted, ens_val)
                elif status == 'exact (in name)':
                    cell = "{}≈{}☆".format(extracted, ens_val)
                elif 'token' in status:
                    cell = "{}~{}".format(extracted, ens_val)
                else:
                    cell = "{}≠{}✗".format(extracted, ens_val)
                row += "│ {:>14} ".format(cell[:14])
            logger.info(row)

        # Footer
        if best_match and best_score >= 0.7:
            bm_code = _get_meta_value(best_match, 'code', self._field_mapping) or 'N/A'
            logger.info("★ Успешный кандидат: %s (score=%.3f)", bm_code, best_score)
        elif best_match:
            logger.info("⚠ Лучший кандидат ниже порога: score=%.3f < 0.7", best_score)
        else:
            logger.info("✗ Совпадение не найдено")

    def _get_generic_pattern(self, standard: str, item_type: str, params: Dict[str, str], mask: Any) -> str:
        """Построить generic-паттерн для V2 exact match."""
        parts = [item_type]
        for key in sorted(params.keys()):
            val = params[key]
            if val is not None and str(val).strip():
                parts.append(str(val).strip())
        parts.append(standard)
        return ' '.join(parts)

    @staticmethod
    def _apply_mask_fixes(pattern: str) -> str:
        r"""Применить фиксы к паттерну маски. Идемпотентный — безопасно вызывать многократно.
        Выделен из _fix_loaded_mask для применения перед КАЖДЫМ использованием маски.
        2026-06-05 11:00 (МСК, UTC+3): Идемпотентность + покрытие не поглощает ГОСТ/ОСТ.
        2026-06-05 18:10 (МСК, UTC+3): Все строковые литералы — raw strings (SyntaxWarning убраны).
        """
        if not pattern:
            return pattern
        # FIX 1b: [mм] → [mмMМ] — заглавная М (кирилл U+041C) в "Винт М1,2-10"
        if r'[mм]' in pattern and r'[mмMМ]' not in pattern:
            pattern = pattern.replace(r'[mм]', r'[mмMМ]')
        # FIX 13: Ё в Заклёпка — ^Заклепка → ^Закл[её]пка
        if r'^Заклепка' in pattern:
            pattern = pattern.replace(r'^Заклепка', r'^Закл[её]пка')
        # FIX 14: суффикс после ОСТ/ГОСТ — "по 226.57...", "080340004" и т.д.
        # Любой текст после номера стандарта: (?:\s+.*)?$ — идемпотентный
        if r'(?:\s+.*)?$' not in pattern and (r'ГОСТ' in pattern or r'ОСТ' in pattern):
            pattern = pattern.replace(r'$', r'(?:\s+.*)?$')
        # FIX 1: покрытие опциональное (только если ещё не опциональное)
        if r')?[-\s]+ОСТ' not in pattern and r')?[-\s]+ГОСТ' not in pattern:
            pattern = pattern.replace(
                r'[-\s]+(?P<покрытие>[\w.]+)[-\s]+ОСТ',
                r'(?:[-\s]+(?P<покрытие>[\w.]+))?[-\s]+ОСТ'
            )
            pattern = pattern.replace(
                r'[-\s]+(?P<покрытие>[\w.]+)[-\s]+ГОСТ',
                r'(?:[-\s]+(?P<покрытие>[\w.]+))?[-\s]+ГОСТ'
            )
        # FIX 2: шаг резьбы через - или х
        pattern = pattern.replace(r'(?:[xXхХ×](?P<шаг_резьбы>', r'(?:[xXхХ×\-](?P<шаг_резьбы>')
        pattern = pattern.replace(r'[xXхХ×](?P<шаг_резьбы>', r'[xXхХ×\-](?P<шаг_резьбы>')
        # FIX 3: класс_допуска опциональный
        if r'(?:(?P<класс_допуска>' not in pattern:
            pattern = pattern.replace(r'(?P<класс_допуска>\d+[a-z])х(?P<длина>', r'(?:(?P<класс_допуска>\d+[a-z])х)?(?P<длина>')
        # FIX 4: исполнение без разделителя перед М
        pattern = pattern.replace(r'(?P<исполнение>\d+)[-\s]+M', r'(?P<исполнение>\d+)[-\s]*M')
        pattern = pattern.replace(r'(?P<исполнение>\d+)[-\s]+)?M', r'(?P<исполнение>\d+)[-\s]*)?M')
        # FIX 5: Штифты ГОСТ 3128/3129
        if 'Штифт' in pattern and (r'ГОСТ\s*3128' in pattern or r'ГОСТ\s*3129' in pattern):
            # Исполнение не захватывает часть диаметра (0 из 0,6)
            pattern = pattern.replace(r'(?P<исполнение>\d+)(?![.,])', r'(?P<исполнение>\d+)(?![.,]\d)')
            # Разделитель может быть точка: 2.0,6х10
            pattern = pattern.replace(r'[-\s]+(?P<наружный_диаметр_сторона_квадр>', r'[-\s.]+(?P<наружный_диаметр_сторона_квадр>')
            pattern = pattern.replace(r'(?:[-\s]+(?P<длина>', r'(?:[xXхХ×\-\s]+(?P<длина>')
            pattern = pattern.replace(r'(?:[xXхХ×](?P<класс_допуска>[a-zA-Z]\d+))?', r'(?:[xXХ×](?P<класс_допуска>[a-zA-Z]\d+))?')
            # FIX 5b: диаметр с запятой — БЕЗ условия, т.к. (?:[.,]\d+)? может быть в длине
            pattern = pattern.replace(
                r'(?P<наружный_диаметр_сторона_квадр>\d+(?:\.\d+)?)',
                r'(?P<наружный_диаметр_сторона_квадр>\d+(?:[.,]\d+)?)'
            )
            pattern = pattern.replace(r'(?P<марка_материала1>\w+))?(?:[-\s]+(?P<состояние_поставки_металлопрок>', r'(?P<марка_материала1>(?!ГОСТ|ОСТ)[\w.]+))?(?:[-\s]+(?P<состояние_поставки_металлопрок>')
            pattern = pattern.replace(r'(?P<покрытие>\w+\.?\w*\.?\w*)', r'(?P<покрытие>[\w.]+)')
            # Покрытие не поглощает ГОСТ/ОСТ и не является чистым числом — начинается с буквы
            if '(?!ГОСТ|ОСТ)' not in pattern:
                pattern = pattern.replace(r'(?P<покрытие>[\w.]+))', r'(?P<покрытие>(?!ГОСТ|ОСТ)[a-zA-Zа-яА-ЯёЁ][\w.]*))')
            else:
                # (?!ГОСТ|ОСТ) уже есть, но [\w.]+ ловит числа — заменить на буква+[\w.]*
                # + делаем покрытие опциональным (?:...)?[-\s]+ГОСТ
                pattern = pattern.replace(
                    r'(?P<покрытие>(?!ГОСТ|ОСТ)[\w.]+))[-\s]+ГОСТ',
                    r'(?P<покрытие>(?!ГОСТ|ОСТ)[a-zA-Zа-яА-ЯёЁ][\w.]*))?[-\s]+ГОСТ'
                )
        # FIX 6: Болт ГОСТ 7798 шаг через х
        if r'(?P<номинальный_диаметр_резьбы>' in pattern and r'ГОСТ\s*7798' in pattern:
            if r'(?:[xXхХ×\-](?P<шаг_резьбы>' not in pattern:
                pattern = pattern.replace(r'(?P<номинальный_диаметр_резьбы>\d+(?:[.,]\d+)?)[-\s]+(?:(?P<класс_допуска>', r'(?P<номинальный_диаметр_резьбы>\d+(?:[.,]\d+)?)(?:[xXхХ×\-](?P<шаг_резьбы>\d+(?:[,\.]\d+)?))?[xXхХ×\-\s]+(?:(?P<класс_допуска>')
        # FIX 7: Винт
        if pattern.startswith('^Винт'):
            if r'\(' not in pattern:
                pattern = pattern.replace(r'(?P<исполнение>\d+))?(?:[-\s]+)?', r'\((?P<исполнение>\d+)\))?[-\s]*')
                pattern = pattern.replace(r'(?:(?:[-\s]+(?P<исполнение>\d+))?)', r'(?:\((?P<исполнение>\d+)\))?')
            # M/m/м перед диаметром — опциональный
            if r'[mм]?(?P<номинальный_диаметр_резьбы>' not in pattern and r'[mм]?(?P<наружный_диаметр_сторона_квадр>' not in pattern:
                pattern = pattern.replace(r'\s*M(?P<', r'\s*[mм]?(?P<')
                pattern = pattern.replace(r'\s+M(?P<', r'\s*[mм]?(?P<')
                pattern = pattern.replace(r'(?P<класс_точности>[ABАВ])\s*M(?P<', r'(?P<класс_точности>[ABАВ])\s*[mм]?(?P<')
            # х между диаметром и длиной — также дефис: 4-10, М1,2-10
            pattern = pattern.replace(r'[-\s]+(?P<длина>', r'[xXхХ×\-\s]+(?P<длина>')
            # [xXхХ×](класс_допуска...х)?длина → [xXхХ×\-](...)
            pattern = pattern.replace(
                r'[xXхХ×](?:(?P<класс_допуска>\d+[a-z])х)?(?P<длина>',
                r'[xXхХ×\-](?:(?P<класс_допуска>\d+[a-z])[xXхХ×])?(?P<длина>'
            )
            # класс_точности [AB] → [ABАВ] (кирилл. А/В)
            pattern = pattern.replace(r'(?P<класс_точности>[AB])', r'(?P<класс_точности>[ABАВ])')
            # покрытие \w+\.?\w*\.?\w* → [\w.]+ (универсальнее)
            pattern = pattern.replace(r'(?P<покрытие>\w+\.?\w*\.?\w*)', r'(?P<покрытие>[\w.]+)')
            # FIX 7b: покрытие опциональное
            if r')?[-\s]+ОСТ' not in pattern and r')?[-\s]+ГОСТ' not in pattern:
                pattern = pattern.replace(
                    r'[-\s]+(?P<покрытие>[\w.]+)[-\s]+ОСТ',
                    r'(?:[-\s]+(?P<покрытие>[\w.]+))?[-\s]+ОСТ'
                )
                pattern = pattern.replace(
                    r'[-\s]+(?P<покрытие>[\w.]+)[-\s]+ГОСТ',
                    r'(?:[-\s]+(?P<покрытие>[\w.]+))?[-\s]+ГОСТ'
                )
            # FIX 7c: покрытие не поглощает ГОСТ/ОСТ
            if '(?!ГОСТ|ОСТ)' not in pattern:
                pattern = pattern.replace(r'(?P<покрытие>[\w.]+))', r'(?P<покрытие>(?!ГОСТ|ОСТ)[a-zA-Zа-яА-ЯёЁ][\w.]*))')
            else:
                pattern = pattern.replace(
                    r'(?P<покрытие>(?!ГОСТ|ОСТ)[\w.]+))',
                    r'(?P<покрытие>(?!ГОСТ|ОСТ)[a-zA-Zа-яА-ЯёЁ][\w.]*))'
                )
            # FIX 7d: покрытие из цифр с точками — 58.029
            pattern = pattern.replace(
                r'(?P<покрытие>\d+)',
                r'(?P<покрытие>\d+(?:\.\d+)*)'
            )
        # FIX 8: Винт без M — опциональный M/m/м перед диаметром
        if pattern.startswith('^Винт'):
            if r'[mм]?(?P<номинальный_диаметр_резьбы>' not in pattern:
                pattern = pattern.replace(
                    r'\s+M(?P<номинальный_диаметр_резьбы>',
                    r'\s*[mм]?(?P<номинальный_диаметр_резьбы>'
                )
        # FIX 10: Винт ГОСТ 1491/17475 — полная замена маски
        # Шаг резьбы опционально: M10x1,25-100-40 = диам(10)+шаг(1,25)+дл(100)+покр(40)
        if r'ГОСТ\s*1491' in pattern:
            pattern = (r'^Винт\s+(?P<класс_точности>[ABАВ])?[mмMМ]'
                       r'(?P<номинальный_диаметр_резьбы>\d+(?:,\d+)?)'
                       r'(?:[xXхХ×](?P<шаг_резьбы>\d+(?:,\d+)?))?[-\s]+'
                       r'(?P<длина>\d+(?:,\d+)?)(?P<класс_допуска>[a-zа-я])?'
                       r'[xXхХ×\-]*(?P<покрытие>[\d.]+)?[-\s]+'
                       r'ГОСТ\s*1491-80$')
        if r'ГОСТ\s*17475' in pattern:
            pattern = (r'^Винт\s+(?P<класс_точности>[ABАВ])?[mмMМ]'
                       r'(?P<номинальный_диаметр_резьбы>\d+(?:,\d+)?)'
                       r'(?:[xXхХ×](?P<шаг_резьбы>\d+(?:,\d+)?))?[-\s]+'
                       r'(?P<длина>\d+(?:,\d+)?)(?P<класс_допуска>[a-zа-я])?'
                       r'[xXхХ×\-]*(?P<покрытие>[\d.]+)?[-\s]+'
                       r'ГОСТ\s*17475-80$')
        # FIX 11: Гайка ГОСТ 5915 — полная замена маски
        # Исполнение перед М: "2М16х1.5-6g.8.019", "М6-6g.5.8"
        # Шаг только через х (не через -), иначе шаг поглощает класс ("-6g" -> шаг=6)
        if r'ГОСТ\s*5915' in pattern:
            pattern = (r'^Гайка\s*(?P<исполнение>\d+)?[mмMМ]'
                       r'(?P<номинальный_диаметр_резьбы>\d+)'
                       r'(?:[xXхХ×](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?'
                       r'[-\s]*'
                       r'(?P<класс_допуска>[\dA-Za-z]+)?\.?'
                       r'(?P<группа_прочности>\d+(?:[.,]\d+)?)?\.?'
                       r'(?P<марка_материала1>\d+[A-ZА-Яа-я]+)?\.?'
                       r'(?P<покрытие>\d+(?:\.\d+)*)?'
                       r'[-\s]*ГОСТ\s*5915-70$')
        # FIX 12: Штифт ГОСТ 3128/3129 — полная замена маски
        # Исполнение опционально: "2.0,6х10" = исполнение(2) + диам(0,6) + дл(10)
        # Покрытие опционально: "-36", "-Ц9.фос.окс" или отсутствует
        if r'ГОСТ\s*3128' in pattern and 'Штифт' in pattern:
            pattern = (r'^Штифт\s+(?:(?P<исполнение>\d+)[-\s.]+)?'
                       r'(?P<наружный_диаметр_сторона_квадр>\d+(?:[.,]\d+)?)[xXхХ×]'
                       r'(?P<длина>\d+(?:[.,]\d+)?)'
                       r'(?:[-\s]+(?P<покрытие>[\w.]+))?[-\s]+'
                       r'ГОСТ\s*3128-70$')
        if r'ГОСТ\s*3129' in pattern and 'Штифт' in pattern:
            pattern = (r'^Штифт\s+(?:(?P<исполнение>\d+)[-\s.]+)?'
                       r'(?P<наружный_диаметр_сторона_квадр>\d+(?:[.,]\d+)?)[xXхХ×]'
                       r'(?P<длина>\d+(?:[.,]\d+)?)'
                       r'(?:[-\s]+(?P<покрытие>[\w.]+))?[-\s]+'
                       r'ГОСТ\s*3129-70$')
        # FIX 18: покрытие с дефисом — "10-Кd", "АН.ОКС." — [\w.]* → [\w.\-]*
        # + цифра в начале: [a-zA-Zа-яА-ЯёЁ] → [a-zA-Zа-яА-ЯёЁ\d]
        if '(?P<покрытие>' in pattern and r'[\w.\-]*' not in pattern:
            pattern = pattern.replace(
                r'(?P<покрытие>(?!ГОСТ|ОСТ)[a-zA-Zа-яА-ЯёЁ][\w.]*)',
                r'(?P<покрытие>(?!ГОСТ|ОСТ)[a-zA-Zа-яА-ЯёЁ\d][\w.\-]*)'
            )
            pattern = pattern.replace(
                r'(?P<покрытие>[\w.]+)',
                r'(?P<покрытие>[\w.\-]+)'
            )
        # FIX 19: пробел после M — "Гайка М 18х1,5" → [mмMМ]\s*(?P<
        # normalize оставляет М кирилл если после неё пробел (М 18 → не заменяет)
        if 'Гайка' in pattern and r'[mмMМ]\s*(?P<' not in pattern:
            pattern = pattern.replace(r'M(?P<', r'[mмMМ]\s*(?P<')
        # FIX 21: разделитель покрытie->OCT — [-.\s]* вместo [-\s]+
        # FIX 21: разделитель перед ОСТ/ГОСТ — [-.\s]+ вместо [-\s]+
        # Ловит "-ОСТ", " .ОСТ" (точка как разделитель после покрытия)
        if r'[.\-\s]+ОСТ' not in pattern:
            pattern = pattern.replace(r'[-\s]+ОСТ', r'[.\-\s]+ОСТ')
        if r'[.\-\s]+ГОСТ' not in pattern:
            pattern = pattern.replace(r'[-\s]+ГОСТ', r'[.\-\s]+ГОСТ')
        # FIX 23: исполнение без скобок — \( и \) опциональны
        # Болт 1-6-20-Бп = исп(1) + диам(6) + дл(20) + покр(Бп)
        if r'\((?P<исполнение>' in pattern and r'\(?\(?' not in pattern:
            pattern = pattern.replace(
                r'(?:[-\s]+\((?P<исполнение>\d+)\))?',
                r'(?:[-\s]+\(?(?P<исполнение>\d+)\)?)?'
            )
        # FIX 22: Винт с параметрами ПОСЛЕ ОСТ (отложено)
        return pattern

    def _fix_loaded_mask(self, mask):
        """DEPRECATED: Фиксы теперь в _apply_mask_fixes, вызывается перед каждым использованием."""
        if not mask or not getattr(mask, 'pattern', None):
            return mask
        mask.pattern = AutomatedParametricProcessor._apply_mask_fixes(mask.pattern)
        return mask

    def _parametric_match(self, text: str, mask: Any, extracted: Dict, start_time: float) -> ProcessingResult:
        """
        Параметрическое сопоставление с ЕНС.
        """
        import time
        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')
        standard = canonicalize_standard(standard_info.normalized) if standard_info else ''

        logger.info("[ЭТАП 2] Извлечение параметров из наименования...")
        # FIX 2026-06-05: применить фиксы перед КАЖДЫМ использованием маски (идемпотентно)
        mask.pattern = self._apply_mask_fixes(mask.pattern)
        params = self.parametric_client.extract_params(text, mask.pattern)
        remapped = self._remap_params(params, mask.params)
        logger.info("[ЭТАП 2] Извлечённые параметры: %s", remapped)

        # Get ENS candidates
        candidates = self._get_cached_ens_candidates(standard, item_type)
        if not candidates:
            logger.info("[ЭТАП 2] ✗ Кандидаты ЕНС не найдены для %s/%s", standard, item_type)
            return self._tfidf_fallback(text, extracted, start_time)

        substitution_info = None  # always defined

        # FEAT 2026-06-02: ЭТАП 1.5 — exact match по наименованию (самый быстрый)
        best_match = None
        best_score = 0.0
        match_type = None
        match_type_ru = None
        text_normalized = text.lower().replace(' ', '').replace('\xa0', '')
        for candidate in candidates:
            cand_name = _get_meta_value(candidate, 'name', self._field_mapping) or ''
            if cand_name:
                cand_normalized = cand_name.lower().replace(' ', '').replace('\xa0', '')
                if text_normalized == cand_normalized:
                    best_match = candidate
                    best_score = 1.0
                    match_type = 'exact_name'
                    match_type_ru = 'Точное совпадение по наименованию'
                    logger.info("[ЭТАП 1.5] ✓ Точное совпадение по имени: %s", cand_name[:60])
                    break
        if best_match:
            return self._build_parametric_result(
                text, mask, extracted, remapped, best_match,
                best_score, match_type, match_type_ru, [],
                substitution_info, start_time
            )

        # Apply coating substitution
        substituted_params, substitution_info = self._apply_coating_substitution(remapped, candidates)
        if substitution_info:
            logger.info("[ЭТАП 2] Подстановка покрытия: %s", substitution_info)
            remapped = substituted_params

        # V2 exact match via generic pattern
        generic_pattern = self._get_generic_pattern(standard, item_type, remapped, mask)
        # Count non-empty params in generic pattern (item_type + standard excluded)
        param_count = len([v for v in remapped.values() if v is not None and str(v).strip()])
        logger.debug("[ЭТАП 2] Generic pattern: %s (params=%d)", generic_pattern, param_count)
        fuzzy_mismatched_params = None
        debug_candidates = []

        # Try exact match first (only if enough params to avoid degenerate matches)
        # FIX 2026-06-09: require min 4 params AND ≥70% of ENS candidate params matched
        # Prevents false exact matches when input is missing critical ENS params
        V2_MIN_PARAMS = 4
        V2_MIN_RATIO = 0.70
        if param_count >= V2_MIN_PARAMS:
            for candidate in candidates:
                cand_name = _get_meta_value(candidate, 'name', self._field_mapping) or ''
                if cand_name and generic_pattern:
                    # FIX 2026-06-02: build candidate generic ONLY from remapped keys
                    cand_params = {k: _find_field_value(candidate, k) for k in remapped.keys()}
                    cand_generic = self._get_generic_pattern(standard, item_type, cand_params, mask)
                    cand_param_count = len([v for v in cand_params.values() if v is not None and str(v).strip()])
                    # Count ALL non-empty params in ENS candidate (for ratio check)
                    total_cand_params = len([v for v in candidate.values() if v is not None and str(v).strip() and not str(k).startswith('_') for k, v in candidate.items()])
                    # Both patterns must have enough params to prevent degenerate matches
                    # Also: input params should cover most of ENS params (avoid matching subset)
                    ratio = param_count / max(cand_param_count, 1)
                    if (cand_param_count >= V2_MIN_PARAMS and
                            generic_pattern.strip() == cand_generic.strip() and
                            ratio >= V2_MIN_RATIO):
                        best_match = candidate
                        best_score = 1.0
                        match_type = 'exact'
                        match_type_ru = 'Точное совпадение (V2 generic)'
                        logger.info("[ЭТАП 2] ✓ Точное совпадение (V2): %s", cand_name[:60])
                        break
                    elif generic_pattern.strip() == cand_generic.strip():
                        logger.debug("[ЭТАП 2] V2 generic match rejected: ratio=%.2f < %.2f (input=%d, cand=%d)",
                                     ratio, V2_MIN_RATIO, param_count, cand_param_count)
        else:
            logger.debug("[ЭТАП 2] V2 generic skipped: %d params < %d", param_count, V2_MIN_PARAMS)

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
                    logger.info("[ЭТАП 2] ✓ Совпадение с вариантом покрытия: %s", variant)
                    break

        # 2026-06-03 16:05 (МСК, UTC+3): coating substitution → exact_after_substitution
        # Раньше match_type='fuzzy' даже после замены покрытия, теперь — точное
        if substitution_info and best_match and match_type in ('fuzzy', 'fuzzy_coating_variant'):
            match_type = 'exact_substitution'
            match_type_ru = 'Точное совпадение (после замены покрытия)'
            best_score = 1.0
            logger.info("[ЭТАП 2] ✓ Точное совпадение (после замены покрытия): %s",
                       substitution_info.get('corrected', '?'))

        processing_time = (time.time() - start_time) * 1000

        if not best_match:
            logger.info("[ЭТАП 2] ✗ Совпадение не найдено для %s", text[:60])
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
                    'mask_id': getattr(mask, 'id', None),
                    'mask_pattern': getattr(mask, 'pattern', None),
                    'debug_candidates': debug_candidates[:5],
                },
                item_type=item_type,
                standard=standard
            )

        # Build ENS match dict
        ens_match = {
            'code': _get_meta_value(best_match, 'code', self._field_mapping),
            'name': _get_meta_value(best_match, 'name', self._field_mapping),
            'mdm_key': best_match.get('mdm_key'),
            'score': best_score,
            'type': match_type,
            **best_match
        }

        return self._build_parametric_result(
            text, mask, extracted, remapped, best_match,
            best_score, match_type, match_type_ru, debug_candidates,
            substitution_info, start_time
        )

    def _build_parametric_result(self, text, mask, extracted, remapped,
                                 best_match, best_score, match_type, match_type_ru,
                                 debug_candidates, substitution_info, start_time):
        """Build ProcessingResult from best match.

        FEAT 2026-06-02: extracted from _parametric_match to reuse for exact_name match.
        """
        standard_info = extracted.get('standard_info')
        standard = canonicalize_standard(standard_info.normalized) if standard_info else ""
        item_type = extracted.get('item_type', '')
        processing_time = (time.time() - start_time) * 1000

        # FIX 2026-06-05: применить фиксы перед КАЖДЫМ использованием маски (идемпотентно)
        mask.pattern = self._apply_mask_fixes(mask.pattern)
        params = self.parametric_client.extract_params(text, mask.pattern)

        # Build ENS match
        ens_match = None
        if best_match:
            ens_match = {
                'code': _get_meta_value(best_match, 'code', self._field_mapping),
                'name': _get_meta_value(best_match, 'name', self._field_mapping),
            }

        # Get ENS params from best_match
        ens_params_from_match = {}
        for key in remapped.keys():
            val = _find_field_value(best_match, key)
            if val is not None:
                ens_params_from_match[key] = val

        # Calculate confidence (capped at 1.0)
        confidence = min(best_score, 1.0)
        if match_type == 'exact_name':
            confidence = 1.0
        elif match_type == 'exact':
            confidence = 1.0
        elif match_type == 'fuzzy':
            confidence = min(best_score, 1.0)
        elif match_type == 'fuzzy_coating_variant':
            confidence = min(best_score * 0.95, 1.0)

        # Determine success
        matching_cfg = _get_matching_config()
        success = confidence >= matching_cfg.success_threshold

        # 2026-06-03 16:00 (МСК, UTC+3): mismatched required params (кроме покрытия) → success=False
        # FIX 2026-06-09: длина — не безусловно soft, а с tolerance (±10% или ±2мм)
        def _soft_param_mismatch_ok(param_name: str, extracted_val, ens_val) -> bool:
            """Check if mismatch for a soft param is within tolerance."""
            if param_name == 'длина':
                try:
                    f1 = float(str(extracted_val).replace(',', '.'))
                    f2 = float(str(ens_val).replace(',', '.'))
                    if f2 == 0:
                        return f1 == 0
                    rel_diff = abs(f1 - f2) / abs(f2)
                    abs_diff = abs(f1 - f2)
                    # Tolerance: < 10% relative AND < 20mm absolute
                    return rel_diff < 0.10 and abs_diff < 20.0
                except (ValueError, TypeError):
                    return False
            return False  # unknown soft param → not OK

        SOFT_PARAMS = {'длина'}  # length with tolerance, not unconditional pass
        best_debug = None
        if success and debug_candidates:
            best_cd_list = [cd for cd in debug_candidates if cd.get('is_best')]
            if best_cd_list:
                best_cd = best_cd_list[0]
                # 1. Прямые mismatches (soft params проверяем tolerance)
                mismatched = best_cd.get('params_mismatched', {})
                for param_name in mismatched:
                    if param_name in SOFT_PARAMS:
                        # Soft param: check tolerance
                        comp = best_cd.get('params_comparison', {}).get(param_name, {})
                        if _soft_param_mismatch_ok(param_name, comp.get('extracted'), comp.get('ens_value')):
                            logger.info("[РЕЗУЛЬТАТ] Soft param '%s' within tolerance: %s ≈ %s",
                                        param_name, comp.get('extracted'), comp.get('ens_value'))
                            continue  # within tolerance → OK
                    if param_name != 'покрытие':
                        success = False
                        logger.info("[РЕЗУЛЬТАТ] Сброс success: mismatched '%s'", param_name)
                        break
                # 2. "exact (in name)" с несовпадающим ENS-значением (soft params проверяем tolerance)
                if success:
                    comparisons = best_cd.get('params_comparison', {})
                    for param_name, comp in comparisons.items():
                        if (comp.get('status') == 'exact (in name)' and
                            comp.get('extracted') != comp.get('ens_value')):
                            if param_name in SOFT_PARAMS:
                                if _soft_param_mismatch_ok(param_name, comp.get('extracted'), comp.get('ens_value')):
                                    continue  # within tolerance → OK
                            if param_name != 'покрытие':
                                success = False
                                logger.info("[РЕЗУЛЬТАТ] Сброс success: '%s' exact (in name) but %s ≠ %s",
                                            param_name, comp.get('extracted'), comp.get('ens_value'))
                                break

        # Build details
        # FEAT 2026-06-04: топ-5 кандидатов от лучших к худшим (score↓, exact_count↓)
        sorted_debug = sorted(
            debug_candidates,
            key=lambda x: (-x['score'], -x.get('exact_count', 0))
        )[:5] if debug_candidates else []
        details = {
            'match_type': match_type,
            'match_type_ru': match_type_ru,
            'mask_id': getattr(mask, 'id', None),
            'mask_pattern': getattr(mask, 'pattern', None),
            'debug_candidates': sorted_debug,
        }

        if substitution_info:
            details['coating_substitution'] = substitution_info
            if best_match:
                success = True
                confidence = max(confidence, matching_cfg.success_threshold)
                logger.info("[РЕЗУЛЬТАТ] Принудительный успех (подстановка покрытия)")

        bm_code = _get_meta_value(best_match, 'code', self._field_mapping) if best_match else 'N/A'
        bm_name = _get_meta_value(best_match, 'name', self._field_mapping) if best_match else 'N/A'

        # FEAT 2026-06-04: ens_params_mask — извлечь параметры из ens_name той же маской
        # (по аналогии с extract_params из text)
        ens_params_from_mask = {}
        if best_match and mask and getattr(mask, 'pattern', None):
            ens_name_for_mask = bm_name
            if ens_name_for_mask and ens_name_for_mask != 'N/A':
                try:
                    ens_extracted_for_mask = self.parametric_client.extract_params(ens_name_for_mask, mask.pattern)
                    ens_params_from_mask = self._remap_params(ens_extracted_for_mask, mask.params)
                except Exception:
                    pass

        # Normalize mask.params (fallback: dict с ключами из mask.params)
        mask_params_norm = mask.params
        if isinstance(mask_params_norm, str):
            try:
                mask_params_norm = json.loads(mask_params_norm)
            except Exception:
                mask_params_norm = {}
        elif isinstance(mask_params_norm, list):
            flat = []
            for item in mask_params_norm:
                if isinstance(item, str):
                    try:
                        parsed = json.loads(item)
                        if isinstance(parsed, list):
                            flat.extend(parsed)
                        else:
                            flat.append(parsed)
                    except Exception:
                        flat.append(item)
                else:
                    flat.append(item)
            mask_params_norm = {p: None for p in flat} if flat else {}
        if success:
            logger.info("[РЕЗУЛЬТАТ] ✓ Успех: code=%s, name=%s, confidence=%.3f",
                        bm_code, bm_name[:50], confidence)
        else:
            logger.info("[РЕЗУЛЬТАТ] ✗ Неудача: лучший code=%s, score=%.3f, порог=%.2f",
                        bm_code, best_score, matching_cfg.success_threshold)

        # FIX 2026-06-04: mask_params_norm может быть list — нормализуем в dict
        _mask_params_norm = mask_params_norm if isinstance(mask_params_norm, dict) else {}
        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH,
            success=success,
            params=params,
            ens_params=ens_params_from_match,
            ens_params_mask=ens_params_from_mask or _mask_params_norm,
            ens_match=ens_match,
            confidence=round(confidence, 3),
            processing_time_ms=round(processing_time, 2),
            details=details,
            item_type=item_type,
            standard=standard
        )

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
                confidence = min(best.get('score', 0.0), 1.0)
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