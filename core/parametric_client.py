# =============================================================================
# FILE: core/parametric_client.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 COMMITS (UTC+3):
# 2026-05-21 08:23:07 51f335da 21.05.2026
# 2026-05-21 08:05:56 ee843b22 21.05.2026
# 2026-05-20 17:47:49 19e8ca02 20.05.2026
# 2026-05-20 17:39:23 b00c4b25 20.05.2026
# 2026-05-20 17:31:34 66c66c93 20.05.2026
# =============================================================================
# FIX 2026-05-21 15:15 UTC+3:
#   1. _find_in_ens threshold: 0.7 -> 0.5
#   2. _compare_param_sets: fractional scoring (matched/checked)
# =============================================================================
"""
Parametric ENS Client Module
Level 6: Параметрическое сопоставление с использованием масок.

VERSION: 2026-05-21

LAST_FIXES:
 2026-05-21 15:15 UTC+3 — Исправление падения качества распознавания:
   - _find_in_ens threshold понижен с 0.7 до 0.5
   - _compare_param_sets возвращает fractional score (matched/checked)
 2026-05-21 08:50 UTC+3 — _normalize_standard теперь использует canonicalize_standard
 (ОСТ 1 с пробелом). Ранее ОСТ 1 → ОСТ1, что ломало exact-match с масками БД.
"""

import re
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from utils.standard_utils import canonicalize_standard

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
                debug_per_parameter = True
            _matching_config = _FallbackMatchingConfig()
    return _matching_config

# Кэш для empty_equivalent_values
_empty_equiv_cache: Optional[Dict[str, List[str]]] = None

def _load_empty_equivalent_values() -> Dict[str, List[str]]:
    """Загрузка значений, эквивалентных пустым, из ens_column_mapping.yaml."""
    global _empty_equiv_cache
    if _empty_equiv_cache is not None:
        return _empty_equiv_cache

    default = {}
    try:
        import yaml
        from pathlib import Path
        for path in ['config/ens_column_mapping.yaml', 'ens_column_mapping.yaml']:
            if Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                equiv = data.get('empty_equivalent_values', {})
                if equiv:
                    _empty_equiv_cache = {k: [str(v).strip() for v in vals]
                                          for k, vals in equiv.items() if vals}
                    return _empty_equiv_cache
                break
    except Exception as e:
        logger.warning("Failed to load empty_equivalent_values: %s", e)

    _empty_equiv_cache = default
    return _empty_equiv_cache

def _text_similarity(a: str, b: str) -> float:
    """Token-based Jaccard similarity для текстовых параметров."""
    import re
    if not a or not b:
        return 0.0
    a_str = str(a).lower().strip()
    b_str = str(b).lower().strip()
    if a_str == b_str:
        return 1.0

    def _extract_tokens(text):
        raw = re.findall(r'[a-zA-Zа-яА-Я0-9]+', text)
        cleaned = []
        for t in raw:
            letters = re.sub(r'[0-9]', '', t)
            if letters:
                cleaned.append(letters)
        return set(cleaned)

    tokens_a = _extract_tokens(a_str)
    tokens_b = _extract_tokens(b_str)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)

def _is_empty_equivalent(field: str, value: Any) -> bool:
    """Проверяет, является ли значение эквивалентным пустому/None."""
    if value is None:
        return True
    val_str = str(value).strip()
    if not val_str:
        return True
    equiv_values = _load_empty_equivalent_values()
    empty_vals = equiv_values.get(field, [])
    return val_str.lower() in [v.lower() for v in empty_vals]

@dataclass
class ParametricMatch:
    """Результат параметрического сопоставления."""
    ens_code: Optional[str]
    ens_name: Optional[str]
    mdm_key: Optional[str]
    matched_params: Dict[str, Any]
    ens_params: Dict[str, Any]
    ens_params_mask: Dict[str, Any]
    score: float
    match_type: str
    confidence: float
    details: Dict[str, Any]

    def __init__(
        self,
        ens_code: Optional[str] = None,
        ens_name: Optional[str] = None,
        mdm_key: Optional[str] = None,
        matched_params: Optional[Dict[str, Any]] = None,
        ens_params: Optional[Dict[str, Any]] = None,
        ens_params_mask: Optional[Dict[str, Any]] = None,
        score: float = 0.0,
        match_type: str = 'failed',
        confidence: float = 0.0,
        details: Optional[Dict[str, Any]] = None
    ):
        self.ens_code = ens_code
        self.ens_name = ens_name
        self.mdm_key = mdm_key
        self.matched_params = matched_params or {}
        self.ens_params = ens_params or {}
        self.ens_params_mask = ens_params_mask or {}
        self.score = score
        self.match_type = match_type
        self.confidence = confidence
        self.details = details or {}

class ParametricENSClient:
    """
    Клиент для параметрического поиска по ЕСН.
    """

    def __init__(
        self,
        mask_db,
        ens_index_path: Optional[str] = None,
        use_tfidf_fallback: bool = True,
        skip_fields: Optional[List[str]] = None
    ):
        """
        Инициализация клиента.
        """
        self.mask_db = mask_db
        self.ens_index_path = ens_index_path
        self.use_tfidf_fallback = use_tfidf_fallback
        self._ens_index = None
        self._find_in_ens_cache: Dict[str, Any] = {}
        self._ens_by_standard_type: Dict[Tuple[str, str], List[Dict]] = {}
        self._ens_by_code: Dict[str, Dict] = {}
        self._pattern_cache: Dict[str, Any] = {}
        self._find_in_ens_lock = threading.Lock()
        self._pattern_lock = threading.Lock()

        # === SKIP FIELDS: загрузка из конфига или fallback ===
        self._skip_fields: set = self._load_skip_fields(skip_fields)
        logger.info("[SKIP_FIELDS] Loaded %d skip fields", len(self._skip_fields))

        if ens_index_path and Path(ens_index_path).exists():
            self._load_ens_index()
            self._build_indexes()

    def _load_skip_fields(self, skip_fields: Optional[List[str]]) -> set:
        """Загрузка списка служебных полей: приоритет аргумент > конфиг > fallback."""
        if skip_fields is not None:
            return set(skip_fields)

        try:
            from config.settings import get_settings
            settings = get_settings()
            if hasattr(settings, 'output') and hasattr(settings.output, 'ens_params_skip_fields'):
                cfg_fields = settings.output.ens_params_skip_fields
                if cfg_fields:
                    return set(cfg_fields)
        except Exception as e:
            logger.debug("[SKIP_FIELDS] Failed to load from settings: %s", e)

        return {
            '_id', '_index', '_source', 'id', 'created_at', 'updated_at',
            'hash', 'pattern_hash',
            'код', 'mdm_key', 'единицы_измерения', 'наименование_типа.1',
            'полное_наименование', 'наименование', 'нтд', 'тип',
            'вести_учет_по_характеристикам', 'гражданская_продукция',
            'заблокировано', 'наименование_1', 'организация_корпорации',
            'автор', 'дата_создания', 'пометка_удаления',
            'базовая_единица_измерения', 'соответствие_тр_тс',
            'габаритные_размеры_масса', 'специальная_приемка',
            'ссылка', 'классификатор_енс', 'классификатор_енс_код',
            'оквэд2', 'оквэд2_код', 'окпд2', 'окпд2_код',
            'дата_последнего_изменения', 'автор_последнего_изменения',
            'марка_материала_1', 'нормативный_документ',
            'нормативный_документ_1', 'тип_изделия', 'наименование_типа',
            'item_type', 'standard', '_match_score', '_match_type'
        }

    def _build_indexes(self):
        """Построить индексы для O(1) / O(small N) доступа к ENS."""
        items = self._ens_index.get('items', []) if self._ens_index else []
        logger.info("Building ENS indexes for %d items...", len(items))
        for item in items:
            std = self._normalize_standard(item.get('нтд') or item.get('standard', ''))
            itype = str(item.get('тип_изделия') or item.get('наименование_типа', '')).upper().strip()
            key = (std, itype)
            self._ens_by_standard_type.setdefault(key, []).append(item)
            code = str(item.get('код', '')).strip()
            mdm = str(item.get('mdm_key', '')).strip()
            if code:
                self._ens_by_code[code] = item
            if mdm and mdm != code:
                self._ens_by_code[mdm] = item
        logger.info("ENS indexes built: %d (std,type) groups, %d codes",
                    len(self._ens_by_standard_type), len(self._ens_by_code))

    def _get_candidates_by_index(self, std_norm: Optional[str], query_type: Optional[str]) -> List[Dict]:
        """Получить кандидатов ENS через индекс (вместо полного скана)."""
        candidates = []
        if std_norm and query_type:
            key = (std_norm, query_type)
            candidates = self._ens_by_standard_type.get(key, [])
        if not candidates and std_norm:
            for key, group in self._ens_by_standard_type.items():
                if key[0] == std_norm:
                    candidates.extend(group)
        if not candidates and query_type:
            for key, group in self._ens_by_standard_type.items():
                if key[1] == query_type:
                    candidates.extend(group)
        if not candidates:
            candidates = self._ens_index.get('items', []) if self._ens_index else []
        return candidates

    def _get_compiled_pattern(self, pattern: str) -> Any:
        """Кэширование compiled regex (thread-safe)."""
        with self._pattern_lock:
            if pattern not in self._pattern_cache:
                self._pattern_cache[pattern] = re.compile(pattern, re.IGNORECASE)
            return self._pattern_cache[pattern]

    def _get_ens_by_code(self, ens_code: str) -> Optional[Dict]:
        """O(1) поиск по коду ЕНС."""
        return self._ens_by_code.get(str(ens_code).strip())

    def _relax_pattern(self, pattern: str, standard: str = None) -> str:
        r"""
        Исправления regex-масок для корректного matching'а.
        """
        relaxed = pattern

        # 1. Латинская t/a → русская т/а (без double \s*)
        _ru_t = chr(0x0442)
        _ru_a = chr(0x0430)
        _ru_b = chr(0x0431)
        _ru_g = chr(0x0433)

        for latin, cyr in [('Винt', 'Вин' + _ru_t), ('Болt', 'Бол' + _ru_t),
                           ('Шайba', 'Шай' + _ru_b + _ru_a), ('Гайka', 'Гай' + _ru_g + _ru_a)]:
            if latin in relaxed:
                has_s = relaxed[relaxed.find(latin) + len(latin):].startswith(r'\s*')
                relaxed = relaxed.replace(latin, cyr + (r'\s*' if not has_s else ''), 1)

        relaxed = relaxed.replace('Винt', 'Вин' + _ru_t)
        relaxed = relaxed.replace('Болt', 'Бол' + _ru_t)
        relaxed = relaxed.replace('Шайb', 'Шай' + _ru_b)
        relaxed = relaxed.replace('Гайk', 'Гай' + _ru_g)

        # 2. )?(?P< → )? \s*(?P<
        relaxed = re.sub(r'\)\?\(?P<', lambda m: r')?\s*(?P<', relaxed)

        # 4. \d+(?:\.\d+)? → \d+(?:[.,]\d+)?
        relaxed = relaxed.replace(r'\d+(?:\\.\d+)?', r'\d+(?:[.,]\d+)?')
        relaxed = relaxed.replace(r'\d+(?:\.\d+)?', r'\d+(?:[.,]\d+)?')

        # 5. ОСТ1 → ОСТ\s*1 (обратная совместимость со старыми масками)
        if r'ОСТ\s*1' not in relaxed:
            relaxed = re.sub(r'ОСТ1', lambda m: r'ОСТ\s*1', relaxed)

        # 6. Винт: вынести номинальный_диаметр_резьбы из опциональной группы
        _opt_fix_old = r')\s*\)\s*-(?P<номинальный_диаметр_резьбы>'
        _opt_fix_new = r')\s*\)\s*-)?(?P<номинальный_диаметр_резьбы>'
        if _opt_fix_old in relaxed:
            relaxed = relaxed.replace(_opt_fix_old, _opt_fix_new, 1)
            relaxed = relaxed.replace(r'))?\s*(?P<длина>', r')\s*-(?P<длина>', 1)

        # 7. Шайба: пропустить промежуточное число между наружным диаметром и покрытием
        _shaiba_old = r'(?P<наружный_диаметр_диаметр_вписа>\d+)\-?(?P<покрытие>[\w.]+)?'
        _shaiba_new = r'(?P<наружный_диаметр_диаметр_вписа>\d+)(?:\-\d+)?\-(?P<покрытие>[\w.]+)'
        if _shaiba_old in relaxed:
            relaxed = relaxed.replace(_shaiba_old, _shaiba_new, 1)
        _shaiba_old2 = r'(?P<наружный_диаметр_диаметр_вписа>\d+(?:[.,]\d+)?)\-(?P<покрытие>[\w.]+)?'
        _shaiba_new2 = r'(?P<наружный_диаметр_диаметр_вписа>\d+(?:[.,]\d+)?)(?:\-\d+)?\-(?P<покрытие>[\w.]+)'
        if _shaiba_old2 in relaxed:
            relaxed = relaxed.replace(_shaiba_old2, _shaiba_new2, 1)

        # 8. Винт: добавить \s* перед \( в группе исполнение
        relaxed = relaxed.replace(
            r'(?:\u005c(\u005cs\*(?P<исполнение>',
            r'(?:\s*\(\s*(?P<исполнение>',
            1
        )

        # 9. Исполнение: сделать скобки опциональными
        relaxed = relaxed.replace('$$?', r'\(')
        relaxed = relaxed.replace(r'$$\s*)', r'\)\s*)')
        relaxed = relaxed.replace(r'$$\s*(?P<', r'\)\s*(?P<')
        relaxed = relaxed.replace('$$', r'\(')
        relaxed = relaxed.replace('$(?P<', r'\((?P<')
        relaxed = relaxed.replace(r'$\s*)', r'\)\s*)')
        relaxed = relaxed.replace(r'$\s*(?P<', r'\)\s*(?P<')
        relaxed = relaxed.replace(r'\$$\?\s*)', r'\)\s*)')
        relaxed = relaxed.replace(r'\$$?(?P<', r'\((?P<')
        relaxed = relaxed.replace(r'\$(?P<', r'\((?P<')
        relaxed = relaxed.replace(r'\)\s*', r'\)[-\s]*')
        if '$' in relaxed.rstrip('$').rstrip():
            relaxed = re.sub(r'\$(?=\s|[-\s]*\(|[-\s]*\d|[-\s]*[A-Z])', r'\)', relaxed)
            relaxed = re.sub(r'\$(?=\s*$)', r'\)', relaxed)

        # 14. Разделители между параметрами
        relaxed = relaxed.replace(r')\s+(?P<', r')[-\s]+(?P<')
        relaxed = relaxed.replace(r'\d+\s+\d+', r'\d+[-\s]+\d+')

        # 10. Метрическая резьба: добавить опциональный шаг
        relaxed = relaxed.replace(
            r'(?:M(?P<номинальный_диаметр_резьбы>\d+))',
            r'(?:M(?P<номинальный_диаметр_резьбы>\d+)(?:[xX\u00d7]\d+(?:[.,]\d+)?)?)',
            1
        )

        # 10a. Сделать M опциональным
        relaxed = relaxed.replace(
            r'(?P<номинальный_диаметр_резьбы>M',
            r'(?P<номинальный_диаметр_резьбы>(?:M)?'
        )

        # 11. Класс поля допуска
        relaxed = relaxed.replace(
            r'(?P<класс_поле_допуска>[\d+][\w]*)',
            r'(?P<класс_поле_допуска>[\d+][a-zA-Z\u0430-\u044f\u0410-\u042f]*)',
            1
        )

        # 12. Удалить конфликтующую группу tipo_rezby=M
        relaxed = relaxed.replace(
            r'(?:\s*(?P<тип_резьбы>M))?\s*[-\s]*',
            r'\s*[-\s]*',
            1
        )

        # 13. Группа прочности
        relaxed = relaxed.replace(
            r'(?P<группа_класс_прочности>\d+\.\d+)',
            r'(?P<группа_класс_прочности>\d{1,2}(?:\.\d+)?)',
            1
        )

        # 9a. Если маска не содержит суффикс стандарта -- добавить
        has_std = any(s in relaxed for s in ['ОСТ', 'ГОСТ', 'ТУ', 'ISO'])
        if standard and not has_std:
            std_suffix = None
            canon_std = canonicalize_standard(standard)
            if canon_std.startswith('ОСТ 1'):
                parts = canon_std.split('1', 1)
                if len(parts) > 1:
                    std_suffix = r'-ОСТ\s*1\s*' + parts[1].strip().replace(' ', r'\s*')
            elif canon_std.startswith('ГОСТ'):
                std_suffix = r'\s*ГОСТ\s*' + canon_std.replace('ГОСТ', '').strip().replace(' ', r'\s*')
            elif canon_std.startswith('ТУ'):
                std_suffix = r'\s*ТУ\s*' + canon_std.replace('ТУ', '').strip().replace(' ', r'\s*')
            elif canon_std.startswith('ISO'):
                std_suffix = r'\s*ISO\s*' + canon_std.replace('ISO', '').strip().replace(' ', r'\s*')

            if std_suffix:
                relaxed = relaxed.rstrip('$').rstrip() + std_suffix + r'\s*$'

        try:
            re.compile(relaxed)
        except re.error as e:
            logger.warning(
                "_relax_pattern produced invalid regex: %s. "
                "Falling back to original pattern. "
                "Original (50 chars): %r. "
                "Relaxed (50 chars): %r",
                e, pattern[:50], relaxed[:50]
            )
            return pattern

        return relaxed

    def _load_ens_index(self):
        """Загрузка индекса ЕСН."""
        try:
            import pickle
            with open(self.ens_index_path, 'rb') as f:
                data = pickle.load(f)
            self._ens_index = data
            logger.info("Loaded ENS index from %s", self.ens_index_path)
        except Exception as e:
            logger.warning("Failed to load ENS index: %s", e)

    def match(
        self,
        text: str,
        standard: Optional[str] = None,
        item_type: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> ParametricMatch:
        """Параметрическое сопоставление."""
        canon_std = canonicalize_standard(standard) if standard else None
        mask = None
        if canon_std and item_type:
            mask = self.mask_db.get_mask(canon_std, item_type)

        if not mask:
            from parsers.standard_extractor import get_standard_extractor
            extractor = get_standard_extractor()
            extracted = extractor.extract_all(text)
            std_info = extracted.get('standard_info')
            extracted_type = extracted.get('item_type')
            if std_info and extracted_type:
                mask = self.mask_db.get_mask(
                    canonicalize_standard(std_info.normalized),
                    extracted_type
                )

        effective_mask = mask
        if pattern and not mask:
            from types import SimpleNamespace
            effective_mask = SimpleNamespace(pattern=pattern, required=[], id=-1)

        if effective_mask:
            effective_standard = getattr(effective_mask, 'standard', None) or canon_std
            relaxed_pattern = self._relax_pattern(
                pattern or effective_mask.pattern,
                standard=effective_standard
            )
            extracted_params = self._apply_mask(relaxed_pattern, text, standard=effective_standard)

            if extracted_params:
                required = getattr(effective_mask, 'required', [])
                if isinstance(required, str):
                    try:
                        import json as _json
                        required = _json.loads(required)
                    except (ValueError, TypeError):
                        required = []

                meta_keys = {'тип_изделия', 'item_type', 'standard', 'нтд', 'нтд_1',
                             'наименование', 'полное_наименование', 'код', 'mdm_key'}
                search_params = {k: v for k, v in self._remap_params(extracted_params).items()
                                 if k not in meta_keys and v is not None}
                match_result = self._find_in_ens(
                    search_params, required,
                    standard=canon_std, text=text, item_type=item_type
                )

                if match_result:
                    ens_code = match_result.get('код')
                    ens_name = match_result.get('полное_наименование') or match_result.get('наименование')
                    ens_params_from_index = {k: v for k, v in match_result.items()
                                             if k not in self._skip_fields and not k.startswith('_')}

                    ens_params_mask = None
                    if ens_name:
                        try:
                            ens_params_mask = self._apply_mask(
                                relaxed_pattern, str(ens_name), standard=effective_standard
                            )
                        except Exception as e:
                            logger.debug("[match] Failed to parse ens_name='%s': %s", ens_name, e)

                    final_score, match_type, details = self._calculate_match_score_v2(
                        text=text,
                        ens_name=ens_name,
                        params=extracted_params,
                        ens_params=ens_params_from_index,
                        ens_params_mask=ens_params_mask,
                        required=required
                    )

                    return ParametricMatch(
                        ens_code=ens_code,
                        ens_name=ens_name,
                        mdm_key=match_result.get('mdm_key'),
                        matched_params=extracted_params,
                        ens_params=ens_params_from_index,
                        ens_params_mask=ens_params_mask or {},
                        score=final_score,
                        match_type=match_type,
                        confidence=final_score,
                        details={
                            'mask_id': getattr(effective_mask, 'id', None),
                            'pattern': getattr(effective_mask, 'pattern', None),
                            **details
                        }
                    )

                required = getattr(effective_mask, 'required', [])
                if required:
                    regex_confidence = self._calculate_confidence(extracted_params, required)
                else:
                    non_none = sum(1 for v in extracted_params.values() if v is not None)
                    regex_confidence = non_none / len(extracted_params) if extracted_params else 0.0

                logger.debug("[REGEX_ONLY] extracted=%s, required=%s, confidence=%.2f",
                             extracted_params, required, regex_confidence)

                if regex_confidence > 0.1:
                    return ParametricMatch(
                        ens_code=None,
                        ens_name=None,
                        mdm_key=None,
                        matched_params=extracted_params,
                        ens_params_mask=extracted_params,
                        score=0.0,
                        match_type='regex_only',
                        confidence=regex_confidence,
                        details={
                            'mask_id': getattr(effective_mask, 'id', None),
                            'pattern': getattr(effective_mask, 'pattern', None)
                        }
                    )

        return ParametricMatch(
            ens_code=None,
            mdm_key=None,
            matched_params={},
            score=0.0,
            match_type='failed',
            confidence=0.0,
            details={'error': 'No mask found and no fallback available'}
        )

    def _apply_mask(self, pattern: str, text: str, standard: str = None) -> Optional[Dict[str, Any]]:
        """Применение regex маски к тексту. Нормализует 'None'/'null'/'nan' → None."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            match = compiled.search(text)

            if match:
                result = {}
                for k, v in match.groupdict().items():
                    if v is None or str(v).strip().lower() in ('none', 'null', 'nan', ''):
                        result[k] = None
                    else:
                        result[k] = v
                return result
            else:
                logger.debug("[_apply_mask] Regex did not match. Pattern: %r, Text: %r",
                             pattern[:150], text[:80])
        except re.error as e:
            logger.error("Invalid mask pattern: %s. Pattern (first 200 chars): %r", e, pattern[:200])

        return None

    @staticmethod
    def _remap_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Переименование параметров — теперь pass-through (без хардкода имён столбцов)."""
        return dict(params) if params else params

    def _expand_coating_variants(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand coating search variants."""
        variants = [params]
        coating = params.get('покрытие')
        if not coating:
            return variants

        coating_str = str(coating).strip().lower()

        if coating_str in ('кд', 'кд.'):
            for variant in ['Кд6', 'Кд9', 'Кд6.фос', 'Кд9.фос',
                            'Кд6.фос.окс', 'Кд9.фос.окс', 'Кд.фос.окс']:
                expanded = dict(params)
                expanded['покрытие'] = variant
                variants.append(expanded)

        return variants

    def _find_in_ens(self, params: Dict[str, Any], required: List[str],
                     standard: Optional[str] = None, text: Optional[str] = None,
                     item_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Поиск по параметрам в индексе ЕНС."""
        match, _ = self._find_in_ens_debug(params, required, standard, text, item_type)
        return match

    def _find_in_ens_debug(self, params: Dict[str, Any], required: List[str],
                           standard: Optional[str] = None, text: Optional[str] = None,
                           item_type: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], List[Dict]]:
        """
        Поиск по параметрам в индексе ЕНС с подробным debug-выводом.
        """
        debug_candidates = []
        if not self._ens_index or 'items' not in self._ens_index:
            logger.warning("[_find_in_ens] ENS index not loaded!")
            return None, debug_candidates

        cache_key = "{}:{}:{}:{}".format(
            hash(str(sorted(params.items()))),
            hash(str(required)),
            standard,
            item_type
        )
        with self._find_in_ens_lock:
            if cache_key in self._find_in_ens_cache:
                cached = self._find_in_ens_cache[cache_key]
                logger.debug("[_find_in_ens] Cache hit for %s/%s", standard, item_type)
                return cached['match'], cached['debug']

        best_match = None
        best_score = 0.0

        query_std_norm = self._normalize_standard(standard) if standard else None
        query_type = item_type.upper().strip() if item_type else None

        candidates = self._get_candidates_by_index(query_std_norm, query_type)
        logger.debug("[_find_in_ens] Candidates via index: %d (std=%s, type=%s)",
                     len(candidates), query_std_norm, query_type)

        param_variants = self._expand_coating_variants(params)

        for try_params in param_variants:
            for item in candidates:
                item_name = item.get('наименование', item.get('полное_наименование', 'N/A'))
                item_code = item.get('код', item.get('mdm_key', 'N/A'))

                if query_std_norm:
                    item_std = item.get('нтд') or item.get('standard')
                    item_std_norm = self._normalize_standard(item_std) if item_std else None
                    if item_std_norm and query_std_norm != item_std_norm:
                        continue

                if query_type:
                    item_type_field = str(item.get('тип_изделия', '') or item.get('наименование_типа', '')).upper().strip()
                    if item_type_field and item_type_field != query_type:
                        continue

                score = self._calculate_match_score(try_params, required, item)

                ens_params = {k: v for k, v in item.items()
                              if k not in self._skip_fields and not k.startswith('_')
                              and v is not None and str(v).strip()}

                debug_entry = {
                    'stage': 'parametric',
                    'name': item_name[:60] if isinstance(item_name, str) else str(item_name)[:60],
                    'ens_code': str(item_code),
                    'score': round(score, 3),
                    'source_params': {k: str(v) for k, v in try_params.items() if v is not None},
                    'ens_params': {k: str(v)[:50] for k, v in list(ens_params.items())[:8]},
                    'method': '_calculate_match_score',
                }

                if score > 0:
                    debug_candidates.append(debug_entry)
                    if _get_matching_config().debug_per_parameter:
                        source_str = ", ".join("{}={}".format(k, v) for k, v in debug_entry['source_params'].items())
                        ens_str = ", ".join("{}={}".format(k, v) for k, v in debug_entry['ens_params'].items())
                        logger.debug("[_find_in_ens] Candidate '%s' (code=%s): score=%.3f",
                                     debug_entry['name'][:50], debug_entry['ens_code'], score)
                        logger.debug("[_find_in_ens] Source params: %s", source_str)
                        logger.debug("[_find_in_ens] ENS params: %s", ens_str)

                if score > best_score:
                    best_score = score
                    best_match = item

        # Fallback: точное совпадение наименования
        if best_score < 0.99 and text:
            text_norm = self._normalize_name(text)
            for item in candidates:
                if query_std_norm:
                    item_std = item.get('нтд') or item.get('standard')
                    item_std_norm = self._normalize_standard(item_std) if item_std else None
                    if item_std_norm and query_std_norm != item_std_norm:
                        continue
                if query_type:
                    item_type_field = str(item.get('тип_изделия', '') or item.get('наименование_типа', '')).upper().strip()
                    if item_type_field and item_type_field != query_type:
                        continue
                for name_field in ['полное_наименование', 'наименование']:
                    item_name = item.get(name_field)
                    if item_name and self._normalize_name(str(item_name)) == text_norm:
                        best_match = item
                        best_score = 1.0
                        debug_candidates.append({
                            'stage': 'name_fallback',
                            'name': str(item_name)[:60],
                            'ens_code': str(item.get('код', item.get('mdm_key', 'N/A'))),
                            'score': 1.0,
                            'method': 'name_exact',
                            'matched_name': str(item_name)[:60],
                        })
                        break
                if best_score >= 0.99:
                    break

        debug_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

        if debug_candidates and _get_matching_config().debug_per_parameter:
            top_n = min(5, len(debug_candidates))
            logger.debug("[_find_in_ens] Top %d parametric candidates:", top_n)
            for i, cd in enumerate(debug_candidates[:top_n], 1):
                ens_p_str = ", ".join("{}={}".format(k, v) for k, v in cd.get('ens_params', {}).items())
                logger.debug("[_find_in_ens] #%d: '%s' score=%s, code=%s",
                             i, cd.get('name', '')[:50], cd.get('score', 0), cd.get('ens_code', 'N/A'))
                logger.debug("[_find_in_ens] ENS: %s", ens_p_str)

        if best_match:
            best_match = dict(best_match)
            best_match['_match_score'] = best_score
            best_match['_match_type'] = 'exact' if best_score > 0.9 else 'partial'
            logger.info("[_find_in_ens] RETURNING match: score=%.2f, name=%s",
                        best_score, best_match.get('наименование', '')[:50])
        else:
            logger.info("[_find_in_ens] No match found (best_score=%.2f < 0.5 threshold)", best_score)

        with self._find_in_ens_lock:
            self._find_in_ens_cache[cache_key] = {
                'match': best_match if best_score >= 0.5 else None,
                'debug': debug_candidates
            }

        # FIX 1: threshold 0.7 -> 0.5
        if best_score < 0.5:
            return None, debug_candidates

        return best_match, debug_candidates

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Нормализация наименования: убираем пробелы, нижний регистр, сортируем токены покрытия."""
        name = re.sub(r'\s+', '', str(name).lower().strip())
        def _sort_coating_tokens(m):
            tokens = m.group(0).split('.')
            tokens = [t.strip() for t in tokens if t.strip()]
            tokens.sort()
            return '.'.join(tokens)
        name = re.sub(r'([a-zа-яё]+(?:\.[a-zа-яё]+)+)', _sort_coating_tokens, name)
        return name

    @staticmethod
    def _normalize_standard(std: Optional[str]) -> str:
        """Нормализация стандарта для сравнения (канонический вид с пробелом)."""
        return canonicalize_standard(std)

    @staticmethod
    def _normalize_ens_value(val: Any) -> Any:
        """Нормализация значения для сравнения с ENS."""
        if isinstance(val, float):
            if val == int(val):
                return int(val)
            return val
        if isinstance(val, str):
            try:
                f = float(val)
                if f == int(f):
                    return int(f)
                return f
            except ValueError:
                return val.strip()
        return val

    def _calculate_confidence(self, params: Dict[str, Any], required: List[str]) -> float:
        """Расчет уверенности в извлечении."""
        if isinstance(required, str):
            try:
                import json as _json
                required = _json.loads(required)
            except (ValueError, TypeError):
                required = []
        if not required:
            return 0.0
        found = sum(1 for p in required if p in params and params[p] is not None)
        return found / len(required)

    def _calculate_match_score(
        self,
        params: Dict[str, Any],
        required: List[str],
        ens_item: Dict[str, Any]
    ) -> float:
        """
        Сравнение params с ENS записью через _compare_param_sets.
        Fallback: если поля записи пустые — парсим наименование ENS той же маской из БД.
        """
        ens_params = {k: v for k, v in ens_item.items()
                      if k not in self._skip_fields and not k.startswith('_')}

        if not required:
            return 0.0
        subset = {k: params[k] for k in required if k in params and params[k] is not None}
        if not subset:
            return 0.0

        score = self._compare_param_sets(subset, ens_params)

        if score == 0 and self.mask_db:
            std = self._normalize_standard(ens_item.get('нтд') or ens_item.get('standard', ''))
            itype = str(ens_item.get('тип_изделия') or ens_item.get('наименование_типа', '')).upper().strip()
            mask = self.mask_db.get_mask(std, itype)
            if mask and getattr(mask, 'pattern', None):
                for name_field in ['полное_наименование', 'наименование']:
                    ens_name = ens_item.get(name_field)
                    if not ens_name:
                        continue
                    try:
                        parsed = self._apply_mask(mask.pattern, str(ens_name), standard=std)
                        if parsed:
                            score = self._compare_param_sets(subset, parsed)
                            if score > 0:
                                break
                    except Exception:
                        continue

        return score

    def _calculate_match_score_v2(
        self,
        text: str,
        ens_name: Optional[str],
        params: Dict[str, Any],
        ens_params: Dict[str, Any],
        ens_params_mask: Optional[Dict[str, Any]],
        required: List[str]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Новая логика (3 уровня):
        1. name_exact: text vs ens_name
        2. params vs ens_params
        3. params vs ens_params_mask
        """
        details = {}

        if text and ens_name:
            text_norm = self._normalize_name(text)
            ens_norm = self._normalize_name(ens_name)
            if text_norm == ens_norm:
                logger.debug("[_match_v2] LEVEL 1: name EXACT")
                return 1.0, 'name_exact', {'level': 'name_exact'}

        score_ens = self._compare_param_sets(params, ens_params)
        if score_ens >= 0.99:
            logger.debug("[_match_v2] LEVEL 2: params vs ens_params EXACT")
            return 1.0, 'params_ens_exact', {'level': 'params_ens', 'score': score_ens}

        if ens_params_mask:
            score_mask = self._compare_param_sets(params, ens_params_mask)
            if score_mask >= 0.99:
                logger.debug("[_match_v2] LEVEL 3: params vs ens_params_mask EXACT")
                return 1.0, 'params_mask_exact', {'level': 'params_mask', 'score': score_mask}

        best_score = max(score_ens, score_mask if ens_params_mask else 0.0)
        return best_score, 'partial', {'level': 'partial', 'score_ens': score_ens,
                                        'score_mask': score_mask if ens_params_mask else None}

    @staticmethod
    def _match_param_keys(key_a: str, keys_b: List[str]) -> Optional[str]:
        """Fuzzy matching имени параметра key_a со списком ключей keys_b."""
        if not key_a or not keys_b:
            return None
        tokens_a = [t for t in key_a.lower().split('_') if len(t) >= 3]
        if not tokens_a:
            return None
        best_match = None
        best_score = 0.0
        for key_b in keys_b:
            tokens_b = [t for t in key_b.lower().split('_') if len(t) >= 3]
            if not tokens_b:
                continue
            matched = 0
            for ta in tokens_a:
                for tb in tokens_b:
                    if ta == tb:
                        matched += 1
                        break
                    if len(ta) >= 4 and len(tb) >= 4:
                        if ta.startswith(tb) or tb.startswith(ta):
                            matched += 1
                            break
            score = matched / max(len(tokens_a), len(tokens_b)) if max(len(tokens_a), len(tokens_b)) > 0 else 0.0
            if score > best_score:
                best_score = score
                best_match = key_b
        if best_score >= 0.5:
            return best_match
        return None

    def _compare_param_sets(self, params_a: Dict[str, Any], params_b: Dict[str, Any]) -> float:
        """
        Сравнение двух наборов параметров с fuzzy matching ключей.
        Score = matched / checked (fractional, 0.0..1.0).
        """
        if not params_a or not params_b:
            return 0.0

        config = _get_matching_config()
        debug_detail = config.debug_per_parameter
        strict_mode = getattr(config, 'strict_union_keys', False)

        skip_params = {'тип_изделия', 'item_type', 'standard', 'нтд', 'нтд_1',
                       'наименование', 'полное_наименование', 'код', 'mdm_key'}

        keys_b_available = set(params_b.keys()) - skip_params
        matched_map = {}
        used_b = set()

        for key_a in params_a.keys():
            if key_a in skip_params:
                continue
            val_a = params_a[key_a]
            if val_a is None or str(val_a).strip() == '':
                continue
            if key_a in keys_b_available and key_a not in used_b:
                matched_map[key_a] = key_a
                used_b.add(key_a)
                continue
            candidates = [k for k in keys_b_available if k not in used_b]
            best_b = self._match_param_keys(key_a, candidates)
            if best_b:
                matched_map[key_a] = best_b
                used_b.add(best_b)
            elif strict_mode:
                if debug_detail:
                    logger.debug("[_compare] KEY MISSING (strict): %s has no match in ENS", key_a)
                return 0.0

        if strict_mode:
            for key_b in keys_b_available:
                if key_b in used_b:
                    continue
                val_b = params_b[key_b]
                if val_b is None or str(val_b).strip() == '':
                    continue
                found = False
                for key_a in params_a.keys():
                    if key_a in skip_params:
                        continue
                    val_a = params_a[key_a]
                    if val_a is None or str(val_a).strip() == '':
                        continue
                    if matched_map.get(key_a) == key_b:
                        found = True
                        break
                if not found:
                    if debug_detail:
                        logger.debug("[_compare] KEY MISSING (strict): ENS key %s has no match in extracted", key_b)
                    return 0.0

        checked = 0
        matched = 0

        for key_a, key_b in matched_map.items():
            val_a = params_a[key_a]
            val_b = params_b.get(key_b)
            str_a = str(val_a).lower().strip() if val_a is not None else ''
            str_b = str(val_b).lower().strip() if val_b is not None else ''

            if not str_a and not str_b:
                continue

            if not str_a or not str_b:
                if strict_mode:
                    if debug_detail:
                        logger.debug("[_compare] VALUE MISSING (strict): %s = '%s' vs '%s'", key_a, val_a, val_b)
                    return 0.0
                else:
                    continue

            checked += 1

            is_coating = (key_a == 'покрытие') or (key_b and 'покрытие' in key_b)
            if is_coating:
                norm_a = self._normalize_coating(str_a)
                norm_b = self._normalize_coating(str_b)
                sim = _text_similarity(norm_a, norm_b)
                if sim >= 0.8:
                    matched += 1
                elif debug_detail:
                    logger.debug("[_compare] COATING MISMATCH: '%s' vs '%s' (sim=%.2f)", norm_a, norm_b, sim)
            else:
                try:
                    num_a = float(str_a.replace(',', '.'))
                    num_b = float(str_b.replace(',', '.'))
                    if num_a == num_b:
                        matched += 1
                    elif debug_detail:
                        logger.debug("[_compare] NUM MISMATCH: %s vs %s", val_a, val_b)
                except ValueError:
                    if str_a == str_b:
                        matched += 1
                    elif debug_detail:
                        logger.debug("[_compare] STR MISMATCH: '%s' vs '%s'", val_a, val_b)

        if checked == 0:
            return 0.0

        # FIX 2: fractional scoring вместо бинарного return 1.0
        return matched / checked

    def _normalize_coating(self, coating: str) -> str:
        """
        Нормализация покрытия:
        - Убирает технологические коды: Кд3 → Кд, Ц9 → Ц
        - Сортирует токены: Окс.Фос.ЭФП → окс.эфп.фос
        """
        if not coating:
            return coating
        coating_str = str(coating).strip().lower()
        if '.' in coating_str:
            tokens = coating_str.split('.')
            tokens = [re.sub(r'\d+', '', t) for t in tokens]
            tokens = [t for t in tokens if t]
            tokens.sort()
            return '.'.join(tokens)
        base = re.sub(r'^(кд|ц|окс|фос|н|ан|хим|пас|бп|неп)\d+', r'\1', coating_str)
        return base

    def _normalize_name(self, name: str) -> str:
        """Нормализация наименования: убираем пробелы, нижний регистр, сортируем токены покрытия."""
        name = re.sub(r'\s+', '', str(name).lower().strip())
        def _sort_coating_tokens(m):
            tokens = m.group(0).split('.')
            tokens = [t.strip() for t in tokens if t.strip()]
            tokens.sort()
            return '.'.join(tokens)
        name = re.sub(r'([a-zа-яё]+(?:\.[a-zа-яё]+)+)', _sort_coating_tokens, name)
        return name

    def _tfidf_fallback(self, text: str) -> 'ParametricMatch':
        """TF-IDF fallback — возвращает пустой результат."""
        return ParametricMatch(
            ens_code=None, ens_name=None, mdm_key=None,
            score=0.0, match_type='failed', confidence=0.0
        )