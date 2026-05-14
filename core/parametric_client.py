"""
Parametric ENS Client Module
Level 6: Параметрическое сопоставление с использованием масок.

VERSION: 2026-05-14

LAST_FIXES:
 2026-05-14 10:32 UTC+3 — индексация ENS по (стандарт, тип) + O(1) поиск по коду + кэш compiled regex (производительность)
 2026-05-08 13:10 UTC+3 — strict_union_keys: учитывается в _compare_param_sets (false=skip пустые ключи)
 2026-05-08 11:45 UTC+3 — _remap_params ДО _find_in_ens: exact match теперь работает на уровне parametric_client
 2026-05-07 12:10 UTC+3 — кэширование _find_in_ens (хеш по params/standard/item_type)
 2026-05-07 12:10 UTC+3 — debug_per_parameter: лог _compare_param_sets под контролем конфига
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

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

    default = {}  # Загружается из ens_column_mapping.yaml

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
    """
    Token-based Jaccard similarity для текстовых параметров.
    Удаляет цифры перед сравнением (Кд9.хр → кд, хр).

    Examples:
    'Хим.Пас' ~ 'Хим.Пас' = 1.0
    'Кд' ~ 'Кд9.хр' = 0.5 (кд общий, хр нет)
    'Кд.фос' ~ 'Кд.фос.окс' = 0.67
    'Окс.Фос' ~ 'Фос.Окс' = 1.0
    """
    import re
    if not a or not b:
        return 0.0

    a_str = str(a).lower().strip()
    b_str = str(b).lower().strip()

    # Exact match
    if a_str == b_str:
        return 1.0

    # Extract tokens, remove digits
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
    ens_params: Dict[str, Any]  # параметры из индекса ENS
    ens_params_mask: Dict[str, Any]  # параметры из ens_name по маске
    score: float
    match_type: str  # 'exact', 'partial', 'fuzzy'
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

    Архитектура:
    1. Проверка наличия маски в MaskDatabase
    2. Применение маски для извлечения параметров
    3. Поиск по параметрам в ENSIndex
    4. Расчет score сопоставления
    """

    def __init__(
        self,
        mask_db,
        ens_index_path: Optional[str] = None,
        use_tfidf_fallback: bool = True
    ):
        """
        Инициализация клиента.

        Args:
            mask_db: Экземпляр MaskDatabase
            ens_index_path: Путь к индексу ЕСН
            use_tfidf_fallback: Использовать TF-IDF fallback
        """
        self.mask_db = mask_db
        self.ens_index_path = ens_index_path
        self.use_tfidf_fallback = use_tfidf_fallback
        self._ens_index = None

        # Кэш для _find_in_ens по хешу (params, standard, item_type)
        self._find_in_ens_cache: Dict[str, Any] = {}

        # === ПРОИЗВОДИТЕЛЬНОСТЬ: индексы и кэши ===
        self._ens_by_standard_type: Dict[Tuple[str, str], List[Dict]] = {}
        self._ens_by_code: Dict[str, Dict] = {}
        self._pattern_cache: Dict[str, Any] = {}

        if ens_index_path and Path(ens_index_path).exists():
            self._load_ens_index()
            self._build_indexes()

    def _build_indexes(self):
        """Построить индексы для O(1) / O(small N) доступа к ENS."""
        items = self._ens_index.get('items', []) if self._ens_index else []
        logger.info("Building ENS indexes for %d items...", len(items))

        for item in items:
            std = self._normalize_standard(item.get('нтд') or item.get('standard', ''))
            itype = str(item.get('тип_изделия') or item.get('наименование_типа', '')).upper().strip()

            # Индекс по (стандарт, тип)
            key = (std, itype)
            self._ens_by_standard_type.setdefault(key, []).append(item)

            # Индекс по коду
            code = str(item.get('код', '')).strip()
            mdm = str(item.get('mdm_key', '')).strip()
            if code:
                self._ens_by_code[code] = item
            if mdm and mdm != code:
                self._ens_by_code[mdm] = item

        total_keys = len(self._ens_by_standard_type)
        logger.info("ENS indexes built: %d (std,type) groups, %d codes", total_keys, len(self._ens_by_code))

    def _get_candidates_by_index(self, std_norm: Optional[str], query_type: Optional[str]) -> List[Dict]:
        """Получить кандидатов ENS через индекс (вместо полного скана)."""
        candidates = []

        if std_norm and query_type:
            key = (std_norm, query_type)
            candidates = self._ens_by_standard_type.get(key, [])

        if not candidates and std_norm:
            # Fallback: все записи с этим стандартом
            for key, group in self._ens_by_standard_type.items():
                if key[0] == std_norm:
                    candidates.extend(group)

        if not candidates and query_type:
            # Fallback: все записи с этим типом
            for key, group in self._ens_by_standard_type.items():
                if key[1] == query_type:
                    candidates.extend(group)

        if not candidates:
            # Ultimate fallback (редко): все записи
            candidates = self._ens_index.get('items', []) if self._ens_index else []

        return candidates

    def _get_compiled_pattern(self, pattern: str) -> Any:
        """Кэширование compiled regex."""
        if pattern not in self._pattern_cache:
            self._pattern_cache[pattern] = re.compile(pattern, re.IGNORECASE)
        return self._pattern_cache[pattern]

    def _get_ens_by_code(self, ens_code: str) -> Optional[Dict]:
        """O(1) поиск по коду ЕНС."""
        return self._ens_by_code.get(str(ens_code).strip())

    def _relax_pattern(self, pattern: str, standard: str = None) -> str:
        r"""
        Исправления regex-масок для корректного matching'а:
        1. Латинская t/a → русская т/а + \s* после типа изделия
        2. \s* после )? опциональной группы
        3. \s* перед \( в опциональной группе
        4. \d+(?:\.\d+)? → \d+(?:[.,]\d+)? (запятая как разделитель)
        5. ОСТ1 → ОСТ\s*1 (пробел между ОСТ и цифрой)
        6. Винт: вынести номинальный_диаметр_резьбы из опциональной группы
        7. Шайба: добавить толщину между наружным_диаметром и покрытием
        """
        relaxed = pattern

        # 1. Латинская t/a → русская т/а (без double \s*)
        _ru_t = chr(0x0442)  # русская т
        _ru_a = chr(0x0430)  # русская а
        _ru_b = chr(0x0431)  # русская б
        _ru_g = chr(0x0433)  # русская г

        for latin, cyr in [('Винt', 'Вин' + _ru_t), ('Болt', 'Бол' + _ru_t),
                           ('Шайba', 'Шай' + _ru_b + _ru_a), ('Гайka', 'Гай' + _ru_g + _ru_a)]:
            if latin in relaxed:
                has_s = relaxed[relaxed.find(latin) + len(latin):].startswith(r'\s*')
                relaxed = relaxed.replace(latin, cyr + (r'\s*' if not has_s else ''), 1)

        # Fallback: fix any remaining mixed-script type names
        relaxed = relaxed.replace('Винt', 'Вин' + _ru_t)
        relaxed = relaxed.replace('Болt', 'Бол' + _ru_t)
        relaxed = relaxed.replace('Шайb', 'Шай' + _ru_b)
        relaxed = relaxed.replace('Гайk', 'Гай' + _ru_g)

        # 2. )?(?P< → )?\s*(?P<
        relaxed = re.sub(r'\)\?\(\?P<', lambda m: r')?\s*(?P<', relaxed)

        # 4. \d+(?:\.\d+)? → \d+(?:[.,]\d+)? (через .replace)
        relaxed = relaxed.replace(r'\d+(?:\\.\d+)?', r'\d+(?:[.,]\d+)?')
        # Also try with single backslash (masks loaded from DB)
        relaxed = relaxed.replace(r'\d+(?:\.\d+)?', r'\d+(?:[.,]\d+)?')

        # 5. ОСТ1 → ОСТ\s*1
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

        # 8. Винт: добавить \s* перед \( в группе исполнения
        relaxed = relaxed.replace(
            r'(?:\u005c(\u005cs*(?P<исполнение>',
            r'(?:\s*\(\s*(?P<исполнение>',
            1
        )

        # 9. Исполнение: сделать скобки опциональными, разрешить пробел/дефис после
        relaxed = relaxed.replace('$$?', r'\(')
        relaxed = relaxed.replace(r'$$\s*)', r'\)\s*)')
        relaxed = relaxed.replace(r'$$\s*(?P<', r'\)\s*(?P<')
        relaxed = relaxed.replace('$$', r'\(')
        relaxed = relaxed.replace('$(?P<', r'\((?P<')
        relaxed = relaxed.replace(r'$\s*)', r'\)\s*)')
        relaxed = relaxed.replace(r'$\s*(?P<', r'\)\s*(?P<')
        relaxed = relaxed.replace(r'\$\$?\s*)', r'\)\s*)')
        relaxed = relaxed.replace(r'\$\$?(?P<', r'\((?P<')
        relaxed = relaxed.replace(r'\$(?P<', r'\((?P<')
        relaxed = relaxed.replace(r'\)\s*', r'\)[-\s]*')
        if '$' in relaxed.rstrip('$').rstrip():
            relaxed = re.sub(r'\$(?=\s|[-\s]*\(|[-\s]*\d|[-\s]*[A-Z])', r'\)', relaxed)
            relaxed = re.sub(r'\$(?=\s*$)', r'\)', relaxed)

        # 14. Разделители между параметрами: \s+ → [-\s]+
        relaxed = relaxed.replace(r')\s+(?P<', r')[-\s]+(?P<')
        relaxed = relaxed.replace(r'\d+\s+\d+', r'\d+[-\s]+\d+')

        # 10. Метрическая резьба: добавить опциональный шаг (x1,25)
        relaxed = relaxed.replace(
            r'(?:M(?P<номинальный_диаметр_резьбы>\d+))',
            r'(?:M(?P<номинальный_диаметр_резьбы>\d+)(?:[xX\u00d7]\d+(?:[.,]\d+)?)?)',
            1
        )

        # 10a. Сделать M опциональным перед номинальный_диаметр_резьбы
        relaxed = relaxed.replace(
            r'(?P<номинальный_диаметр_резьбы>M',
            r'(?P<номинальный_диаметр_резьбы>(?:M)?'
        )

        # 11. Класс поля допуска: ограничить буквами (без цифр)
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

        # 13. Группа прочности: ограничить цифры перед точкой
        relaxed = relaxed.replace(
            r'(?P<группа_класс_прочности>\d+\.\d+)',
            r'(?P<группа_класс_прочности>\d{1,2}(?:\.\d+)?)',
            1
        )

        # 9a. Если маска не содержит суффикс стандарта -- добавить
        has_std = any(s in relaxed for s in ['ОСТ', 'ГОСТ', 'ТУ', 'ISO'])
        if standard and not has_std:
            std_suffix = None
            if standard.startswith('ОСТ 1'):
                parts = standard.split('1', 1)
                if len(parts) > 1:
                    std_suffix = r'-ОСТ\s*1\s*' + parts[1].strip().replace(' ', r'\s*')
            elif standard.startswith('ГОСТ'):
                std_suffix = r'\s*ГОСТ\s*' + standard.replace('ГОСТ', '').strip().replace(' ', r'\s*')
            elif standard.startswith('ТУ'):
                std_suffix = r'\s*ТУ\s*' + standard.replace('ТУ', '').strip().replace(' ', r'\s*')
            elif standard.startswith('ISO'):
                std_suffix = r'\s*ISO\s*' + standard.replace('ISO', '').strip().replace(' ', r'\s*')

            if std_suffix:
                relaxed = relaxed.rstrip('$').rstrip() + std_suffix + r'\s*$'

        # Проверяем что результат - валидный regex
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
        """
        Параметрическое сопоставление.

        Args:
            text: Текст номенклатуры
            standard: Стандарт (если известен)
            item_type: Тип изделия (если известен)

        Returns:
            ParametricMatch
        """
        # Шаг 1: Получаем маску
        mask = None
        if standard and item_type:
            mask = self.mask_db.get_mask(standard, item_type)

        if not mask:
            # Fallback: пробуем извлечь стандарт и тип из текста
            from parsers.standard_extractor import get_standard_extractor
            extractor = get_standard_extractor()
            extracted = extractor.extract_all(text)

            std_info = extracted.get('standard_info')
            extracted_type = extracted.get('item_type')

            if std_info and extracted_type:
                mask = self.mask_db.get_mask(
                    std_info.normalized,
                    extracted_type
                )

        # Шаг 2: Применяем маску (с релаксацией для совместимости)
        # Если pattern передан напрямую - используем его (skip БД)
        effective_mask = mask
        if pattern and not mask:
            # Создаём временный mask-like объект
            from types import SimpleNamespace
            effective_mask = SimpleNamespace(
                pattern=pattern, required=[], id=-1
            )

        if effective_mask:
            # Используем стандарт из параметра match() если в маске пусто
            effective_standard = getattr(effective_mask, 'standard', None) or standard
            relaxed_pattern = self._relax_pattern(pattern or effective_mask.pattern, standard=effective_standard)
            extracted_params = self._apply_mask(relaxed_pattern, text, standard=effective_standard)

            if extracted_params:
                # Шаг 3: Ищем в ЕСН
                required = getattr(effective_mask, 'required', [])
                # Если required - JSON строка, парсим
                if isinstance(required, str):
                    try:
                        import json as _json
                        required = _json.loads(required)
                    except (ValueError, TypeError):
                        required = []

                # Remap params для поиска в ENS (ENS индекс использует нормализованные имена)
                search_params = self._remap_params(extracted_params)
                match_result = self._find_in_ens(search_params, required, standard=standard, text=text, item_type=item_type)

                if match_result:
                    # Извлекаем данные из ENS
                    ens_code = match_result.get('код')
                    ens_name = match_result.get('полное_наименование') or match_result.get('наименование')
                    ens_params_from_index = {k: v for k, v in match_result.items() if k not in ['_match_score', '_match_type', 'код', 'полное_наименование', 'наименование', 'mdm_key', 'нтд']}

                    # Парсим ens_name той же маской → ens_params_mask (raw, те же имена групп)
                    ens_params_mask = None
                    if ens_name:
                        try:
                            ens_params_mask = self._apply_mask(relaxed_pattern, str(ens_name), standard=effective_standard)
                        except Exception as e:
                            logger.debug("[match] Failed to parse ens_name='%s': %s", ens_name, e)

                    # Новая логика сопоставления
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
                # Если ЕСН не нашёл - всё равно возвращаем extracted params
                # (confidence от regex match)
                required = getattr(effective_mask, 'required', [])
                if required:
                    regex_confidence = self._calculate_confidence(extracted_params, required)
                else:
                    # Если required не заданы - считаем confidence по всем non-None полям
                    non_none = sum(1 for v in extracted_params.values() if v is not None)
                    regex_confidence = non_none / len(extracted_params) if extracted_params else 0.0

                logger.debug("[REGEX_ONLY] extracted=%s, required=%s, confidence=%.2f", extracted_params, required, regex_confidence)

                # Всегда возвращаем regex_only при непустых extracted (threshold 0.1)
                if regex_confidence > 0.1:
                    return ParametricMatch(
                        ens_code=None,
                        ens_name=None,
                        mdm_key=None,
                        matched_params=extracted_params,
                        score=0.0,
                        match_type='regex_only',
                        confidence=regex_confidence,
                        details={
                            'mask_id': getattr(effective_mask, 'id', None),
                            'pattern': getattr(effective_mask, 'pattern', None)
                        }
                    )

        # Fallback: TF-IDF поиск
        # NOTE: _tfidf_fallback не реализован в ParametricENSClient,
        # используйте automated_processor для TF-IDF fallback
        # if self.use_tfidf_fallback and self._ens_index:
        #     return self._tfidf_fallback(text)

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
        """Применение regex маски к тексту."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            match = compiled.search(text)

            if match:
                return match.groupdict()
            else:
                logger.debug("[_apply_mask] Regex did not match. Pattern: %r, Text: %r", pattern[:150], text[:80])
        except re.error as e:
            # Логируем маску для диагностики - показываем первые 200 символов
            logger.error("Invalid mask pattern: %s. Pattern (first 200 chars): %r", e, pattern[:200])

        return None

    @staticmethod
    def _remap_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Переименование неправильных имён групп из LLM-генерированных масок
        в корректные ENS-имена параметров.
        """
        if not params:
            return params
        remapped = dict(params)
        aliases = {
            'наружный_диаметр_диаметр_вписа': 'номинальный_диаметр_резьбы',
            'наружный_диаметр': 'номинальный_диаметр_резьбы',
            'диаметр_вписанной_окружности': 'номинальный_диаметр_резьбы',
            'd_вп': 'номинальный_диаметр_резьбы',
            'наружный_диаметр_головки': 'диаметр_головки',
            'диаметр_резьбы': 'номинальный_диаметр_резьбы',
        }
        for wrong, correct in aliases.items():
            if wrong in remapped:
                value = remapped.pop(wrong)
                if correct in remapped and correct == 'номинальный_диаметр_резьбы':
                    remapped['длина'] = remapped[correct]
                remapped[correct] = value
                logger.debug("[REMAP] %s → %s: %s", wrong, correct, value)
        return remapped

    def _expand_coating_variants(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Expand coating search variants.
        When coating is 'Кд', also try 'Кд6', 'Кд9.фос.окс' etc.
        Returns list of param dicts to try.
        """
        variants = [params]
        coating = params.get('покрытие')
        if not coating:
            return variants

        coating_str = str(coating).strip().lower()

        # Кд variants
        if coating_str in ('кд', 'кд.'):
            for variant in ['Кд6', 'Кд9', 'Кд6.фос', 'Кд9.фос',
                              'Кд6.фос.окс', 'Кд9.фос.окс', 'Кд.фос.окс']:
                expanded = dict(params)
                expanded['покрытие'] = variant
                variants.append(expanded)

        return variants

    def _find_in_ens(self, params: Dict[str, Any], required: List[str], standard: Optional[str] = None, text: Optional[str] = None, item_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Поиск по параметрам в индексе ЕНС. Fallback: точное совпадение наименования с проверкой типа."""
        match, _ = self._find_in_ens_debug(params, required, standard, text, item_type)
        return match

    def _find_in_ens_debug(self, params: Dict[str, Any], required: List[str], standard: Optional[str] = None, text: Optional[str] = None, item_type: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], List[Dict]]:
        """
        Поиск по параметрам в индексе ЕНС с подробным debug-выводом.
        Возвращает: (best_match, debug_candidates_list)
        """
        debug_candidates = []
        if not self._ens_index or 'items' not in self._ens_index:
            logger.warning("[_find_in_ens] ENS index not loaded!")
            return None, debug_candidates

        # === КЭШИРОВАНИЕ ===
        # Кэшируем результат по хешу (params, required, standard, item_type)
        cache_key = "{}:{}:{}:{}".format(
            hash(str(sorted(params.items()))),
            hash(str(required)),
            standard,
            item_type
        )
        if cache_key in self._find_in_ens_cache:
            cached = self._find_in_ens_cache[cache_key]
            logger.debug("[_find_in_ens] Cache hit for %s/%s", standard, item_type)
            return cached['match'], cached['debug']

        best_match = None
        best_score = 0.0

        # Нормализуем запрошенный стандарт и тип для сравнения
        query_std_norm = self._normalize_standard(standard) if standard else None
        query_type = item_type.upper().strip() if item_type else None

        # === ИНДЕКСАЦИЯ: получаем кандидатов вместо полного скана ===
        candidates = self._get_candidates_by_index(query_std_norm, query_type)
        logger.debug("[_find_in_ens] Candidates via index: %d (std=%s, type=%s)", len(candidates), query_std_norm, query_type)

        # Try with coating variants
        param_variants = self._expand_coating_variants(params)

        for try_params in param_variants:
            for item in candidates:
                item_name = item.get('наименование', item.get('полное_наименование', 'N/A'))
                item_code = item.get('код', item.get('mdm_key', 'N/A'))

                # Фильтр по стандарту (нтд) - обязательное совпадение
                if query_std_norm:
                    item_std = item.get('нтд') or item.get('standard')
                    item_std_norm = self._normalize_standard(item_std) if item_std else None
                    if item_std_norm and query_std_norm != item_std_norm:
                        continue

                # Фильтр по типу изделия (если указан)
                if query_type:
                    item_type_field = str(item.get('тип_изделия', '') or item.get('наименование_типа', '')).upper().strip()
                    if item_type_field and item_type_field != query_type:
                        continue

                score = self._calculate_match_score(try_params, required, item)

                # Собираем debug info
                skip_fields = {'_match_score', '_match_type', 'код', 'полное_наименование', 'наименование', 'mdm_key', 'нтд', 'тип_изделия', 'наименование_типа'}
                ens_params = {k: v for k, v in item.items() if k not in skip_fields and not k.startswith('_') and v is not None and str(v).strip()}

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
                    # Подробный debug per-parameter (управляется через config.yaml matching.debug_per_parameter)
                    if _get_matching_config().debug_per_parameter:
                        source_str = ", ".join("{}={}".format(k, v) for k, v in debug_entry['source_params'].items())
                        ens_str = ", ".join("{}={}".format(k, v) for k, v in debug_entry['ens_params'].items())
                        logger.debug("[_find_in_ens] Candidate '%s' (code=%s): score=%.3f", debug_entry['name'][:50], debug_entry['ens_code'], score)
                        logger.debug("[_find_in_ens] Source params: %s", source_str)
                        logger.debug("[_find_in_ens] ENS params: %s", ens_str)

                if score > best_score:
                    best_score = score
                    best_match = item

        # Fallback: точное совпадение наименования (с проверкой типа)
        fallback_matched = False
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
                        fallback_matched = True
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

        # Сортируем по score убыванию
        debug_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Итоговый debug: top candidates (только при debug_per_parameter=true)
        if debug_candidates and _get_matching_config().debug_per_parameter:
            top_n = min(5, len(debug_candidates))
            logger.debug("[_find_in_ens] Top %d parametric candidates:", top_n)
            for i, cd in enumerate(debug_candidates[:top_n], 1):
                ens_p_str = ", ".join("{}={}".format(k, v) for k, v in cd.get('ens_params', {}).items())
                logger.debug("[_find_in_ens] #%d: '%s' score=%s, code=%s", i, cd.get('name', '')[:50], cd.get('score', 0), cd.get('ens_code', 'N/A'))
                logger.debug("[_find_in_ens] ENS: %s", ens_p_str)

        if best_match:
            best_match = dict(best_match)
            best_match['_match_score'] = best_score
            best_match['_match_type'] = 'exact' if best_score > 0.9 else 'partial'
            logger.info("[_find_in_ens] RETURNING match: score=%.2f, name=%s", best_score, best_match.get('наименование', '')[:50])
        else:
            logger.info("[_find_in_ens] No match found (best_score=%.2f < 0.7 threshold)", best_score)

        # Сохраняем в кэш перед возвратом
        self._find_in_ens_cache[cache_key] = {
            'match': best_match if best_score >= 0.7 else None,
            'debug': debug_candidates
        }

        if best_score < 0.7:
            return None, debug_candidates

        return best_match, debug_candidates

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Нормализация наименования: убираем пробелы, нижний регистр."""
        return re.sub(r'\s+', '', str(name).lower().strip())

    @staticmethod
    def _normalize_standard(std: Optional[str]) -> str:
        """Нормализация стандарта для сравнения: ОСТ 1 → ОСТ1."""
        if not std:
            return ''
        s = str(std).strip()
        s = re.sub(r'ОСТ\s*1', 'ОСТ1', s)
        return s

    @staticmethod
    def _normalize_ens_value(val: Any) -> Any:
        """Нормализация значения для сравнения с ENS:
        - float 2.0 → int 2
        - str '2.0' → int 2
        - str 'abc' → str 'abc'
        """
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
        """Расчет уверенности в извлечении (заполненность required-полей)."""
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
        Используем только required поля для поиска кандидатов.
        """
        # Извлекаем параметры из ENS записи (без служебных полей)
        skip_fields = {
            '_match_score', '_match_type', 'код', 'полное_наименование',
            'наименование', 'mdm_key', 'нтд', 'тип_изделия', 'наименование_типа',
            'item_type', 'standard'
        }
        ens_params = {k: v for k, v in ens_item.items()
                     if k not in skip_fields and not k.startswith('_')}

        # Берём только required поля из params
        if not required:
            return 0.0
        subset = {k: params[k] for k in required if k in params and params[k] is not None}
        if not subset:
            return 0.0

        return self._compare_param_sets(subset, ens_params)

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
        Возвращает: (score, match_type, details)
        """
        details = {}

        # LEVEL 1: name exact
        if text and ens_name:
            text_norm = self._normalize_name(text)
            ens_norm = self._normalize_name(ens_name)
            if text_norm == ens_norm:
                logger.debug("[_match_v2] LEVEL 1: name EXACT")
                return 1.0, 'name_exact', {'level': 'name_exact'}

        # LEVEL 2: params vs ens_params
        score_ens = self._compare_param_sets(params, ens_params)
        if score_ens >= 0.99:
            logger.debug("[_match_v2] LEVEL 2: params vs ens_params EXACT")
            return 1.0, 'params_ens_exact', {'level': 'params_ens', 'score': score_ens}

        # LEVEL 3: params vs ens_params_mask
        if ens_params_mask:
            score_mask = self._compare_param_sets(params, ens_params_mask)
            if score_mask >= 0.99:
                logger.debug("[_match_v2] LEVEL 3: params vs ens_params_mask EXACT")
                return 1.0, 'params_mask_exact', {'level': 'params_mask', 'score': score_mask}

        best_score = max(score_ens, score_mask if ens_params_mask else 0.0)
        return best_score, 'partial', {'level': 'partial', 'score_ens': score_ens, 'score_mask': score_mask if ens_params_mask else None}

    def _compare_param_sets(self, params_a: Dict[str, Any], params_b: Dict[str, Any]) -> float:
        """
        Сравнение двух наборов параметров.
        Score=1.0 если все НЕ-ПУСТЫЕ параметры из ОБОИХ наборов совпадают.
        Ключи, пустые в ОБОИХ наборах — игнорируются.
        Ключ, пустой только в одном наборе:
        - strict_union_keys=true → mismatch (return 0.0)
        - strict_union_keys=false → skip (continue)
        """
        if not params_a or not params_b:
            return 0.0

        config = _get_matching_config()
        debug_detail = config.debug_per_parameter
        strict_mode = getattr(config, 'strict_union_keys', False)

        # Параметры, которые не участвуют в сравнении (метаданные/служебные)
        skip_params = {'тип_изделия', 'item_type', 'standard', 'нтд'}

        # Сравниваем по ОБЪЕДИНЕНИЮ ключей
        all_keys = set(params_a.keys()) | set(params_b.keys())
        checked = 0
        matched = 0

        for param in all_keys:
            if param in skip_params:
                continue

            val_a = params_a.get(param)
            val_b = params_b.get(param)
            str_a = str(val_a).lower().strip() if val_a is not None else ''
            str_b = str(val_b).lower().strip() if val_b is not None else ''

            # Если оба пустые — пропускаем (не влияет на score)
            if not str_a and not str_b:
                continue

            # Если один пустой (None/''), а другой нет
            if not str_a or not str_b:
                if strict_mode:
                    # Строгий режим: пустой ключ = mismatch
                    if debug_detail:
                        logger.debug("[_compare] KEY MISSING (strict): %s = '%s' vs '%s'", param, val_a, val_b)
                    return 0.0
                else:
                    # Нестрогий режим: пустой ключ = skip (не влияет на score)
                    if debug_detail:
                        logger.debug("[_compare] KEY SKIP (non-strict): %s = '%s' vs '%s'", param, val_a, val_b)
                    continue

            checked += 1

            # Сравниваем значения
            if param == 'покрытие':
                norm_a = self._normalize_coating(str_a)
                norm_b = self._normalize_coating(str_b)
                sim = _text_similarity(norm_a, norm_b)
                if sim < 0.8:
                    if debug_detail:
                        logger.debug("[_compare] COATING MISMATCH: '%s' vs '%s' (sim=%.2f)", norm_a, norm_b, sim)
                    return 0.0
                matched += 1
            else:
                try:
                    num_a = float(str_a.replace(',', '.'))
                    num_b = float(str_b.replace(',', '.'))
                    if num_a != num_b:
                        if debug_detail:
                            logger.debug("[_compare] NUM MISMATCH: %s vs %s", val_a, val_b)
                        return 0.0
                    matched += 1
                except ValueError:
                    if str_a != str_b:
                        if debug_detail:
                            logger.debug("[_compare] STR MISMATCH: '%s' vs '%s'", val_a, val_b)
                        return 0.0
                    matched += 1

        # Если нечего сравнивать (все поля пустые) — mismatch
        if checked == 0:
            return 0.0

        return 1.0

    def _normalize_coating(self, coating: str) -> str:
        """
        Нормализация покрытия:
        - Убирает технологические коды: Кд3 → Кд, Ц9 → Ц
        - Убирает суффиксы: Кд3.хр → Кд
        """
        if not coating:
            return coating
        coating_str = str(coating).strip().lower()
        # Убираем цифры после базового покрытия
        base = re.sub(r'^(кд|ц|окс|фос|н|ан|хим|пас|бп|неп)\d+', r'\1', coating_str)
        # Убираем суффиксы .хр, .фос
        base = re.sub(r'\.(хр|фос|окс|пас)$', '', base)
        return base

    def _normalize_name(self, name: str) -> str:
        """Нормализация наименования: убираем пробелы, нижний регистр."""
        return re.sub(r'\s+', '', str(name).lower().strip())

    def _tfidf_fallback(self, text: str) -> 'ParametricMatch':
        """TF-IDF fallback — возвращает пустой результат."""
        return ParametricMatch(
            ens_code=None, ens_name=None, mdm_key=None,
            score=0.0, match_type='failed', confidence=0.0
        )