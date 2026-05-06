"""
Parametric ENS Client Module
Level 6: Параметрическое сопоставление с использованием масок.

VERSION: 2025-05-06-fix7 (double-dollar-fix)
LAST_FIX: 2026-05-06 15:15 — STRICT EXACT MATCH with coating permutation: exact for all params, fuzzy for покрытие (token reorder), score=1.0 or 0.0
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

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
        logger.warning(f"Failed to load empty_equivalent_values: {e}")

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
    score: float
    match_type: str  # 'exact', 'partial', 'fuzzy'
    confidence: float
    details: Dict[str, Any]


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

        if ens_index_path and Path(ens_index_path).exists():
            self._load_ens_index()

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

        # 3. REMOVED: was broken - matched (?:(?P< and produced unbalanced parentheses.
        #    If LLM generates (?:( instead of (?:(?: , the pattern is already wrong.
        #    The correct fix is to regenerate the mask, not to hack the regex here.

        # 4. \d+(?:\.\d+)? → \d+(?:[.,]\d+)? (через .replace)
        relaxed = relaxed.replace(r'\\d+(?:\\.\\d+)?', r'\\d+(?:[.,]\\d+)?')
        # Also try with single backslash (masks loaded from DB)
        relaxed = relaxed.replace(r'\d+(?:\.\d+)?', r'\d+(?:[.,]\d+)?')

        # 5. ОСТ1 → ОСТ\s*1
        if r'ОСТ\s*1' not in relaxed:
            relaxed = re.sub(r'ОСТ1', lambda m: r'ОСТ\s*1', relaxed)

        # 6. Винт: вынести номинальный_диаметр_резьбы из опциональной группы
        #    )\s*\)\s*-(?P<номинальный_диаметр_резьбы>  →  )\s*\)\s*-)?(?P<номинальный_диаметр_резьбы>
        _opt_fix_old = r')\s*\)\s*-(?P<номинальный_диаметр_резьбы>'
        _opt_fix_new = r')\s*\)\s*-)?(?P<номинальный_диаметр_резьбы>'
        if _opt_fix_old in relaxed:
            relaxed = relaxed.replace(_opt_fix_old, _opt_fix_new, 1)
            #    Затем: ))?\s*(?P<длина> → )\s*-(?P<длина>
            relaxed = relaxed.replace(r'))?\s*(?P<длина>', r')\s*-(?P<длина>', 1)

        # 7. Шайба: пропустить промежуточное число между наружным диаметром и покрытием
        #    (Шайба 0,5-4-8-Кд → 0,5=диаметр, 4=наружный, 8 пропускается, Кд=покрытие)
        _shaiba_old = r'(?P<наружный_диаметр_диаметр_вписа>\d+)\-?(?P<покрытие>[\w.]+)?'
        _shaiba_new = r'(?P<наружный_диаметр_диаметр_вписа>\d+)(?:\-\d+)?\-(?P<покрытие>[\w.]+)'
        if _shaiba_old in relaxed:
            relaxed = relaxed.replace(_shaiba_old, _shaiba_new, 1)
        # Альтернатива: после rule 4 (с [.,])
        _shaiba_old2 = r'(?P<наружный_диаметр_диаметр_вписа>\d+(?:[.,]\d+)?)\-(?P<покрытие>[\w.]+)?'
        _shaiba_new2 = r'(?P<наружный_диаметр_диаметр_вписа>\d+(?:[.,]\d+)?)(?:\-\d+)?\-(?P<покрытие>[\w.]+)'
        if _shaiba_old2 in relaxed:
            relaxed = relaxed.replace(_shaiba_old2, _shaiba_new2, 1)

        # 8. Винт: добавить \s* перед \( в группе исполнения
        #    "Винт (4)-5-..." -> паттерн "(?:\((?P<исполнение>" без пробела
        relaxed = relaxed.replace(
            r'(?:\u005c(\u005cs*(?P<исполнение>',
            r'(?:\s*\(\s*(?P<исполнение>',
            1
        )

        # 9. Исполнение: сделать скобки опциональными, разрешить пробел/дефис после
        #    LLM иногда генерирует "$$" или "$$?" вместо экранирования скобок.
        #    Обрабатываем ДО одиночных $, иначе regex ломается.
        #    $$?(?P< → \((?P<    (два $ + опциональный ?)
        relaxed = relaxed.replace('$$?', r'\(')
        #    $$\s*) → \)\s*)   (два $ перед \s*)
        relaxed = relaxed.replace(r'$$\s*)', r'\)\s*)')
        #    $$\s*(?P< → \)\s*(?P<
        relaxed = relaxed.replace(r'$$\s*(?P<', r'\)\s*(?P<')
        #    $$ → \(             (оставшиеся два $ подряд)
        relaxed = relaxed.replace('$$', r'\(')

        # 9a. "$" в маске заменяет "\(" или "\)" (баг LLM с экранированием)
        #    Заменяем $(?P< на \((?P<  и $\s*) на \)\s*)
        relaxed = relaxed.replace('$(?P<', r'\((?P<')
        relaxed = relaxed.replace(r'$\s*)', r'\)\s*)')
        # Дополнительно: $\s*(?P< → \)\s*(?P< (если после $ идет следующая группа)
        relaxed = relaxed.replace(r'$\s*(?P<', r'\)\s*(?P<')

        # 9c. \$\$?\s*) → \)\s*) — LLM использует \$ вместо \) (закрывающая скобка)
        relaxed = relaxed.replace(r'\$\$?\s*)', r'\)\s*)')
        # 9d. \$\$?(?P< → \((?P< — LLM использует \$ вместо \( (открывающая скобка)
        relaxed = relaxed.replace(r'\$\$?(?P<', r'\((?P<')
        # 9e. Оставшийся \$(?P< → \((?P< (одиночный экранированный $)
        relaxed = relaxed.replace(r'\$(?P<', r'\((?P<')

        # 9f. После всех замен $ → скобки, применяем [-\s]* к \)\s*
        #     (должно идти ПОСЛЕ 9c/d/e, т.к. они создают новые \)\s*)
        relaxed = relaxed.replace(r'\)\s*', r'\)[-\s]*')

        # 9b. Любой оставшийся $ в середине паттерна (не anchor) — заменить на \)
        #     LLM иногда использует $ как замену \) в произвольных местах
        if '$' in relaxed.rstrip('$').rstrip():
            # Заменяем $ которой не в конце строки
            relaxed = re.sub(r'\$(?=\s|[-\s]*\(|[-\s]*\d|[-\s]*[A-Z])', r'\\)', relaxed)
            # Если остался $ перед концом — тоже заменяем
            relaxed = re.sub(r'\$(?=\s*$)', r'\\)', relaxed)

        # 14. Разделители между параметрами: \s+ → [-\s]+
        #     LLM иногда генерирует пробелы между числами, но в тексте дефисы
        #     "Болт 2 12 44" vs "Болт 2-12-44"
        #     Используем .replace() вместо re.sub чтобы избежать regex escaping hell
        #     Заменяем )\s+(?P< на )[-\s]+(?P<
        relaxed = relaxed.replace(r')\s+(?P<', r')[-\s]+(?P<')
        #     Также \d+\s+\d+ → \d+[-\s]+\d+ (между двумя числовыми группами)
        relaxed = relaxed.replace(r'\d+\s+\d+', r'\d+[-\s]+\d+')

        # 10. Метрическая резьба: добавить опциональный шаг (x1,25)
        #     "M12x1,25" -> M12 + x1,25
        relaxed = relaxed.replace(
            r'(?:M(?P<номинальный_диаметр_резьбы>\d+))',
            r'(?:M(?P<номинальный_диаметр_резьбы>\d+)(?:[xX\u00d7]\d+(?:[.,]\d+)?)?)',
            1
        )

        # 11. Класс поля допуска: ограничить буквами (без цифр)
        #     [\w]* съедает цифры -> [a-zA-Z\u0430-\u044f\u0410-\u042f]*
        relaxed = relaxed.replace(
            r'(?P<класс_поле_допуска>[\d+][\w]*)',
            r'(?P<класс_поле_допуска>[\d+][a-zA-Z\u0430-\u044f\u0410-\u042f]*)',
            1
        )

        # 12. Удалить конфликтующую группу tipo_rezby=M
        #     (?:\s*(?P<tipo_rezby>M))?\s*[-\s]* крадёт M из M12
        relaxed = relaxed.replace(
            r'(?:\s*(?P<тип_резьбы>M))?\s*[-\s]*',
            r'\s*[-\s]*',
            1
        )

        # 13. Группа прочности: ограничить цифры перед точкой
        #     \d+\.\d+ -> 100.58 интерпретирует как длина.100 + группа.58
        #     \d{1,2}(?:\.\d+)? -> 5.8, 10.9, но не 100.58
        relaxed = relaxed.replace(
            r'(?P<группа_класс_прочности>\d+\.\d+)',
            r'(?P<группа_класс_прочности>\d{1,2}(?:\.\d+)?)',
            1
        )

        # 9a. Если маска не содержит суффикс стандарта -- добавить
        #    Болт ...-Окс.Фос.ЭФП$ -> Болт ...-Окс.Фос.ЭФП-ОСТ\s*1\s*31133-80$
        has_std = any(s in relaxed for s in ['ОСТ', 'ГОСТ', 'ТУ', 'ISO'])
        if standard and not has_std:
            std_suffix = None
            # Используем startswith чтобы не спутать ГОСТ с ОСТ
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
                # Убираем $ из конца, добавляем суффикс, возвращаем $
                relaxed = relaxed.rstrip('$').rstrip() + std_suffix + r'\s*$'

        # Проверяем что результат - валидный regex
        try:
            re.compile(relaxed)
        except re.error as e:
            logger.warning(
                f"_relax_pattern produced invalid regex: {e}. "
                f"Falling back to original pattern. "
                f"Original (50 chars): {pattern[:50]!r}. "
                f"Relaxed (50 chars): {relaxed[:50]!r}"
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
            logger.info(f"Loaded ENS index from {self.ens_index_path}")
        except Exception as e:
            logger.warning(f"Failed to load ENS index: {e}")

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

                match_result = self._find_in_ens(extracted_params, required, standard=standard)

                if match_result:
                    return ParametricMatch(
                        ens_code=match_result.get('код'),
                        ens_name=match_result.get('полное_наименование') or match_result.get('наименование'),
                        mdm_key=match_result.get('mdm_key'),
                        matched_params=extracted_params,
                        score=match_result.get('_match_score', 0.0),
                        match_type=match_result.get('_match_type', 'exact'),
                        confidence=self._calculate_confidence(
                            extracted_params,
                            getattr(effective_mask, 'required', [])
                        ),
                        details={
                            'mask_id': getattr(effective_mask, 'id', None),
                            'pattern': getattr(effective_mask, 'pattern', None)
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

                logger.debug(f"[REGEX_ONLY] extracted={extracted_params}, required={required}, confidence={regex_confidence:.2f}")

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
        if self.use_tfidf_fallback and self._ens_index:
            return self._tfidf_fallback(text)

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
                logger.debug(f"[_apply_mask] Regex did not match. Pattern: {pattern[:150]!r}, Text: {text[:80]!r}")
        except re.error as e:
            # Логируем маску для диагностики - показываем первые 200 символов
            logger.error(f"Invalid mask pattern: {e}. Pattern (first 200 chars): {pattern[:200]!r}")

        return None

    def _find_in_ens(self, params: Dict[str, Any], required: List[str], standard: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Поиск по параметрам в индексе ЕСН."""
        if not self._ens_index or 'items' not in self._ens_index:
            return None

        items = self._ens_index['items']
        best_match = None
        best_score = 0.0

        # Нормализуем запрошенный стандарт для сравнения
        query_std_norm = self._normalize_standard(standard) if standard else None

        for item in items:
            # Фильтр по стандарту (нтд) - обязательное совпадение
            if query_std_norm:
                item_std = item.get('нтд') or item.get('standard')
                item_std_norm = self._normalize_standard(item_std) if item_std else None
                if item_std_norm and query_std_norm != item_std_norm:
                    continue  # Пропускаем записи с другим стандартом

            score = self._calculate_match_score(params, required, item)

            if score > best_score:
                best_score = score
                best_match = item
                best_match['_match_score'] = score
                best_match['_match_type'] = 'exact' if score > 0.9 else 'partial'

        # Minimum threshold
        if best_score < 0.7:
            return None

        return best_match

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

    def _calculate_match_score(
        self,
        params: Dict[str, Any],
        required: List[str],
        ens_item: Dict[str, Any]
    ) -> float:
        """
        STRICT EXACT MATCH: все required-параметры должны точно совпадать.
        - Exact match: score=1.0, иначе score=0.0 (all or nothing)
        - Поле есть в ЕНС, но нет в params - игнорируется
        - null в params и отсутствие в ЕНС - считаем совпадением
        """
        if not required:
            return 0.0

        total = 0
        for param in required:
            query_val_raw = params.get(param)
            ens_val_raw = ens_item.get(param)

            query_val = None if query_val_raw is None or str(query_val_raw).strip() == '' else query_val_raw
            ens_val = None if ens_val_raw is None or str(ens_val_raw).strip() == '' else ens_val_raw

            # Случай 1: null ≡ null - OK
            if query_val is None and ens_val is None:
                continue

            # Случай 2: поле в ЕНС, но не в params - игнорируем
            if query_val is None and ens_val is not None:
                continue

            # Случай 3: поле в params, но не в ЕНС - mismatch
            if query_val is not None and ens_val is None:
                logger.debug(f"[_calculate_match_score] param='{param}': MISMATCH (not in ENS)")
                return 0.0

            # Случай 4: оба не None - требуем точное совпадение
            total += 1
            query_val = self._normalize_ens_value(query_val)
            ens_val = self._normalize_ens_value(ens_val)
            query_str = str(query_val).lower().strip()
            ens_str = str(ens_val).lower().strip()

            # Для покрытия допускаем перестановку токенов (Окс.Фос.ЭФП ≡ Фос.Окс.ЭФП)
            if param == 'покрытие':
                sim = _text_similarity(query_str, ens_str)
                if sim >= 0.8:
                    logger.debug(f"[_calculate_match_score] param='{param}': COATING match. sim={sim:.2f}, query='{query_str}', ens='{ens_str}'")
                else:
                    logger.debug(f"[_calculate_match_score] param='{param}': COATING MISMATCH. sim={sim:.2f}, query='{query_str}' != ens='{ens_str}'")
                    return 0.0
            elif query_str != ens_str:
                logger.debug(f"[_calculate_match_score] param='{param}': MISMATCH. query='{query_str}' != ens='{ens_str}'")
                return 0.0
            else:
                logger.debug(f"[_calculate_match_score] param='{param}': EXACT match. val='{query_str}'")

        score = 1.0 if total >= 0 else 0.0
        logger.debug(f"[_calculate_match_score] ALL MATCHED, score=1.0, checked={total}")
        return score

    def _calculate_confidence(self, params: Dict[str, Any], required: List[str]) -> float:
        """Расчет уверенности в извлечении."""
        # Если required - JSON строка, парсим
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

    def _tfidf_fallback(self, text: str) -> ParametricMatch:
        """TF-IDF fallback поиск."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            items = self._ens_index.get('items', [])
            if not items:
                raise ValueError("No items in ENS index")

            # Формируем тексты для индексации
            texts = []
            for item in items:
                name = item.get('полное_наименование') or item.get('наименование', '')
                texts.append(str(name))

            # TF-IDF
            vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char', lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([text])

            # Схожесть
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            best_idx = similarities.argmax()
            best_score = float(similarities[best_idx])

            if best_score > 0.1:
                best_item = dict(items[best_idx])
                best_item['_match_score'] = best_score
                best_item['_match_type'] = 'fuzzy'

                return ParametricMatch(
                    ens_code=best_item.get('код'),
                    ens_name=best_item.get('полное_наименование') or best_item.get('наименование'),
                    mdm_key=best_item.get('mdm_key'),
                    matched_params={},
                    score=best_score,
                    match_type='fuzzy',
                    confidence=best_score,
                    details={'similarity': best_score, 'tfidf_fallback': True}
                )

        except Exception as e:
            logger.warning(f"TF-IDF fallback failed: {e}")

        return ParametricMatch(
            ens_code=None,
            mdm_key=None,
            matched_params={},
            score=0.0,
            match_type='failed',
            confidence=0.0,
            details={'error': 'Fallback failed'}
        )

    def batch_match(
        self,
        texts: List[str],
        standards: Optional[List[str]] = None,
        item_types: Optional[List[str]] = None
    ) -> List[ParametricMatch]:
        """Пакетное сопоставление."""
        results = []

        for i, text in enumerate(texts):
            standard = standards[i] if standards and i < len(standards) else None
            item_type = item_types[i] if item_types and i < len(item_types) else None

            result = self.match(text, standard, item_type)
            results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Статистика клиента."""
        return {
            'mask_db_connected': self.mask_db is not None,
            'ens_index_loaded': self._ens_index is not None,
            'use_tfidf_fallback': self.use_tfidf_fallback
        }