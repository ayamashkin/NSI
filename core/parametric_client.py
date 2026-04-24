"""
Parametric ENS Client Module
Level 6: Параметрическое сопоставление с использованием масок.
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
    Fuzzy similarity для текстовых параметров (покрытие, материал).

    1. Exact match: 'Кд' == 'Кд' = 1.0
    2. Prefix match: 'Кд' ~ 'Кд9.хр' = 0.8 (короткий токен — префикс длинного)
    3. Substring match: 'Хим.Пас' in 'Хим.Пас' = 1.0
    4. Token Jaccard: 'Окс.Фос' ~ 'Фос.Окс' = 1.0
    """
    import re
    if not a or not b:
        return 0.0

    a_str = str(a).lower().strip()
    b_str = str(b).lower().strip()

    # Exact match
    if a_str == b_str:
        return 1.0

    # Substring match
    if a_str in b_str or b_str in a_str:
        # Масштабируем по длине: чем больше разница, тем меньше score
        ratio = min(len(a_str), len(b_str)) / max(len(a_str), len(b_str))
        return 0.5 + ratio * 0.3  # 0.5..0.8

    # Token-based Jaccard с удалением цифр
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

    # Jaccard
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(intersection) / len(union) if union else 0.0

    # Prefix matching между токенами
    prefix_matches = 0
    for ta in tokens_a:
        for tb in tokens_b:
            if tb.startswith(ta) or ta.startswith(tb):
                ratio = min(len(ta), len(tb)) / max(len(ta), len(tb))
                prefix_matches += ratio

    max_pairs = len(tokens_a) * len(tokens_b)
    prefix_bonus = prefix_matches / max_pairs if max_pairs > 0 else 0.0

    final = max(jaccard, prefix_bonus * 0.8)
    return min(final, 1.0)


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

    @staticmethod
    def _relax_pattern(pattern: str) -> str:
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

        # 3. (?:( → (?:\s*\(
        relaxed = re.sub(r'\(\?\:\(', lambda m: r'(?:\s*\(', relaxed)

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

        # 7. Шайба: поддержка толщины как опционального параметра
        #    Толщина есть в тексте (0,5-4-8), но не в ЕНС — делаем опциональной
        _thick_old = r'(?P<наружный_диаметр_диаметр_вписа>\d+)\-?(?P<покрытие>[\w.]+)?'
        _thick_new = r'(?P<наружный_диаметр_диаметр_вписа>\d+)(?:\-(?P<толщина>\d+(?:[.,]\d+)?))?\-(?P<покрытие>[\w.]+)'
        if _thick_old in relaxed:
            relaxed = relaxed.replace(_thick_old, _thick_new, 1)
        # Альтернатива: уже с \d+(?:[.,]\d+)? (после rule 4)
        _thick_old2 = r'(?P<наружный_диаметр_диаметр_вписа>\d+(?:[.,]\d+)?)\-(?P<покрытие>[\w.]+)?'
        _thick_new2 = r'(?P<наружный_диаметр_диаметр_вписа>\d+(?:[.,]\d+)?)(?:\-(?P<толщина>\d+(?:[.,]\d+)?))?\-(?P<покрытие>[\w.]+)'
        if _thick_old2 in relaxed:
            relaxed = relaxed.replace(_thick_old2, _thick_new2, 1)

        # Проверяем что результат — валидный regex
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
        # Если pattern передан напрямую — используем его (skip БД)
        effective_mask = mask
        if pattern and not mask:
            # Создаём временный mask-like объект
            from types import SimpleNamespace
            effective_mask = SimpleNamespace(
                pattern=pattern, required=[], id=-1
            )

        if effective_mask:
            relaxed_pattern = self._relax_pattern(pattern or effective_mask.pattern)
            extracted_params = self._apply_mask(relaxed_pattern, text)

            if extracted_params:
                # Шаг 3: Ищем в ЕСН
                required = getattr(effective_mask, 'required', [])
                match_result = self._find_in_ens(extracted_params, required)

                if match_result:
                    return ParametricMatch(
                        ens_code=match_result.get('код'),
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

    def _apply_mask(self, pattern: str, text: str) -> Optional[Dict[str, Any]]:
        """Применение regex маски к тексту."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            match = compiled.search(text)

            if match:
                return match.groupdict()
        except re.error as e:
            # Логируем маску для диагностики — показываем первые 200 символов
            logger.error(f"Invalid mask pattern: {e}. Pattern (first 200 chars): {pattern[:200]!r}")

        return None

    def _find_in_ens(self, params: Dict[str, Any], required: List[str]) -> Optional[Dict[str, Any]]:
        """Поиск по параметрам в индексе ЕСН."""
        if not self._ens_index or 'items' not in self._ens_index:
            return None

        items = self._ens_index['items']
        best_match = None
        best_score = 0.0

        for item in items:
            score = self._calculate_match_score(params, required, item)

            if score > best_score:
                best_score = score
                best_match = item
                best_match['_match_score'] = score
                best_match['_match_type'] = 'exact' if score > 0.9 else 'partial'

        return best_match

    def _calculate_match_score(
        self,
        params: Dict[str, Any],
        required: List[str],
        ens_item: Dict[str, Any]
    ) -> float:
        """Расчет score сопоставления с учетом эквивалентности пустых значений."""
        if not required:
            return 0.0

        matches = 0
        weights = []

        for param in required:
            query_val_raw = params.get(param)
            ens_val_raw = ens_item.get(param)

            # Проверяем эквивалентность пустым значениям (БП ≡ None)
            query_is_empty = query_val_raw is None or _is_empty_equivalent(param, query_val_raw)
            ens_is_empty = ens_val_raw is None or _is_empty_equivalent(param, ens_val_raw)

            if query_is_empty and ens_is_empty:
                # Оба пустые (или эквивалентны пустым) — полный match
                matches += 1
                weights.append(1.0)
            elif query_is_empty or ens_is_empty:
                # Одно пустое, другое заполнено — partial match
                # (поле опционально, но значения различаются)
                weights.append(0.5)
            else:
                # Оба заполнены — сравниваем значения
                query_val = str(query_val_raw).lower().strip()
                ens_val = str(ens_val_raw).lower().strip()

                if query_val == ens_val:
                    matches += 1
                    weights.append(1.0)
                elif query_val in ens_val or ens_val in query_val:
                    matches += 0.5
                    weights.append(0.5)
                else:
                    weights.append(0.0)

                # Fuzzy matching fallback для текстовых полей
                if weights[-1] == 0.0:
                    TEXT_FIELDS = {'покрытие', 'материал', 'марка_материала', 'марка_стали'}
                    if param in TEXT_FIELDS:
                        sim = _text_similarity(query_val, ens_val)
                        if sim >= 0.5:
                            weights[-1] = sim
                            matches += sim

        if not weights:
            return 0.0

        return sum(weights) / len(weights)

    def _calculate_confidence(self, params: Dict[str, Any], required: List[str]) -> float:
        """Расчет уверенности в извлечении."""
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