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
        item_type: Optional[str] = None
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

        # Шаг 2: Применяем маску
        if mask:
            extracted_params = self._apply_mask(mask.pattern, text)

            if extracted_params:
                # Шаг 3: Ищем в ЕСН
                match_result = self._find_in_ens(extracted_params, mask.required)

                if match_result:
                    return ParametricMatch(
                        ens_code=match_result.get('код'),
                        mdm_key=match_result.get('mdm_key'),
                        matched_params=extracted_params,
                        score=match_result.get('_match_score', 0.0),
                        match_type=match_result.get('_match_type', 'exact'),
                        confidence=self._calculate_confidence(extracted_params, mask.required),
                        details={'mask_id': mask.id, 'pattern': mask.pattern}
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
            logger.error(f"Invalid mask pattern: {e}")

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
        """Расчет score сопоставления."""
        if not required:
            return 0.0

        matches = 0
        weights = []

        for param in required:
            if param in params and params[param] is not None:
                # Нормализуем значения
                query_val = str(params[param]).lower().strip()
                ens_val = str(ens_item.get(param, '')).lower().strip()

                if query_val == ens_val:
                    matches += 1
                    weights.append(1.0)
                elif query_val in ens_val or ens_val in query_val:
                    matches += 0.5
                    weights.append(0.5)
                else:
                    weights.append(0.0)

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
