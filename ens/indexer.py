"""
ENS Indexer Module
Адаптивное построение индексов для быстрого поиска похожих записей в ЕСН.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ENSIndex:
    """Адаптивный индекс для поиска похожих записей в справочнике ЕСН."""

    def __init__(self, items: List[Dict[str, Any]], category_field: str = '_ens_category',
                 text_fields: Optional[List[str]] = None):
        """
        Инициализация индекса.

        Args:
            items: Список нормализованных записей ЕСН
            category_field: Поле с категорией
            text_fields: Поля для индексации (если None - автоопределение)
        """
        self.items = items
        self.category_field = category_field
        self.text_fields = text_fields or self._detect_text_fields()
        self.vectorizer = None
        self.tfidf_matrix = None
        self._build_index()

    def _detect_text_fields(self) -> List[str]:
        """Автоопределение текстовых полей для индексации."""
        if not self.items:
            return ['полное_наименование', 'наименование']

        # Берём первый item для анализа
        sample = self.items[0]

        # Приоритетные поля для индексации
        priority_fields = [
            'полное_наименование', 'наименование', 'полное наименование',
            'тип', 'стандарт', 'нтд', 'марка', 'материал', 'покрытие'
        ]

        detected = []
        for field in priority_fields:
            if field in sample:
                detected.append(field)

        # Если ничего не нашли, берём все строковые поля без _
        if not detected:
            for key, value in sample.items():
                if not key.startswith('_') and isinstance(value, str):
                    detected.append(key)

        return detected[:10]  # Максимум 10 полей

    def _build_index(self):
        """Построение TF-IDF индекса."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Формируем тексты для индексации
        texts = []
        for item in self.items:
            parts = []

            for field in self.text_fields:
                value = item.get(field)
                if value and not isinstance(value, (dict, list)):
                    parts.append(str(value))

            # Также добавляем все _original_ поля если есть
            for key, value in item.items():
                if key.startswith('_original_') and value and not isinstance(value, (dict, list)):
                    parts.append(str(value))

            texts.append(' '.join(str(p) if p is not None else '' for p in parts))

        if not texts:
            logger.warning("No texts to index")
            return

        # TF-IDF с символьными n-граммами для устойчивости к опечаткам
        self.vectorizer = TfidfVectorizer(
            ngram_range=(2, 5),  # символьные биграммы до 5-грамм
            analyzer='char',
            lowercase=True,
            max_features=50000
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"Built index for {len(texts)} items, "
                   f"matrix shape: {self.tfidf_matrix.shape}, "
                   f"fields used: {self.text_fields}")

    def search(self, query: str, k: int = 5, min_score: float = 0.1) -> List[Dict]:
        """
        Поиск похожих записей.

        Args:
            query: Строка запроса
            k: Количество результатов
            min_score: Минимальный score схожести

        Returns:
            Список похожих записей с score
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            logger.warning("Index not built")
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        # Векторизуем запрос
        query_vec = self.vectorizer.transform([query])

        # Вычисляем схожесть
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Получаем top-k
        top_indices = similarities.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                item = dict(self.items[idx])
                item['_similarity_score'] = score
                results.append(item)

        return results

    def search_by_params(self, params: Dict[str, Any], k: int = 5) -> List[Dict]:
        """Поиск по параметрам."""
        # Формируем запрос из параметров
        parts = []

        priority_params = ['тип', 'диаметр', 'длина', 'стандарт', 'марка', 'нтд']
        for param in priority_params:
            if params.get(param):
                parts.append(str(params[param]))

        # Добавляем остальные параметры
        for key, value in params.items():
            if key not in priority_params and value and not key.startswith('_'):
                parts.append(str(value))

        query = ' '.join(parts)
        return self.search(query, k=k)

    def save(self, path: str):
        """Сохранение индекса на диск."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'items': self.items,
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'category_field': self.category_field,
            'text_fields': self.text_fields
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Saved index to {path}")

    @classmethod
    def load(cls, path: str) -> 'ENSIndex':
        """Загрузка индекса с диска."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = cls.__new__(cls)
        index.items = data['items']
        index.vectorizer = data['vectorizer']
        index.tfidf_matrix = data['tfidf_matrix']
        index.category_field = data['category_field']
        index.text_fields = data.get('text_fields', ['полное_наименование', 'наименование'])

        logger.info(f"Loaded index from {path} ({len(index.items)} items)")
        return index


class HybridENSIndex:
    """Гибридный индекс: точный + нечеткий поиск."""

    def __init__(self, items: List[Dict[str, Any]],
                 exact_fields: Optional[List[str]] = None):
        self.items = items
        self.exact_fields = exact_fields or ['код', 'mdm_key']
        self.fuzzy_index = ENSIndex(items)
        self._build_exact_index()

    def _build_exact_index(self):
        """Построение индекса для точного поиска."""
        self.exact_index = {}

        for item in self.items:
            # Индексируем по code
            code = item.get('код')
            if code:
                self.exact_index[f"code:{code}"] = item

            # Индексируем по MDM Key
            mdm = item.get('mdm_key')
            if mdm:
                self.exact_index[f"mdm:{mdm}"] = item

            # Индексируем по комбинации тип+стандарт (если есть)
            item_type = str(item.get('тип', '')).lower()
            standard = item.get('стандарт') or item.get('нтд', '')
            if item_type and standard:
                key = f"type_std:{item_type}:{standard}"
                if key not in self.exact_index:
                    self.exact_index[key] = []
                if isinstance(self.exact_index[key], list):
                    self.exact_index[key].append(item)
                else:
                    self.exact_index[key] = [self.exact_index[key], item]

    def search(self, query: str, params: Optional[Dict] = None, k: int = 5) -> List[Dict]:
        """Гибридный поиск: сначала точный, потом нечеткий."""
        results = []

        # 1. Пробуем точный поиск по коду
        if params and params.get('код'):
            exact = self.exact_index.get(f"code:{params['код']}")
            if exact:
                exact = dict(exact)
                exact['_match_type'] = 'exact_code'
                results.append(exact)

        # 2. Пробуем точный поиск по MDM
        if params and params.get('mdm_key'):
            exact = self.exact_index.get(f"mdm:{params['mdm_key']}")
            if exact:
                exact = dict(exact)
                exact['_match_type'] = 'exact_mdm'
                if not any(r.get('mdm_key') == exact.get('mdm_key') for r in results):
                    results.append(exact)

        # 3. Пробуем точный поиск по типу+стандарту
        if params and params.get('тип') and (params.get('стандарт') or params.get('нтд')):
            std = params.get('стандарт') or params.get('нтд')
            key = f"type_std:{params['тип'].lower()}:{std}"
            exact_list = self.exact_index.get(key, [])
            if not isinstance(exact_list, list):
                exact_list = [exact_list]
            for item in exact_list:
                item = dict(item)
                item['_match_type'] = 'exact_type_standard'
                if not any(r.get('код') == item.get('код') for r in results):
                    results.append(item)

        # 4. Если мало результатов — добавляем нечеткий поиск
        if len(results) < k:
            fuzzy = self.fuzzy_index.search(query, k=k*2)
            for item in fuzzy:
                if not any(r.get('код') == item.get('код') for r in results):
                    item['_match_type'] = 'fuzzy'
                    results.append(item)
                    if len(results) >= k:
                        break

        return results[:k]

    def save(self, path: str):
        """Сохранение гибридного индекса."""
        # Сохраняем только fuzzy_index, exact_index перестроим при загрузке
        self.fuzzy_index.save(path)

    @classmethod
    def load(cls, path: str) -> 'HybridENSIndex':
        """Загрузка гибридного индекса."""
        fuzzy = ENSIndex.load(path)
        # Создаём новый инстанс и восстанавливаем exact_index
        index = cls.__new__(cls)
        index.items = fuzzy.items
        index.fuzzy_index = fuzzy
        index._build_exact_index()
        return index