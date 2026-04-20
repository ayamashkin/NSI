"""
ENS Registry Module
Управление загруженными справочниками ЕСН.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

from ens.loader import ENSLoader, ENSCategory
from ens.indexer import HybridENSIndex

logger = logging.getLogger(__name__)


@dataclass
class ENSRegistryEntry:
    """Запись в реестре ЕСН."""
    name: str
    category: ENSCategory
    loader: ENSLoader
    index: HybridENSIndex
    file_path: str


class ENSRegistry:
    """Реестр загруженных справочников ЕСН."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._entries: Dict[str, ENSRegistryEntry] = {}
        return cls._instance

    def register(self, name: str, file_path: str, category: Optional[ENSCategory] = None) -> ENSRegistryEntry:
        """Регистрация справочника ЕСН."""
        logger.info(f"Registering ENS reference: {name} from {file_path}")

        # Загружаем
        loader = ENSLoader(file_path, category=category)
        items = loader.load()

        # Строим индекс
        index = HybridENSIndex(items)

        entry = ENSRegistryEntry(
            name=name,
            category=loader.category,
            loader=loader,
            index=index,
            file_path=file_path
        )

        self._entries[name] = entry
        return entry

    def get(self, name: str) -> Optional[ENSRegistryEntry]:
        """Получение записи по имени."""
        return self._entries.get(name)

    def find_for_nomenclature(self, text: str) -> Optional[ENSRegistryEntry]:
        """Поиск подходящего справочника по тексту номенклатуры."""
        text_lower = text.lower()

        # Определяем категорию по ключевым словам
        category_scores = {}

        for name, entry in self._entries.items():
            score = 0

            # Проверяем ключевые слова категории
            if entry.category == ENSCategory.HARDWARE:
                keywords = ['болт', 'винт', 'гайка', 'шуруп', 'шпилька', 'заклепка']
            elif entry.category == ENSCategory.WASHER:
                keywords = ['шайба']
            elif entry.category == ENSCategory.ROLLED_METAL:
                keywords = ['лист', 'труба', 'уголок', 'швеллер', 'балка', 'прокат']
            else:
                keywords = []

            for kw in keywords:
                if kw in text_lower:
                    score += 1

            category_scores[name] = score

        # Возвращаем лучший match
        if category_scores:
            best = max(category_scores, key=category_scores.get)
            if category_scores[best] > 0:
                return self._entries[best]

        # По умолчанию первый
        return next(iter(self._entries.values())) if self._entries else None

    def list_entries(self) -> List[str]:
        """Список зарегистрированных справочников."""
        return list(self._entries.keys())


# Глобальный singleton
_registry = None

def get_ens_registry() -> ENSRegistry:
    """Получение глобального реестра."""
    global _registry
    if _registry is None:
        _registry = ENSRegistry()
    return _registry
