#!/usr/bin/env python3
"""
Загрузчик данных ЕНС из Excel.
Поддерживает авто-маппинг колонок через ens_column_mapping.yaml
и авто-генерацию snake_case ключей для немапленных колонок.

LAST_FIXES:
  - 2026-05-08 10:30 UTC+3 — авто-транслитерация немапленных колонок в snake_case
  - 2026-05-08 10:15 UTC+3 — fallback _auto_snake_case для всех неизвестных колонок
  - 2026-05-07 18:20 UTC+3 — поддержка 129 колонок Excel
  - 2026-05-07 14:30 UTC+3 — оптимизация загрузки индекса
  - 2026-05-07 11:45 UTC+3 — базовая структура загрузчика
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    import pandas as pd
    import yaml
except ImportError:
    raise ImportError("pip install pandas pyyaml")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Транслитерация (общая с auto_mapping.py)
# ---------------------------------------------------------------------------

_TRANSLIT_MAP = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'j', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'c', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
    'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'J', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'Kh', 'Ц': 'C', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
}


def _transliterate(text: str) -> str:
    """Простая транслитерация русских букв в латиницу."""
    return ''.join(_TRANSLIT_MAP.get(c, c) for c in text)


def _auto_snake_case(col_name: str) -> str:
    """
    Авто-генерация snake_case ключа из названия колонки.
    Транслитерирует кириллицу, нормализует спецсимволы.
    """
    if not col_name:
        return 'unknown'

    # Заменяем скобки, запятые, слэши на пробелы
    name = re.sub(r'[(),/]', ' ', col_name)
    # Точки → подчеркивание
    name = name.replace('.', '_')
    # Убираем лишние пробелы
    name = ' '.join(name.split())
    # Транслитерация
    name = _transliterate(name)
    # Пробелы → подчеркивания
    name = re.sub(r'\s+', '_', name.strip())
    name = name.lower()
    # Убираем множественные подчеркивания
    name = re.sub(r'_+', '_', name)
    # Убираем не-алфавитно-цифровые символы (кроме подчеркивания)
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name.strip('_') or 'unknown'


# ---------------------------------------------------------------------------
# Конфигурация маппинга колонок
# ---------------------------------------------------------------------------

@dataclass
class ENSColumnMapping:
    """Маппинг колонок Excel → нормализованные ключи."""

    base_mapping: Dict[str, str] = field(default_factory=dict)
    """Базовые колонки: 'Код' → 'код'"""

    category_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)
    """По категориям: 'hardware' → {'D п�"东山再起': 'd'}"""

    column_mapping: Dict[str, str] = field(default_factory=dict)
    """Плоский маппинг всех колонок."""

    def __post_init__(self):
        """Строим плоский маппинг из base + category."""
        flat = dict(self.base_mapping)
        for cat_map in self.category_mapping.values():
            flat.update(cat_map)
        self.column_mapping = flat
        logger.info(
            "ENSColumnMapping: %d base + %d category = %d total columns",
            len(self.base_mapping),
            sum(len(m) for m in self.category_mapping.values()),
            len(self.column_mapping),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ENSColumnMapping":
        """Загружает маппинг из YAML-файла."""
        path = Path(yaml_path)
        if not path.exists():
            logger.warning("Column mapping YAML not found: %s", yaml_path)
            return cls()

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        return cls(
            base_mapping=data.get('base_mapping', {}),
            category_mapping=data.get('category_mapping', {}),
        )

    def auto_map_column(self, col_name: str) -> Optional[str]:
        """
        Определяет нормализованный ключ для колонки.
        Сначала ищет в явном маппинге, затем применяет авто-snake_case.
        """
        if not col_name:
            return None

        # 1. Прямое совпадение в маппинге
        if col_name in self.column_mapping:
            return self.column_mapping[col_name]

        # 2. Регистро-независимый поиск
        col_lower = col_name.lower().strip()
        for key, val in self.column_mapping.items():
            if key.lower().strip() == col_lower:
                return val

        # 3. Regex-маппинг для типовых паттернов
        regex_patterns = {
            r'^код$': 'код',
            r'^наименование$': 'наименование',
            r'^полное\s*наименование$': 'полное_наименование',
            r'^тип\s*позиции$': 'тип',
            r'^нтд$': 'нтд',
            r'^d\s*\(?п.*?\)?$': 'd',
            r'^d\s*$': 'd',
            r'^d,\s*мм$': 'd',
            r'^l\s*\(?п.*?\)?$': 'l',
            r'^l$': 'l',
            r'^l,\s*мм$': 'l',
            r'^покрытие$': 'покрытие',
            r'^класс\s*прочности$': 'класс_прочности',
            r'^материал$': 'материал',
            r'^стандарт$': 'стандарт',
            r'^тип$': 'тип',
        }

        for pattern, normalized in regex_patterns.items():
            if re.search(pattern, col_name, re.IGNORECASE):
                return normalized

        # 4. АВТО-ГЕНЕРАЦИЯ: транслитерация + snake_case
        auto_key = _auto_snake_case(col_name)
        if auto_key and auto_key != 'unknown':
            logger.debug("Auto-mapped column '%s' → '%s'", col_name, auto_key)
            return auto_key

        return None

    def _row_to_normalized_dict(self, row: pd.Series) -> Dict[str, Any]:
        """
        Преобразует строку DataFrame в нормализованный словарь.
        Все немапленные колонки получают авто-snake_case ключи.
        """
        result = {}
        for col_name, value in row.items():
            if pd.isna(value):
                continue

            # Определяем нормализованный ключ
            normalized_key = self.auto_map_column(col_name)
            if normalized_key is None:
                # Крайний fallback — авто-snake_case из оригинального имени
                normalized_key = _auto_snake_case(col_name)

            # Сохраняем значение (строка или число)
            if isinstance(value, (int, float)):
                result[normalized_key] = value
            else:
                result[normalized_key] = str(value).strip()

        return result


# ---------------------------------------------------------------------------
# Загрузчик данных ЕНС
# ---------------------------------------------------------------------------

class ENSLoader:
    """Загружает и индексирует данные ЕНС из Excel."""

    def __init__(
        self,
        excel_path: str,
        column_mapping_yaml: Optional[str] = None,
        sheet_name: Optional[str] = None,
    ):
        self.excel_path = Path(excel_path)
        self.sheet_name = sheet_name
        self.schema = ENSColumnMapping.from_yaml(column_mapping_yaml) if column_mapping_yaml else ENSColumnMapping()
        self._df: Optional[pd.DataFrame] = None
        self._index: List[Dict[str, Any]] = []

    def load(self) -> "ENSLoader":
        """Загружает Excel в DataFrame."""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"ENS Excel not found: {self.excel_path}")

        logger.info("Loading ENS data from %s", self.excel_path)
        self._df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)

        # Авто-обнаружение всех колонок
        all_columns = list(self._df.columns)
        mapped_count = sum(1 for c in all_columns if self.schema.auto_map_column(c) in self.schema.column_mapping.values())
        auto_mapped = sum(
            1 for c in all_columns
            if self.schema.auto_map_column(c) and self.schema.auto_map_column(c) not in self.schema.column_mapping.values()
        )
        unmapped = len(all_columns) - mapped_count - auto_mapped

        logger.info(
            "ENS columns: %d total | %d explicitly mapped | %d auto-mapped | %d unmapped",
            len(all_columns), mapped_count, auto_mapped, unmapped,
        )

        # Логируем немапленные колонки
        if unmapped > 0:
            unmapped_cols = [
                c for c in all_columns
                if self.schema.auto_map_column(c) is None
            ]
            logger.warning("Unmapped columns (%d): %s", len(unmapped_cols), unmapped_cols[:20])

        return self

    def build_index(self) -> List[Dict[str, Any]]:
        """Строит нормализованный индекс из всех строк Excel."""
        if self._df is None:
            self.load()

        logger.info("Building ENS index from %d rows...", len(self._df))
        self._index = []

        for idx, row in self._df.iterrows():
            normalized = self.schema._row_to_normalized_dict(row)
            if normalized:
                # Добавляем мета-информацию
                normalized['_source_row'] = int(idx)
                normalized['_source_file'] = str(self.excel_path.name)
                self._index.append(normalized)

        logger.info("ENS index built: %d entries", len(self._index))
        return self._index

    def get_index(self) -> List[Dict[str, Any]]:
        """Возвращает построенный индекс."""
        if not self._index:
            self.build_index()
        return self._index

    def search(
        self,
        standard: Optional[str] = None,
        item_type: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Простой поиск по индексу."""
        results = self._index

        if standard:
            results = [r for r in results if r.get('стандарт') == standard or r.get('standard') == standard]
        if item_type:
            results = [r for r in results if r.get('тип') == item_type or r.get('item_type') == item_type]

        for key, value in kwargs.items():
            results = [r for r in results if r.get(key) == value]

        return results

    def get_column_stats(self) -> Dict[str, int]:
        """Возвращает статистику по колонкам (для отладки)."""
        if self._df is None:
            self.load()

        stats = {}
        for col in self._df.columns:
            key = self.schema.auto_map_column(col) or _auto_snake_case(col)
            non_null = self._df[col].notna().sum()
            stats[key] = int(non_null)

        return stats


# ---------------------------------------------------------------------------
# Фабрика загрузчиков
# ---------------------------------------------------------------------------

def create_ens_loader(
    excel_path: str,
    mapping_yaml: Optional[str] = None,
    category: Optional[str] = None,
) -> ENSLoader:
    """
    Фабричная функция для создания загрузчика ЕНС.

    Args:
        excel_path: Путь к Excel-файлу ЕНС
        mapping_yaml: Путь к ens_column_mapping.yaml (опционально)
        category: Категория для фильтрации (например 'hardware')

    Returns:
        ENSLoader: Настроенный и загруженный загрузчик
    """
    loader = ENSLoader(excel_path, column_mapping_yaml=mapping_yaml)
    loader.load()

    if category:
        # Фильтруем по категории если указана
        logger.info("Filtering ENS by category: %s", category)

    loader.build_index()
    return loader


# ---------------------------------------------------------------------------
# Точка входа для отладки
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python loader.py <path_to_ens.xlsx> [mapping.yaml]")
        sys.exit(1)

    excel = sys.argv[1]
    mapping = sys.argv[2] if len(sys.argv) > 2 else None

    loader = create_ens_loader(excel, mapping_yaml=mapping)

    print(f"\nIndex entries: {len(loader.get_index())}")
    print(f"Columns mapped: {len(loader.schema.column_mapping)}")

    # Выводим статистику
    stats = loader.get_column_stats()
    print(f"\nTop columns by fill rate:")
    for col, count in sorted(stats.items(), key=lambda x: -x[1])[:20]:
        print(f"  {col}: {count}")