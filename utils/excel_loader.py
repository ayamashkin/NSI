"""
Excel Loader Module
Загрузка и валидация данных номенклатуры из Excel файлов.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NomenclatureItem:
    """Модель элемента номенклатуры."""
    article: str
    name: str
    guid: str

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            'article': self.article,
            'name': self.name,
            'guid': self.guid
        }


class ExcelLoader:
    """
    Загрузчик данных номенклатуры из Excel файлов.

    Features:
    - Автоматическое определение кодировки
    - Валидация обязательных колонок
    - Очистка данных (strip, удаление пустых строк)
    - Поддержка разных форматов (.xlsx, .xls)
    """

    # Ожидаемые колонки
    REQUIRED_COLUMNS = ['артикул', 'Краткое наименование', 'GUID']

    def __init__(self, file_path: str):
        """
        Инициализация загрузчика.

        Args:
            file_path: Путь к Excel файлу
        """
        self.file_path = Path(file_path)
        self.df: Optional[Any] = None

    def load(self) -> List[NomenclatureItem]:
        """
        Загрузка и валидация данных из Excel.

        Returns:
            Список элементов номенклатуры

        Raises:
            FileNotFoundError: Если файл не существует
            ValueError: Если отсутствуют обязательные колонки
            ImportError: Если pandas не установлен
        """
        # ЛЕНИВЫЙ ИМПОРТ pandas
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas не установлен. Установите: pip install pandas openpyxl"
            )

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        logger.info(f"Loading Excel file: {self.file_path}")

        # Определение движка по расширению
        engine = 'openpyxl' if self.file_path.suffix == '.xlsx' else 'xlrd'

        try:
            self.df = pd.read_excel(self.file_path, engine=engine)
        except Exception as e:
            logger.error(f"Failed to read Excel: {e}")
            raise

        # Валидация колонок
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Очистка данных
        items = self._parse_dataframe()

        logger.info(f"Loaded {len(items)} items from Excel")
        return items

    def _parse_dataframe(self) -> List[NomenclatureItem]:
        """Парсинг DataFrame в список объектов."""
        # ЛЕНИВЫЙ ИМПОРТ pandas
        import pandas as pd

        items = []

        for idx, row in self.df.iterrows():
            try:
                # Получение значений с очисткой
                article = str(row['артикул']).strip() if pd.notna(row['артикул']) else ''
                name = str(row['Краткое наименование']).strip() if pd.notna(row['Краткое наименование']) else ''
                guid = str(row['GUID']).strip() if pd.notna(row['GUID']) else ''

                # Пропуск пустых строк
                if not article or not name:
                    logger.warning(f"Skipping row {idx}: empty article or name")
                    continue

                items.append(NomenclatureItem(
                    article=article,
                    name=name,
                    guid=guid
                ))

            except Exception as e:
                logger.error(f"Error parsing row {idx}: {e}")
                continue

        return items

    def get_preview(self, n: int = 5) -> Any:
        """
        Получение предпросмотра данных.

        Args:
            n: Количество строк для отображения

        Returns:
            DataFrame с первыми n строками
        """
        if self.df is None:
            self.load()
        return self.df.head(n)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по файлу.

        Returns:
            Словарь со статистикой
        """
        if self.df is None:
            self.load()

        return {
            'total_rows': len(self.df),
            'unique_articles': self.df['артикул'].nunique(),
            'empty_names': self.df['Краткое наименование'].isna().sum(),
            'file_size': self.file_path.stat().st_size
        }

    def validate_unique_articles(self) -> Tuple[bool, List[str]]:
        """
        Проверка уникальности артикулов.

        Returns:
            Кортеж (is_valid, duplicates)
        """
        if self.df is None:
            self.load()

        duplicates = self.df[self.df.duplicated(subset=['артикул'], keep=False)]

        if len(duplicates) > 0:
            return False, duplicates['артикул'].tolist()
        return True, []


def load_nomenclature(file_path: str) -> List[Dict[str, Any]]:
    """
    Упрощенная функция загрузки номенклатуры.

    Args:
        file_path: Путь к Excel файлу

    Returns:
        Список словарей с данными
    """
    loader = ExcelLoader(file_path)
    items = loader.load()
    return [item.to_dict() for item in items]
