"""
Модуль сопоставления покрытий: номенклатура ↔ ЕНС.

Загружает Excel-справочник и строит двунаправленный mapping:
- ГОСТ/ОСТ код покрытия → Покрытие в ЕНС
- Покрытие в ЕНС → нормализованная форма
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CoatingMapper:
    """Сопоставитель покрытий номенклатуры с покрытиями ЕНС."""

    def __init__(self, excel_path: Optional[str] = None):
        self._gost_to_ens: Dict[str, str] = {}   # ГОСТ-код → ЕНС-код
        self._ost_to_ens: Dict[str, str] = {}    # ОСТ-код → ЕНС-код
        self._ens_names: Set[str] = set()         # Все покрытия ЕНС
        self._reverse_ens: Dict[str, str] = {}    # ЕНС-код → ГОСТ/ОСТ код
        if excel_path:
            self.load(excel_path)

    def load(self, excel_path: str):
        """Загрузить справочник покрытий из Excel."""
        path = Path(excel_path)
        if not path.exists():
            logger.warning(f"[CoatingMapper] Файл не найден: {excel_path}")
            return

        try:
            import pandas as pd
            df = pd.read_excel(excel_path)
        except Exception as e:
            logger.warning(f"[CoatingMapper] Ошибка загрузки Excel: {e}")
            return

        # Определяем колонки
        cols = list(df.columns)
        logger.info(f"[CoatingMapper] Колонки: {cols}")

        # Ищем колонки по ключевым словам
        ens_col = self._find_column(cols, ['покрытие в енс', 'енс', 'ens'])
        gost_col = self._find_column(cols, ['гост', '9.306', '1759'])
        ost_col = self._find_column(cols, ['ост', '31101'])

        if not ens_col:
            # Fallback: первая колонка — ЕНС, третья — ГОСТ, четвёртая — ОСТ
            if len(cols) >= 1:
                ens_col = cols[0]
            if len(cols) >= 3:
                gost_col = cols[2]
            if len(cols) >= 4:
                ost_col = cols[3]

        logger.info(f"[CoatingMapper] ENS='{ens_col}', GOST='{gost_col}', OST='{ost_col}'")

        for _, row in df.iterrows():
            ens_val = self._clean(str(row.get(ens_col, '')))
            if not ens_val or ens_val.lower() in ('nan', 'none', 'пусто', ''):
                continue

            self._ens_names.add(ens_val)

            # ГОСТ → ЕНС
            if gost_col:
                gost_val = self._clean(str(row.get(gost_col, '')))
                if gost_val and gost_val.lower() not in ('nan', 'none', 'пусто', ''):
                    self._gost_to_ens[gost_val] = ens_val
                    # Дополнительно: если ЕНС-код ≠ ГОСТ-код — обратный mapping
                    if gost_val != ens_val:
                        self._reverse_ens[ens_val] = gost_val

            # ОСТ → ЕНС
            if ost_col:
                ost_val = self._clean(str(row.get(ost_col, '')))
                if ost_val and ost_val.lower() not in ('nan', 'none', 'пусто', ''):
                    self._ost_to_ens[ost_val] = ens_val
                    if ost_val != ens_val:
                        self._reverse_ens[ens_val] = ost_val

        logger.info(
            f"[CoatingMapper] Загружено: {len(self._gost_to_ens)} GOST mappings, "
            f"{len(self._ost_to_ens)} OST mappings, {len(self._ens_names)} ENS coatings"
        )

    def _find_column(self, columns: List[str], keywords: List[str]) -> Optional[str]:
        """Найти колонку по ключевым словам (case-insensitive)."""
        for col in columns:
            col_lower = str(col).lower()
            for kw in keywords:
                if kw.lower() in col_lower:
                    return col
        return None

    def _clean(self, val: str) -> str:
        """Очистка значения от пробелов и пунктуации."""
        return val.strip().rstrip('.')

    def normalize(self, coating: Optional[str]) -> Optional[str]:
        """
        Нормализовать покрытие из номенклатуры в покрытие ЕНС.

        Returns:
            Покрытие в формате ЕНС или None.
        """
        if not coating:
            return None

        coating = self._clean(coating)

        # 1. Точное совпадение с ЕНС
        if coating in self._ens_names:
            return coating

        # 2. Прямой mapping ГОСТ → ЕНС
        if coating in self._gost_to_ens:
            return self._gost_to_ens[coating]

        # 3. Прямой mapping ОСТ → ЕНС
        if coating in self._ost_to_ens:
            return self._ost_to_ens[coating]

        # 4. Token-based перестановка (Окс.Фос.ЭФП ↔ Фос.Окс.ЭФП)
        tokens = self._tokenize(coating)
        for ens in self._ens_names:
            ens_tokens = self._tokenize(ens)
            if tokens == ens_tokens:
                return ens

        # 5. Substring matching (Кд3.хр → Кд.хр)
        for ens in self._ens_names:
            if self._is_subvariant(coating, ens):
                return ens

        return None

    def _tokenize(self, text: str) -> frozenset:
        """Разбить покрытие на токены (без точек, регистра, цифр-префиксов)."""
        # Убираем цифры в начале токенов (Кд3 → Кд)
        text = re.sub(r'(?<=[a-zA-Zа-яА-Я])\d+', '', text)
        # Разбиваем по точкам, дефисам, пробелам
        tokens = re.split(r'[.\-\s]+', text.lower())
        return frozenset(t for t in tokens if t)

    def _is_subvariant(self, candidate: str, target: str) -> bool:
        """Проверить является ли candidate подвариантом target."""
        # Кд3.хр → Кд.хр (цифры — это класс толщины)
        candidate_norm = re.sub(r'(?<=[a-zA-Zа-яА-Я])\d+', '', candidate)
        target_norm = re.sub(r'(?<=[a-zA-Zа-яА-Я])\d+', '', target)
        return candidate_norm.lower() == target_norm.lower()

    def get_all_ens_names(self) -> Set[str]:
        """Все покрытия ЕНС."""
        return self._ens_names.copy()

    def is_loaded(self) -> bool:
        """Загружен ли справочник."""
        return bool(self._ens_names)


# Singleton для глобального доступа
_global_mapper: Optional[CoatingMapper] = None


def init_mapper(excel_path: str) -> CoatingMapper:
    """Инициализировать глобальный mapper."""
    global _global_mapper
    _global_mapper = CoatingMapper(excel_path)
    return _global_mapper


def get_mapper() -> Optional[CoatingMapper]:
    """Получить глобальный mapper (или None)."""
    return _global_mapper