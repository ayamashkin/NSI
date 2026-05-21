# =============================================================================
# FILE: parsers/standard_extractor.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 COMMITS (UTC+3):
#   2026-05-21 08:23:07  51f335da  21.05.2026
#   2026-05-21 08:05:56  ee843b22  21.05.2026
#   2026-05-20 17:47:49  19e8ca02  20.05.2026
#   2026-05-20 17:39:23  b00c4b25  20.05.2026
#   2026-05-20 17:31:34  66c66c93  20.05.2026
# =============================================================================
"""
Standard Extractor Module
Level 0: Детерминированное извлечение стандарта из текста номенклатуры.

LAST_FIX: 2026-05-21 08:50 UTC+3 — normalized теперь использует canonicalize_standard,
чтобы ОСТ 1 всегда возвращался с пробелом (ОСТ 1 34505-80, не ОСТ1 34505-80).
"""

import re
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum

from utils.standard_utils import canonicalize_standard

logger = logging.getLogger(__name__)


class StandardType(str, Enum):
    """Типы стандартов."""
    GOST = "ГОСТ"
    OST = "ОСТ"
    TU = "ТУ"
    ISO = "ISO"
    DIN = "DIN"
    RAM = "РАМ"
    UNKNOWN = "UNKNOWN"


@dataclass
class StandardInfo:
    """Информация о стандарте."""
    standard_type: StandardType
    standard_number: str
    year: Optional[str] = None
    full_name: str = ""
    start_pos: int = 0
    end_pos: int = 0

    @property
    def normalized(self) -> str:
        """Нормализованное название стандарта (канонический вид)."""
        raw = self.full_name if self.full_name else f"{self.standard_type.value} {self.standard_number}"
        return canonicalize_standard(raw)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для JSON."""
        return {
            'standard_type': self.standard_type.value if self.standard_type else None,
            'standard_number': self.standard_number,
            'year': self.year,
            'full_name': self.full_name,
            'normalized': self.normalized,
        }


class StandardExtractor:
    """
    Извлечение стандарта из текста номенклатуры.
    """

    PATTERNS = {
        StandardType.GOST: [
            r'ГОСТ\s*(\d+)-(\d+)',
            r'ГОСТ\s*[РR]?\s*(\d+)-(\d+)',
            r'ГОСТ\s*ISO\s*(\d+)-(\d+)',
        ],
        StandardType.OST: [
            r'ОСТ\s*(\d+)\s*(\d+)-(\d+)',
            r'ОСТ(\d+)\s*(\d+)-(\d+)',
        ],
        StandardType.TU: [
            r'ТУ\s*(\d+-\d+-\d+)',
        ],
        StandardType.ISO: [
            r'ISO\s*(\d+):?(\d+)?',
        ],
        StandardType.DIN: [
            r'DIN\s*(\d+)',
            r'DIN\s*EN\s*ISO\s*(\d+):?(\d+)?',
        ],
        StandardType.RAM: [
            r'РАМ\.(\d+)\.(\d+)',
            r'РАМ\s+(\d+)\s+(\d+)',
        ]
    }

    TYPE_PATTERNS = {
        'болт': r'\b[Бб]олт\b',
        'винт': r'\b[Вв]инт\b',
        'гайка': r'\b[Гг]айка\b',
        'шайба': r'\b[Шш]айба\b',
        'шуруп': r'\b[Шш]уруп\b',
        'шпилька': r'\b[Шш]пилька\b',
        'заклепка': r'\b[Зз]аклепка\b',
        'штифт': r'\b[Шш]тифт\b',
        'хомут': r'\b[Хх]омут\b',
        'анкер': r'\b[Аа]нкер\b',
        'саморез': r'\b[Сс]аморез\b',
        'гвоздь': r'\b[Гг]воздь\b',
    }

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Компиляция regex паттернов."""
        self._compiled = {}
        for std_type, patterns in self.PATTERNS.items():
            self._compiled[std_type] = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._type_compiled = {
            item_type: re.compile(pattern, re.IGNORECASE)
            for item_type, pattern in self.TYPE_PATTERNS.items()
        }

    def extract(self, text: str) -> Optional[StandardInfo]:
        """Извлечение стандарта из текста."""
        if not text:
            return None
        text = text.strip().rstrip(',.;: ')
        for std_type, patterns in self._compiled.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    return self._parse_match(text, match, std_type)
        return None

    def _parse_match(self, text: str, match: re.Match, std_type: StandardType) -> StandardInfo:
        """Парсинг match в StandardInfo."""
        groups = match.groups()
        if std_type == StandardType.GOST:
            number = groups[0]
            year = groups[1] if len(groups) > 1 else None
        elif std_type == StandardType.OST:
            if len(groups) >= 3:
                number = f"{groups[0]} {groups[1]}"
                year = groups[2]
            else:
                number = groups[0]
                year = groups[1] if len(groups) > 1 else None
        elif std_type == StandardType.TU:
            number = groups[0]
            year = None
        elif std_type == StandardType.ISO:
            number = groups[0]
            year = groups[1] if len(groups) > 1 and groups[1] else None
        elif std_type == StandardType.DIN:
            number = groups[0]
            year = groups[1] if len(groups) > 1 and groups[1] else None
        elif std_type == StandardType.RAM:
            number = f"{groups[0]}.{groups[1]}" if len(groups) > 1 else groups[0]
            year = None
        else:
            number = groups[0]
            year = None

        full_name = match.group(0)
        return StandardInfo(
            standard_type=std_type,
            standard_number=number,
            year=year,
            full_name=full_name,
            start_pos=match.start(),
            end_pos=match.end()
        )

    def extract_type(self, text: str) -> Optional[str]:
        """Определение типа изделия из текста."""
        if not text:
            return None
        text_lower = text.lower()
        for item_type, pattern in self._type_compiled.items():
            if pattern.search(text_lower):
                return item_type
        return None

    def extract_all(self, text: str) -> Dict[str, Any]:
        """Извлечение всех данных (стандарт + тип)."""
        standard = self.extract(text)
        item_type = self.extract_type(text)
        return {
            'standard_info': standard,
            'item_type': item_type,
            'has_standard': standard is not None,
            'has_type': item_type is not None
        }

    def batch_extract(self, texts: List[str]) -> List[Optional[StandardInfo]]:
        """Пакетное извлечение стандартов."""
        return [self.extract(text) for text in texts]

    def get_stats(self, texts: List[str]) -> Dict[str, Any]:
        """Статистика по извлечению стандартов."""
        total = len(texts)
        found = 0
        by_type = {t.value: 0 for t in StandardType}
        for text in texts:
            info = self.extract(text)
            if info:
                found += 1
                by_type[info.standard_type.value] += 1
        return {
            'total': total,
            'found': found,
            'coverage': found / total if total > 0 else 0,
            'by_type': {k: v for k, v in by_type.items() if v > 0}
        }


_extractor_instance: Optional[StandardExtractor] = None


def get_standard_extractor() -> StandardExtractor:
    """Получение глобального экземпляра StandardExtractor."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = StandardExtractor()
    return _extractor_instance