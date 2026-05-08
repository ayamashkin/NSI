#!/usr/bin/env python3
"""
Автоматическая генерация ens_column_mapping.yaml из Excel-файла ЕНС.
Конвертирует все колонки в snake_case, исключает служебные.

Usage:
    python auto_mapping.py "data/Крепежные изделия_1.xlsx" -o config/ens_column_mapping.yaml
    python auto_mapping.py "data/Крепежные изделия_1.xlsx" --append -o config/ens_column_mapping.yaml

LAST_FIX: 2026-05-08 11:30 UTC+3
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set

try:
    import pandas as pd
    import yaml
except ImportError:
    print("ERROR: pip install pandas pyyaml")
    sys.exit(1)


# Служебные колонки которые НЕ маппятся (regex patterns)
EXCLUDE_PATTERNS = [
    r"^автор$",
    r"^дата\s*(создания|изменения|проверки)$",
    r"^mdm\s*key$",
    r"^заблокировано$",
    r"^пометка\s*удаления$",
    r"^окпд2?$",
    r"^оквэд2?$",
    r"^ссылка$",
    r"^id$",
    r"^код$",           # базовый — маппится отдельно
    r"^наименование$",  # базовый — маппится отдельно
    r"^полное\s*наименование$",
    r"^тип\s*позиции$",
    r"^нтд$",
]

# Уже замапленные базовые колонки (не трогаем)
BASE_COLUMNS = {
    "Код": "код",
    "Наименование": "наименование",
    "Полное наименование": "полное_наименование",
    "Тип позиции": "тип",
    "НТД": "нтд",
}


def to_snake_case(name: str) -> str:
    """Конвертирует название колонки в snake_case."""
    # Заменяем скобки и спецсимволы
    name = re.sub(r'[(),/]', ' ', name)
    # Заменяем точки на подчеркивание
    name = name.replace('.', '_')
    # Убираем лишние пробелы
    name = ' '.join(name.split())
    # Транслитерация кириллицы
    name = transliterate(name)
    # В snake_case
    name = re.sub(r'\s+', '_', name.strip())
    name = name.lower()
    # Убираем множественные подчеркивания
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def transliterate(text: str) -> str:
    """Простая транслитерация русских букв."""
    mapping = {
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
    return ''.join(mapping.get(c, c) for c in text)


def is_excluded(col_name: str) -> bool:
    """Проверяет, является ли колонка служебной."""
    col_lower = col_name.lower().strip()
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, col_lower, re.IGNORECASE):
            return True
    return False


def generate_mapping(excel_path: str, append: bool = False, existing_yaml: str = None) -> Dict:
    """Генерирует маппинг колонок из Excel файла."""
    df = pd.read_excel(excel_path)
    all_columns = list(df.columns)

    print(f"Всего колонок в Excel: {len(all_columns)}")

    # Загружаем существующий маппинг если нужно
    existing_category = {}
    existing_base = dict(BASE_COLUMNS)

    if append and existing_yaml and Path(existing_yaml).exists():
        with open(existing_yaml, 'r', encoding='utf-8') as f:
            existing = yaml.safe_load(f) or {}
        existing_base = existing.get('base_mapping', {})
        # Берем первую категорию
        cats = existing.get('category_mapping', {})
        if cats:
            existing_category = list(cats.values())[0]
        print(f"Загружен существующий маппинг: {len(existing_base)} base + {len(existing_category)} category")

    # Формируем новый маппинг
    new_mappings = {}
    skipped = []

    for col in all_columns:
        # Пропускаем базовые колонки (уже замаплены)
        if col in existing_base:
            continue
        # Пропускаем служебные
        if is_excluded(col):
            skipped.append(col)
            continue
        # Пропускаем уже замапленные в категории
        if col in existing_category:
            continue

        normalized = to_snake_case(col)
        # Обрезаем слишком длинные
        if len(normalized) > 80:
            normalized = normalized[:80]

        new_mappings[col] = normalized

    print(f"Новых маппингов: {len(new_mappings)}")
    print(f"Пропущено (служебные): {len(skipped)}")

    # Формируем результат
    result = {
        'base_mapping': existing_base,
        'category_mapping': {
            'hardware': {**existing_category, **new_mappings}
        }
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Автогенерация ens_column_mapping.yaml')
    parser.add_argument('excel', help='Путь к Excel файлу ЕНС')
    parser.add_argument('-o', '--output', required=True, help='Путь для сохранения YAML')
    parser.add_argument('--append', action='store_true', help='Дополнить существующий YAML')
    args = parser.parse_args()

    mapping = generate_mapping(
        args.excel,
        append=args.append,
        existing_yaml=args.output if args.append else None
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(mapping, f, allow_unicode=True, sort_keys=False)

    print(f"\nСохранено: {args.output}")


if __name__ == '__main__':
    main()