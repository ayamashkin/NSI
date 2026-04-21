#!/usr/bin/env python3
"""
Seed Default Masks - Заполнение БД дефолтными масками
Выполните: python seed_default_masks.py cache/masks.db
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь (default/seed_default_masks.py -> корень проекта)
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mask_database import MaskDatabase, MaskRecord


def seed_default_masks(db_path="cache/masks.db"):
    """Создание масок для топ-20 стандартов с auto_score=1.0"""

    db = MaskDatabase(db_path=db_path)

    # Проверяем текущее состояние
    stats_before = db.get_statistics()
    print(f"📊 До заполнения: {stats_before.get('total', 0)} масок")

    # Дефолтные маски (из cascade.py)
    default_masks = [
        # ОСТ 1 31133-80 - Болты (ваш пример)
        {
            "standard": "ОСТ 1 31133-80",
            "item_type": "болт",
            "pattern": r'Болт\s*\((?P<исполнение>\d)\)-(?P<диаметр>\d+)-(?P<длина>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*31133-80)',
            "params": ["исполнение", "диаметр", "длина", "покрытие", "стандарт"],
            "required": ["исполнение", "диаметр", "длина", "покрытие"],
        },
        # ОСТ 1 31502-80 - Винты
        {
            "standard": "ОСТ 1 31502-80",
            "item_type": "винт",
            "pattern": r'Винт\s*\((?P<исполнение>\d)\)-(?P<диаметр>\d+)-(?P<длина>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*31502-80)',
            "params": ["исполнение", "диаметр", "длина", "покрытие", "стандарт"],
            "required": ["исполнение", "диаметр", "длина"],
        },
        # ОСТ 1 31503-80 - Винты
        {
            "standard": "ОСТ 1 31503-80",
            "item_type": "винт",
            "pattern": r'Винт\s*\((?P<исполнение>\d)\)-(?P<диаметр>\d+)-(?P<длина>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*31503-80)',
            "params": ["исполнение", "диаметр", "длина", "покрытие", "стандарт"],
            "required": ["исполнение", "диаметр", "длина"],
        },
        # ОСТ 1 34505-80 - Шайбы
        {
            "standard": "ОСТ 1 34505-80",
            "item_type": "шайба",
            "pattern": r'Шайба\s+(?P<толщина>[\d,]+)-(?P<диаметр_внутренний>\d+)-(?P<диаметр_наружный>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*34505-80)',
            "params": ["толщина", "диаметр_внутренний", "диаметр_наружный", "покрытие", "стандарт"],
            "required": ["толщина", "диаметр_внутренний", "диаметр_наружный"],
        },
        # ОСТ 1 34507-80 - Шайбы
        {
            "standard": "ОСТ 1 34507-80",
            "item_type": "шайба",
            "pattern": r'Шайба\s+(?P<толщина>[\d,]+)-(?P<диаметр_внутренний>\d+)-(?P<диаметр_наружный>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*34507-80)',
            "params": ["толщина", "диаметр_внутренний", "диаметр_наружный", "покрытие", "стандарт"],
            "required": ["толщина", "диаметр_внутренний", "диаметр_наружный"],
        },
        # ГОСТ 7795-70 - Болты с допуском
        {
            "standard": "ГОСТ 7795-70",
            "item_type": "болт",
            "pattern": r'Болт\s+(?P<исполнение>\d)?M(?P<диаметр>[\d,]+)(?:x(?P<шаг>[\d,]+))?-(?P<допуск>[\dgx]+)\.(?P<длина>[\d,]+)\.(?P<материал>\d+)?\s*(?P<стандарт>ГОСТ\s*7795-70)',
            "params": ["исполнение", "диаметр", "шаг", "допуск", "длина", "материал", "стандарт"],
            "required": ["диаметр", "длина"],
        },
        # ГОСТ 7798-70 - Болты базовые
        {
            "standard": "ГОСТ 7798-70",
            "item_type": "болт",
            "pattern": r'Болт\s+(?P<исполнение>\d)?M?(?P<диаметр>[\d,]+)(?:x(?P<шаг>[\d,]+))?-?(?P<длина>[\d,]+)\s*(?P<стандарт>ГОСТ\s*7798-70)',
            "params": ["исполнение", "диаметр", "шаг", "длина", "стандарт"],
            "required": ["диаметр", "длина"],
        },
    ]

    added = 0
    skipped = 0

    for mask_data in default_masks:
        mask = MaskRecord(
            standard=mask_data["standard"],
            item_type=mask_data["item_type"],
            pattern=mask_data["pattern"],
            params=mask_data["params"],
            required=mask_data["required"],
            auto_score=1.0,  # Идеальная маска
            is_active=True,
            source="default",
        )

        # Проверяем по pattern_hash
        existing = db.get_mask_by_pattern(mask.pattern, mask.standard)
        if not existing:
            mask_id = db.save_mask(mask, auto_activate=True)
            if mask_id:
                added += 1
                print(f"✅ Добавлена: {mask.standard} ({mask.item_type}) - ID {mask_id}")
            else:
                print(f"❌ Ошибка сохранения: {mask.standard}")
        else:
            skipped += 1
            print(f"⏭️  Уже есть: {mask.standard} (ID: {existing.id})")

    # Итоговая статистика
    stats_after = db.get_statistics()
    print(f"\n{'='*50}")
    print(f"📊 РЕЗУЛЬТАТ:")
    print(f"   Добавлено: {added}")
    print(f"   Пропущено (уже есть): {skipped}")
    print(f"   Всего в БД: {stats_after.get('total', 0)}")
    print(f"   Активных: {stats_after.get('active', 0)}")

    db.close()
    return added


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "cache/masks.db"
    print(f"🚀 Заполнение БД: {db_path}\n")
    seed_default_masks(db_path)