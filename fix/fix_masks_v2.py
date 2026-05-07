"""
Script to fix masks for known problematic standards.
Run: python fix_masks_v2.py
"""

import sqlite3
import logging
import hashlib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_path():
    """Find database file."""
    for p in ['nomenclature.db', 'database/nomenclature.db', '../database/nomenclature.db']:
        if Path(p).exists():
            return p
    # Search recursively
    for p in Path('.').rglob('nomenclature.db'):
        return str(p)
    return None


def fix_gost_7795(cursor):
    """Fix ГОСТ 7795-70 mask: correct length/group parsing."""
    standard = 'ГОСТ 7795-70'
    item_type = 'БОЛТ'

    # Correct pattern: Болт [исполнение]M[диаметр]x[шаг]-[класс]x[длина].[группа]
    correct_pattern = (
        r'^Болт\s*(?P<исполнение>\d+)?\s*'
        r'M(?P<номинальный_диаметр_резьбы>\d+)'
        r'(?:[xX\u00d7](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?\s*[-\s]*'
        r'(?P<класс_поле_допуска>\d+[a-zA-Z])\s*[xX\u00d7]'
        r'(?P<длина>\d+)\.(?P<группа_класс_прочности>\d+(?:\.\d+)?)\s*'
        r'ГОСТ\s*7795-70$'
    )

    params = [
        'исполнение', 'номинальный_диаметр_резьбы', 'шаг_резьбы',
        'класс_поле_допуска', 'длина', 'группа_класс_прочности'
    ]
    required = ['номинальный_диаметр_резьбы', 'длина', 'группа_класс_прочности']

    pattern_hash = hashlib.md5(correct_pattern.encode()).hexdigest()

    # Check existing
    cursor.execute(
        "SELECT id FROM masks WHERE standard = ? AND item_type = ?",
        (standard, item_type)
    )
    existing = cursor.fetchone()

    if existing:
        cursor.execute(
            """UPDATE masks 
               SET pattern = ?, params = ?, required = ?, 
                   pattern_hash = ?, is_active = 1, auto_score = 0.95
               WHERE id = ?""",
            (correct_pattern, ','.join(params), ','.join(required),
             pattern_hash, existing[0])
        )
        logger.info(f"Updated ГОСТ 7795-70 mask (id={existing[0]})")
    else:
        cursor.execute(
            """INSERT INTO masks 
               (standard, item_type, pattern, params, required, 
                pattern_hash, is_active, auto_score, source)
               VALUES (?, ?, ?, ?, ?, ?, 1, 0.95, 'manual_fix')""",
            (standard, item_type, correct_pattern,
             ','.join(params), ','.join(required), pattern_hash)
        )
        logger.info(f"Inserted ГОСТ 7795-70 mask")


def add_washer_masks(cursor):
    """Add masks for Шайба standards."""

    washer_masks = [
        # ОСТ 1 34505-80
        {
            'standard': 'ОСТ 1 34505-80',
            'item_type': 'ШАЙБА',
            'pattern': (
                r'^Шайба\s*'
                r'(?P<диаметр>\d+(?:[.,]\d+)?)\s*[-\s]+\s*'
                r'(?P<наружный_диаметр>\d+(?:[.,]\d+)?)\s*[-\s]+\s*'
                r'(?P<толщина>\d+(?:[.,]\d+)?)\s*[-\s]+\s*'
                r'(?P<покрытие>[\w.]+)\s*[-\s]*\s*'
                r'ОСТ\s*1\s*34505-80$'
            ),
            'params': ['диаметр', 'наружный_диаметр', 'толщина', 'покрытие'],
            'required': ['диаметр', 'наружный_диаметр', 'толщина', 'покрытие']
        },
        # ОСТ 1 34507-80
        {
            'standard': 'ОСТ 1 34507-80',
            'item_type': 'ШАЙБА',
            'pattern': (
                r'^Шайба\s*'
                r'(?P<диаметр>\d+(?:[.,]\d+)?)\s*[-\s]+\s*'
                r'(?P<наружный_диаметр>\d+(?:[.,]\d+)?)\s*[-\s]+\s*'
                r'(?P<толщина>\d+(?:[.,]\d+)?)\s*[-\s]+\s*'
                r'(?P<покрытие>[\w.]+)\s*[-\s]*\s*'
                r'ОСТ\s*1\s*34507-80$'
            ),
            'params': ['диаметр', 'наружный_диаметр', 'толщина', 'покрытие'],
            'required': ['диаметр', 'наружный_диаметр', 'толщина', 'покрытие']
        },
    ]

    for mask in washer_masks:
        pattern_hash = hashlib.md5(mask['pattern'].encode()).hexdigest()

        # Check existing
        cursor.execute(
            "SELECT id FROM masks WHERE standard = ? AND item_type = ?",
            (mask['standard'], mask['item_type'])
        )
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                """UPDATE masks 
                   SET pattern = ?, params = ?, required = ?, 
                       pattern_hash = ?, is_active = 1, auto_score = 0.95
                   WHERE id = ?""",
                (mask['pattern'], ','.join(mask['params']),
                 ','.join(mask['required']), pattern_hash, existing[0])
            )
            logger.info(f"Updated {mask['standard']} mask (id={existing[0]})")
        else:
            cursor.execute(
                """INSERT INTO masks 
                   (standard, item_type, pattern, params, required, 
                    pattern_hash, is_active, auto_score, source)
                   VALUES (?, ?, ?, ?, ?, ?, 1, 0.95, 'manual_fix')""",
                (mask['standard'], mask['item_type'], mask['pattern'],
                 ','.join(mask['params']), ','.join(mask['required']),
                 pattern_hash)
            )
            logger.info(f"Inserted {mask['standard']} mask")


def fix_ost_31141(cursor):
    """Fix ОСТ 1 31141-80 mask if needed."""
    standard = 'ОСТ 1 31141-80'
    item_type = 'БОЛТ'

    # Check current mask
    cursor.execute(
        "SELECT id, pattern FROM masks WHERE standard = ? AND item_type = ?",
        (standard, item_type)
    )
    row = cursor.fetchone()

    if row:
        mask_id, pattern = row
        # Test if pattern has issues
        if '(?:' in pattern and pattern.count('(') != pattern.count(')'):
            logger.warning(f"Mask {standard} has unbalanced parentheses, fixing")

            correct_pattern = (
                r'^Болт\s*'
                r'(?P<номинальный_диаметр_резьбы>\d+(?:[.,]\d+)?)\s*[-\s]+\s*'
                r'(?P<длина>\d+(?:[.,]\d+)?)\s*[-\s]+\s*'
                r'(?P<покрытие>[\w.]+)\s*[-\s]*\s*'
                r'ОСТ\s*1\s*31141-80$'
            )

            pattern_hash = hashlib.md5(correct_pattern.encode()).hexdigest()
            params = ['номинальный_диаметр_резьбы', 'длина', 'покрытие']
            required = ['номинальный_диаметр_резьбы', 'длина', 'покрытие']

            cursor.execute(
                """UPDATE masks 
                   SET pattern = ?, params = ?, required = ?, 
                       pattern_hash = ?, is_active = 1
                   WHERE id = ?""",
                (correct_pattern, ','.join(params), ','.join(required),
                 pattern_hash, mask_id)
            )
            logger.info(f"Fixed {standard} mask")
        else:
            logger.info(f"{standard} mask looks OK")
    else:
        logger.warning(f"No mask found for {standard}")


def main():
    db_path = get_db_path()
    if not db_path:
        logger.error("Database not found!")
        return

    logger.info(f"Using database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        fix_gost_7795(cursor)
        add_washer_masks(cursor)
        fix_ost_31141(cursor)

        conn.commit()
        logger.info("All fixes applied successfully!")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error applying fixes: {e}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()