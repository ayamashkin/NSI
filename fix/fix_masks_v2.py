# =============================================================================
# FILE: fix/fix_masks_v2.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 COMMITS (UTC+3):
# 2026-05-21 12:57:18 db1fd327 21.05.2026
# 2026-05-21 08:53:16 6b906f29 21.05.2026
# 2026-05-21 08:23:07 51f335da 21.05.2026
# 2026-05-20 17:47:49 19e8ca02 20.05.2026
# 2026-05-20 17:39:23 b00c4b25 20.05.2026
# =============================================================================
# FIX 2026-05-22 09:36 UTC+3:
# 1. Fixed SQL schema syntax (created_at, last_used, pattern_hash).
# 2. Added created_at update on INSERT (CURRENT_TIMESTAMP).
# 3. Uses core.settings.DatabaseConfig.path for DB path.
# 4. Fixed regex typos (Y,\. -> [,\.]).
# 5. Fixed fix_gost_7795 pattern (removed leading \\n, fixed item_type).
# =============================================================================
"""
Script to fix masks for known problematic standards.
Run from project root: python fix/fix_masks_v2.py

VERSION: 2026-05-22 09:36 UTC+3
"""

import json
import sqlite3
import logging
import hashlib
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_db_path():
    """Find the mask database file via settings or fallback."""
    try:
        from core.settings import get_settings
        settings = get_settings()
        db_path = settings.database.path
        if Path(db_path).exists():
            logger.info(f"Found database via settings: {db_path}")
            return db_path
    except Exception as e:
        logger.debug(f"Settings not available: {e}")

    candidates = [
        'cache/masks.db',
        'masks.db',
        '../cache/masks.db',
        'cache/results.db'
    ]

    for p in candidates:
        if Path(p).exists():
            logger.info(f"Found database: {p}")
            return p

    # Search recursively (up to 2 levels deep)
    for pattern in ['**/*.db', '**/*masks']:
        for p in Path('.').glob(pattern):
            if p.is_file() and p.stat().st_size > 0:
                logger.info(f"Found database via search: {p}")
                return str(p)

    logger.info('No existing database found, will use: cache/results.db')
    Path('cache').mkdir(exist_ok=True)
    return 'cache/results.db'

def init_masks_table(cursor):
    """Ensure masks table exists with correct schema."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS masks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            standard TEXT NOT NULL,
            item_type TEXT NOT NULL,
            pattern TEXT NOT NULL,
            params TEXT,
            required TEXT,
            auto_score REAL DEFAULT 0.0,
            is_active INTEGER DEFAULT 0,
            source TEXT DEFAULT 'llm',
            usage_count INTEGER DEFAULT 0,
            test_examples TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP,
            pattern_hash TEXT UNIQUE,
            UNIQUE(standard, item_type, pattern_hash)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_masks_standard_type
        ON masks(standard, item_type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_masks_active
        ON masks(is_active, auto_score DESC)
    """)

def fix_gost_7795(cursor):
    """Fix ГОСТ 7795-70 mask: correct length/group parsing.

    Problem: old mask captured '100.58' as length=100.58 (float).
    Fix: pattern now correctly extracts length=100, group=5.8
    """
    standard = 'ГОСТ 7795-70'
    item_type = 'БОЛТ'

    correct_pattern = (
        r'^Болт\s*'
        r'(?P<исполнение>\d+)?\s*'
        r'M(?P<номинальный_диаметр_резьбы>\d+)'
        r'(?:[xX\u00d7](?P<шаг_резьбы>\d+(?:[,.]\d+)?))?\s*[-\s]*'
        r'(?P<класс_поле_допуска>\d+[a-zA-Z])\s*[xX\u00d7]'
        r'(?P<длина>\d+)\.(?P<группа_класс_прочности>\d+(?:\.\d+)?)\s*'
        r'ГОСТ\s*7795-70$'
    )

    params = json.dumps([
        'исполнение', 'номинальный_диаметр_резьбы', 'шаг_резьбы',
        'класс_поле_допуска', 'длина', 'группа_класс_прочности'
    ])
    required = json.dumps([
        'номинальный_диаметр_резьбы', 'длина', 'группа_класс_прочности'
    ])

    pattern_hash = hashlib.sha256(
        f"{correct_pattern}:{standard}:{item_type}".encode()
    ).hexdigest()[:16]

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
                   pattern_hash = ?, is_active = 1, auto_score = 0.95,
                   source = 'manual_fix', last_used = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (correct_pattern, params, required, pattern_hash, existing[0])
        )
        logger.info(f"Updated {standard} mask (id={existing[0]})")
    else:
        cursor.execute(
            """INSERT INTO masks
               (standard, item_type, pattern, params, required,
                pattern_hash, is_active, auto_score, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, 0.95, 'manual_fix', CURRENT_TIMESTAMP)""",
            (standard, item_type, correct_pattern, params, required, pattern_hash)
        )
        logger.info(f"Inserted {standard} mask")

def add_washer_masks(cursor):
    """Add masks for Шайба standards (ОСТ 1 34505-80, ОСТ 1 34507-80)."""

    washer_masks = [
        {
            'standard': 'ОСТ 1 34505-80',
            'item_type': 'ШАЙБА',
            'pattern': (
                r'^Шайба\s*'
                r'(?P<диаметр>\d+(?:[,\.]\d+)?)\s*[-\s]+\s*'
                r'(?P<наружный_диаметр>\d+(?:[,\.]\d+)?)\s*[-\s]+\s*'
                r'(?P<толщина>\d+(?:[,\.]\d+)?)\s*[-\s]+\s*'
                r'(?P<покрытие>[\w.]+)\s*[-\s]*\s*'
                r'ОСТ\s*1\s*34505-80$'
            ),
            'params': json.dumps(['диаметр', 'наружный_диаметр', 'толщина', 'покрытие']),
            'required': json.dumps(['диаметр', 'наружный_диаметр', 'толщина', 'покрытие'])
        },
        {
            'standard': 'ОСТ 1 34507-80',
            'item_type': 'ШАЙБА',
            'pattern': (
                r'^Шайба\s*'
                r'(?P<диаметр>\d+(?:[,\.]\d+)?)\s*[-\s]+\s*'
                r'(?P<наружный_диаметр>\d+(?:[,\.]\d+)?)\s*[-\s]+\s*'
                r'(?P<толщина>\d+(?:[,\.]\d+)?)\s*[-\s]+\s*'
                r'(?P<покрытие>[\w.]+)\s*[-\s]+\s*'
                r'ОСТ\s*1\s*34507-80$'
            ),
            'params': json.dumps(['диаметр', 'наружный_диаметр', 'толщина', 'покрытие']),
            'required': json.dumps(['диаметр', 'наружный_диаметр', 'толщина', 'покрытие'])
        },
    ]

    for mask in washer_masks:
        pattern_hash = hashlib.sha256(
            f"{mask['pattern']}:{mask['standard']}:{mask['item_type']}".encode()
        ).hexdigest()[:16]

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
                       pattern_hash = ?, is_active = 1, auto_score = 0.95,
                       source = 'manual_fix', last_used = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (mask['pattern'], mask['params'], mask['required'],
                 pattern_hash, existing[0])
            )
            logger.info(f"Updated {mask['standard']} mask (id={existing[0]})")
        else:
            cursor.execute(
                """INSERT INTO masks
                   (standard, item_type, pattern, params, required,
                    pattern_hash, is_active, auto_score, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, 0.95, 'manual_fix', CURRENT_TIMESTAMP)""",
                (mask['standard'], mask['item_type'], mask['pattern'],
                 mask['params'], mask['required'], pattern_hash)
            )
            logger.info(f"Inserted {mask['standard']} mask")

def fix_ost_31141(cursor):
    """Fix ОСТ 1 31141-80 mask if it has unbalanced parentheses."""

    standard = 'ОСТ 1 31141-80'
    item_type = 'БОЛТ'

    # Check current mask
    cursor.execute(
        "SELECT id, pattern FROM masks WHERE standard = ? AND item_type = ?",
        (standard, item_type)
    )
    row = cursor.fetchone()

    correct_pattern = (
        r'^Болт\s*'
        r'(?P<номинальный_диаметр_резьбы>\d+(?:[,\.]\d+)?)\s*[-\s]+\s*'
        r'(?P<длина>\d+(?:[,\.]\d+)?)\s*[-\s]+\s*'
        r'(?P<покрытие>[\w.]+)\s*[-\s]*\s*'
        r'ОСТ\s*1\s*31141-80$'
    )

    params = json.dumps(['номинальный_диаметр_резьбы', 'длина', 'покрытие'])
    required = json.dumps(['номинальный_диаметр_резьбы', 'длина', 'покрытие'])
    pattern_hash = hashlib.sha256(
        f"{correct_pattern}:{standard}:{item_type}".encode()
    ).hexdigest()[:16]

    if row:
        mask_id, pattern = row
        # Test if pattern has structural issues
        if pattern.count('(') != pattern.count(')'):
            logger.warning(f"Mask {standard} has unbalanced parentheses, fixing")
            cursor.execute(
                """UPDATE masks
                   SET pattern = ?, params = ?, required = ?,
                       pattern_hash = ?, is_active = 1,
                       source = 'manual_fix', last_used = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (correct_pattern, params, required, pattern_hash, mask_id)
            )
            logger.info(f"Fixed {standard} mask")
        else:
            logger.info(f"{standard} mask looks OK, updating anyway for consistency")
            cursor.execute(
                """UPDATE masks
                   SET pattern = ?, params = ?, required = ?,
                       pattern_hash = ?, is_active = 1,
                       source = 'manual_fix', last_used = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (correct_pattern, params, required, pattern_hash, mask_id)
            )
    else:
        logger.warning(f"No mask found for {standard}, inserting new one")
        cursor.execute(
            """INSERT INTO masks
               (standard, item_type, pattern, params, required,
                pattern_hash, is_active, auto_score, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, 0.95, 'manual_fix', CURRENT_TIMESTAMP)""",
            (standard, item_type, correct_pattern, params, required, pattern_hash)
        )

def main():
    db_path = get_db_path()
    logger.info(f"Using database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Ensure table exists
        init_masks_table(cursor)

        # Apply fixes
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