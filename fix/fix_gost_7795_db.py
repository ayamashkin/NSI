"""
Direct fix for GOST 7795-70 mask in SQLite DB.
Dot = parameter separator, not decimal point.

Run: python fix_gost_7795_db.py [path/to/masks.db]
"""

import sys
import sqlite3
from pathlib import Path

DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cache/masks.db")

# Correct pattern: длина=\d+ then dot then группа=\d\d (e.g. 100.58 → длина=100, группа=5.8)
PATTERN = r'^Болт\s*(?P<исполнение>\d+)?\s*M\s*(?P<номинальный_диаметр_резьбы>\d+)\s*[-\.\s]*\s*(?P<класс_поле_допуска>\w+)\s*[-\.\s]*\s*(?P<длина>\d+)\.(?P<группа_класс_прочности>\d\d)(?:\s*[-\.\s]*\s*(?P<марка_материала>\d{2,3})?\s*[-\.\s]*\s*(?P<покрытие>\d{2})?\s*[-\.\s]*\s*(?P<толщина_покрытия>\d{1,2})?)?\s*[-\.\s]*\s*ГОСТ\s*7795-70$'
PARAMS = '["тип_изделия", "тип_резьбы", "номинальный_диаметр_резьбы", "длина", "класс_поле_допуска", "группа_класс_прочности", "марка_материала", "покрытие", "толщина_покрытия", "исполнение"]'
REQUIRED = '["тип_изделия", "номинальный_диаметр_резьбы", "длина", "класс_поле_допуска", "группа_класс_прочности"]'

def fix():
    if not DB_PATH.exists():
        print(f"[fix] DB not found: {DB_PATH}")
        return
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT id, pattern FROM masks WHERE standard = 'ГОСТ 7795-70' AND item_type = 'БОЛТ'")
    row = cur.fetchone()
    if not row:
        print("[fix] No mask found")
        conn.close()
        return
    mask_id, old = row
    print(f"[fix] Mask ID={mask_id}, old длина pattern: {old[old.find("длина"):old.find("длина")+30]}")
    cur.execute("UPDATE masks SET pattern = ?, params = ?, required = ?, last_used = datetime(\'now\') WHERE id = ?", (PATTERN, PARAMS, REQUIRED, mask_id))
    conn.commit()
    conn.close()
    print("[fix] Updated: длина=\d+, группа=\d\d (100.58 → длина=100, группа=5.8)")

if __name__ == "__main__":
    fix()
