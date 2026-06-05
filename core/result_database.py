#!/usr/bin/env python3
# =============================================================================
# ФАЙЛ: core/result_database.py
# ПОСЛЕДНИЕ 5 ИЗМЕНЕНИЙ (МСК, UTC+3), от новых к старым:
# 2026-06-04 15:00:00 — FEAT: ResultDatabaseManager — SQLite хранение результатов.
#   upsert_result, get_result, search. Для batch processing и верификации.
# =============================================================================

"""
Result Database — SQLite хранение результатов обработки номенклатуры.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultDatabaseManager:
    """Менеджер SQLite БД результатов."""

    def __init__(self, db_path: str = 'cache/result.db'):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """Создать таблицы если не существуют."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    article TEXT,
                    item_type TEXT,
                    standard TEXT,
                    ens_code TEXT,
                    ens_name TEXT,
                    success BOOLEAN DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    params TEXT,
                    ens_params TEXT,
                    ens_params_mask TEXT,
                    match_type TEXT,
                    match_type_ru TEXT,
                    coating_substitution TEXT,
                    fuzzy_mismatched_params TEXT,
                    mask_id TEXT,
                    mask_pattern TEXT,
                    details TEXT,
                    processing_time_ms REAL,
                    verified BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, article)
                )
            """)
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON results(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_standard ON results(standard)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_item_type ON results(item_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ens_code ON results(ens_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_success ON results(success)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON results(confidence)")
            conn.commit()

    def _json_dumps(self, val: Any) -> Optional[str]:
        if val is None:
            return None
        try:
            return json.dumps(val, ensure_ascii=False, default=str)
        except Exception:
            return str(val)

    def _json_loads(self, val: str) -> Any:
        if not val:
            return None
        try:
            return json.loads(val)
        except Exception:
            return val

    def upsert_result(
        self,
        name: str,
        article: Optional[str] = None,
        item_type: Optional[str] = None,
        standard: Optional[str] = None,
        ens_code: Optional[str] = None,
        ens_name: Optional[str] = None,
        success: bool = False,
        confidence: float = 0.0,
        params: Optional[Dict] = None,
        ens_params: Optional[Dict] = None,
        ens_params_mask: Optional[Dict] = None,
        match_type: Optional[str] = None,
        match_type_ru: Optional[str] = None,
        coating_substitution: Optional[Dict] = None,
        fuzzy_mismatched_params: Optional[Dict] = None,
        mask_id: Optional[str] = None,
        mask_pattern: Optional[str] = None,
        details: Optional[Dict] = None,
        processing_time_ms: float = 0.0,
        verified: bool = False,
    ) -> Tuple[bool, str]:
        """Сохранить или обновить результат."""
        with sqlite3.connect(self.db_path) as conn:
            # Check existing
            cursor = conn.execute(
                "SELECT id, ens_code, success, confidence FROM results WHERE name = ? AND (article = ? OR (article IS NULL AND ? IS NULL))",
                (name, article, article)
            )
            existing = cursor.fetchone()

            now = datetime.now().isoformat()

            if existing:
                # Update
                conn.execute("""
                    UPDATE results SET
                        item_type = ?, standard = ?, ens_code = ?, ens_name = ?,
                        success = ?, confidence = ?, params = ?, ens_params = ?,
                        ens_params_mask = ?, match_type = ?, match_type_ru = ?,
                        coating_substitution = ?, fuzzy_mismatched_params = ?,
                        mask_id = ?, mask_pattern = ?, details = ?,
                        processing_time_ms = ?, verified = ?, updated_at = ?
                    WHERE id = ?
                """, (
                    item_type, standard, ens_code, ens_name,
                    success, confidence,
                    self._json_dumps(params),
                    self._json_dumps(ens_params),
                    self._json_dumps(ens_params_mask),
                    match_type, match_type_ru,
                    self._json_dumps(coating_substitution),
                    self._json_dumps(fuzzy_mismatched_params),
                    mask_id, mask_pattern,
                    self._json_dumps(details),
                    processing_time_ms,
                    verified,
                    now,
                    existing[0],
                ))
                conn.commit()

                changed = (existing[1] != ens_code or existing[2] != success or
                          abs(existing[3] - confidence) > 0.001)
                reason = "updated" if changed else "unchanged"
                logger.debug("[RESULT_DB] Updated id=%s changed=%s", existing[0], changed)
                return changed, reason
            else:
                # Insert
                conn.execute("""
                    INSERT INTO results (
                        name, article, item_type, standard, ens_code, ens_name,
                        success, confidence, params, ens_params, ens_params_mask,
                        match_type, match_type_ru, coating_substitution,
                        fuzzy_mismatched_params, mask_id, mask_pattern,
                        details, processing_time_ms, verified, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, article, item_type, standard, ens_code, ens_name,
                    success, confidence,
                    self._json_dumps(params),
                    self._json_dumps(ens_params),
                    self._json_dumps(ens_params_mask),
                    match_type, match_type_ru,
                    self._json_dumps(coating_substitution),
                    self._json_dumps(fuzzy_mismatched_params),
                    mask_id, mask_pattern,
                    self._json_dumps(details),
                    processing_time_ms,
                    verified,
                    now,
                ))
                conn.commit()
                logger.debug("[RESULT_DB] Inserted new: name=%s", name[:50])
                return True, "inserted"

    def get_result(self, name: str, article: Optional[str] = None) -> Optional[Dict]:
        """Получить результат по наименованию."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if article:
                row = conn.execute(
                    "SELECT * FROM results WHERE name = ? AND article = ?",
                    (name, article)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM results WHERE name = ?",
                    (name,)
                ).fetchone()

            if row:
                return self._row_to_dict(row)
            return None

    def search(
        self,
        query: Optional[str] = None,
        standard: Optional[str] = None,
        item_type: Optional[str] = None,
        confidence_min: Optional[float] = None,
        confidence_max: Optional[float] = None,
        success_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict]:
        """Поиск по фильтрам."""
        conditions = []
        params = []

        if query:
            conditions.append("name LIKE ?")
            params.append(f"%{query}%")
        if standard:
            conditions.append("standard = ?")
            params.append(standard)
        if item_type:
            conditions.append("item_type = ?")
            params.append(item_type)
        if confidence_min is not None:
            conditions.append("confidence >= ?")
            params.append(confidence_min)
        if confidence_max is not None:
            conditions.append("confidence <= ?")
            params.append(confidence_max)
        if success_only:
            conditions.append("success = 1")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM results WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        result = dict(row)
        # Parse JSON fields
        for field in ['params', 'ens_params', 'ens_params_mask', 'coating_substitution',
                      'fuzzy_mismatched_params', 'details']:
            if result.get(field):
                result[field] = self._json_loads(result[field])
        return result
