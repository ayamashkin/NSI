#!/usr/bin/env python3
"""
Result Database Manager
Хранение результатов обработки номенклатуры с отслеживанием версий масок.

VERSION: 2026-05-19

LAST_FIXES:
 2026-05-19 22:05 UTC+3 — Ключ кэша изменён с UNIQUE(article, name) на UNIQUE(name, standard).
   Теперь кэш работает по наименованию + стандарту, а не по артикулу.
"""

import sqlite3
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def _json_dumps(value: Any) -> Optional[str]:
    """Безопасная сериализация в JSON."""
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning("JSON serialization error: %s", e)
        return str(value)


def _json_loads(value: Optional[str]) -> Any:
    """Безопасная десериализация из JSON."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


class ResultDatabaseManager:
    """Менеджер для result.db — хранение результатов сопоставления номенклатуры."""

    def __init__(self, db_path: str = "result.db"):
        self.db_path = Path(db_path)
        self._init_db()
        self._cleanup_old_records()

    def _cleanup_old_records(self):
        """Очистка старых записей с NULL standard или некорректными name (с конечной запятой)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Удаляем записи с NULL standard (старые, до фикса)
                c1 = conn.execute("DELETE FROM nomenclature_results WHERE standard IS NULL").rowcount
                # Удаляем записи с name заканчивающимся на , ; : .
                c2 = conn.execute(
                    "DELETE FROM nomenclature_results WHERE name LIKE '%,' OR name LIKE '%;' OR name LIKE '%:' OR name LIKE '%.'"
                ).rowcount
                conn.commit()
                total = c1 + c2
                if total > 0:
                    logger.info("[RESULT_DB] Cleaned %d old invalid records (NULL standard or trailing punctuation)", total)
        except Exception as e:
            logger.debug("[RESULT_DB] Cleanup error: %s", e)

    def _init_db(self):
        """Инициализация таблиц result.db. Пересоздаёт таблицу если article имеет NOT NULL."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # Проверяем существующую схему
            cursor = conn.execute("PRAGMA table_info(nomenclature_results)")
            cols = {row[1]: row for row in cursor.fetchall()}

            # Если article существует и имеет NOT NULL — пересоздаём таблицу
            if 'article' in cols and cols['article'][3] == 1:  # notnull=1
                logger.warning("[RESULT_DB] Recreating table: article has NOT NULL constraint")
                conn.execute("ALTER TABLE nomenclature_results RENAME TO nomenclature_results_old")
                conn.execute("""
                    CREATE TABLE nomenclature_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article TEXT,
                        name TEXT NOT NULL,
                        standard TEXT,
                        item_type TEXT,
                        level TEXT,
                        success INTEGER NOT NULL DEFAULT 0,
                        params TEXT,
                        ens_code TEXT,
                        ens_name TEXT,
                        ens_params TEXT,
                        ens_params_mask TEXT,
                        confidence REAL,
                        match_type TEXT,
                        match_type_ru TEXT,
                        coating_substitution TEXT,
                        fuzzy_mismatched_params TEXT,
                        mask_id INTEGER,
                        mask_pattern TEXT,
                        mask_pattern_hash TEXT,
                        details TEXT,
                        processing_time_ms REAL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(name, standard)
                    )
                """)
                # Копируем данные (article -> пустая строка если NULL)
                conn.execute("""
                    INSERT INTO nomenclature_results 
                    SELECT id, COALESCE(article, ''), name, standard, item_type, level, success,
                           params, ens_code, ens_name, ens_params, ens_params_mask, confidence,
                           match_type, match_type_ru, coating_substitution, fuzzy_mismatched_params,
                           mask_id, mask_pattern, mask_pattern_hash, details, processing_time_ms,
                           created_at, updated_at
                    FROM nomenclature_results_old
                """)
                conn.execute("DROP TABLE nomenclature_results_old")
                conn.commit()
                logger.info("[RESULT_DB] Table recreated successfully")
                return

            # Обычное создание если таблицы нет
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nomenclature_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article TEXT,
                    name TEXT NOT NULL,
                    standard TEXT,
                    item_type TEXT,
                    level TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    params TEXT,
                    ens_code TEXT,
                    ens_name TEXT,
                    ens_params TEXT,
                    ens_params_mask TEXT,
                    confidence REAL,
                    match_type TEXT,
                    match_type_ru TEXT,
                    coating_substitution TEXT,
                    fuzzy_mismatched_params TEXT,
                    mask_id INTEGER,
                    mask_pattern TEXT,
                    mask_pattern_hash TEXT,
                    details TEXT,
                    processing_time_ms REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, standard)
                )
            """)
            # Индексы для быстрого поиска
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON nomenclature_results(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_standard ON nomenclature_results(standard)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name_std ON nomenclature_results(name, standard)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ens_code ON nomenclature_results(ens_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_success ON nomenclature_results(success)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mask_hash ON nomenclature_results(mask_pattern_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_updated ON nomenclature_results(updated_at)")
            conn.commit()

    @staticmethod
    def _hash_mask(pattern: Optional[str]) -> str:
        """Хеш маски для отслеживания изменений."""
        if not pattern:
            return ""
        return hashlib.md5(pattern.encode("utf-8")).hexdigest()[:16]

    def upsert_result(
        self,
        name: str,
        article: Optional[str] = None,
        standard: Optional[str] = None,
        item_type: Optional[str] = None,
        level: Optional[str] = None,
        success: bool = False,
        params: Optional[Dict] = None,
        ens_code: Optional[str] = None,
        ens_name: Optional[str] = None,
        ens_params: Optional[Dict] = None,
        ens_params_mask: Optional[Dict] = None,
        confidence: float = 0.0,
        match_type: Optional[str] = None,
        match_type_ru: Optional[str] = None,
        coating_substitution: Optional[Dict] = None,
        fuzzy_mismatched_params: Optional[Dict] = None,
        mask_id: Optional[int] = None,
        mask_pattern: Optional[str] = None,
        details: Optional[Dict] = None,
        processing_time_ms: float = 0.0,
    ) -> Tuple[bool, str]:
        """Upsert результата. Возвращает (changed, reason)."""
        logger.info("[RESULT_DB] upsert_result: name=%r standard=%r ens_code=%s", name[:60], standard, ens_code)
        """
        Upsert результата. Возвращает (changed, reason).
        changed=True если запись новая или маска изменилась.
        Ключ: name + standard.
        """
        mask_hash = self._hash_mask(mask_pattern)
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Проверяем существующую запись по name + standard
            cursor = conn.execute(
                "SELECT id, mask_pattern_hash FROM nomenclature_results WHERE name = ? AND (standard = ? OR (standard IS NULL AND ? IS NULL))",
                (name, standard, standard)
            )
            existing = cursor.fetchone()

            if existing:
                existing_id, existing_hash = existing
                if existing_hash == mask_hash:
                    # Маска не изменилась — обновляем только время (soft update)
                    conn.execute(
                        "UPDATE nomenclature_results SET updated_at = ? WHERE id = ?",
                        (now, existing_id)
                    )
                    conn.commit()
                    logger.info("[RESULT_DB] UNCHANGED (mask same) for name=%r", name[:60])
                    return False, "mask_unchanged"
                else:
                    # Маска изменилась — полное обновление
                    logger.info("[RESULT_DB] Mask changed for '%s' / %s: %s -> %s",
                                name[:50], standard, existing_hash, mask_hash)
                    conn.execute(
                        """UPDATE nomenclature_results SET
                            article = ?, standard = ?, item_type = ?, level = ?, success = ?,
                            params = ?, ens_code = ?, ens_name = ?, ens_params = ?,
                            ens_params_mask = ?, confidence = ?, match_type = ?,
                            match_type_ru = ?, coating_substitution = ?,
                            fuzzy_mismatched_params = ?, mask_id = ?, mask_pattern = ?,
                            mask_pattern_hash = ?, details = ?, processing_time_ms = ?,
                            updated_at = ?
                        WHERE id = ?""",
                        (
                            article, standard, item_type, level, int(success),
                            _json_dumps(params), ens_code, ens_name,
                            _json_dumps(ens_params), _json_dumps(ens_params_mask),
                            confidence, match_type, match_type_ru,
                            _json_dumps(coating_substitution),
                            _json_dumps(fuzzy_mismatched_params),
                            mask_id, mask_pattern, mask_hash,
                            _json_dumps(details), processing_time_ms,
                            now, existing_id
                        )
                    )
                    conn.commit()
                    logger.info("[RESULT_DB] UPDATED (mask changed) for name=%r", name[:60])
                    return True, "mask_changed"
            else:
                # Новая запись
                conn.execute(
                    """INSERT INTO nomenclature_results (
                        article, name, standard, item_type, level, success,
                        params, ens_code, ens_name, ens_params, ens_params_mask,
                        confidence, match_type, match_type_ru, coating_substitution,
                        fuzzy_mismatched_params, mask_id, mask_pattern, mask_pattern_hash,
                        details, processing_time_ms, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        article, name, standard, item_type, level, int(success),
                        _json_dumps(params), ens_code, ens_name,
                        _json_dumps(ens_params), _json_dumps(ens_params_mask),
                        confidence, match_type, match_type_ru,
                        _json_dumps(coating_substitution),
                        _json_dumps(fuzzy_mismatched_params),
                        mask_id, mask_pattern, mask_hash,
                        _json_dumps(details), processing_time_ms,
                        now, now
                    )
                )
                conn.commit()
                return True, "new_record"

    def get_result(self, name: str, standard: Optional[str] = None) -> Optional[Dict]:
        """Получить одну запись по наименованию (и опционально по стандарту).
        Fallback: если не нашли по name+standard, ищем только по name."""
        logger.info("[RESULT_DB] get_result: name=%r standard=%r", name[:60], standard)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) FROM nomenclature_results").fetchone()[0]
            logger.info("[RESULT_DB] Total records in DB: %d", total)

            # 1. Точный поиск по name + standard
            if standard:
                cursor = conn.execute(
                    "SELECT * FROM nomenclature_results WHERE name = ? AND standard = ? ORDER BY updated_at DESC LIMIT 1",
                    (name, standard)
                )
                row = cursor.fetchone()
                if row:
                    logger.info("[RESULT_DB] FOUND (exact): id=%s ens_code=%s", row['id'], row['ens_code'])
                    return self._deserialize_row(dict(row))

            # 2. Fallback: поиск только по name (без standard)
            cursor = conn.execute(
                "SELECT * FROM nomenclature_results WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
                (name,)
            )
            row = cursor.fetchone()
            if row:
                logger.info("[RESULT_DB] FOUND (fallback by name only): id=%s ens_code=%s standard=%s", row['id'], row['ens_code'], row['standard'])
                return self._deserialize_row(dict(row))

            # 3. Fallback: поиск по name с конечной запятой (совместимость со старыми записями)
            old_name = name + ','
            cursor = conn.execute(
                "SELECT * FROM nomenclature_results WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
                (old_name,)
            )
            row = cursor.fetchone()
            if row:
                logger.info("[RESULT_DB] FOUND (old format with comma): id=%s", row['id'])
                return self._deserialize_row(dict(row))

            logger.info("[RESULT_DB] NOT FOUND for name=%r standard=%r", name[:60], standard)
            return None

    def get_all_results(
        self,
        success: Optional[bool] = None,
        ens_code: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict]:
        """Получить список результатов с фильтрами."""
        query = "SELECT * FROM nomenclature_results WHERE 1=1"
        params = []
        if success is not None:
            query += " AND success = ?"
            params.append(int(success))
        if ens_code:
            query += " AND ens_code = ?"
            params.append(ens_code)
        query += " ORDER BY updated_at DESC"
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [self._deserialize_row(dict(r)) for r in rows]

    def get_changed_records(self, since: Optional[str] = None) -> List[Dict]:
        """Получить записи, измененные после указанной даты (ISO format)."""
        query = "SELECT * FROM nomenclature_results WHERE 1=1"
        params = []
        if since:
            query += " AND updated_at > ?"
            params.append(since)
        query += " ORDER BY updated_at DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [self._deserialize_row(dict(r)) for r in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика по результатам."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM nomenclature_results").fetchone()[0]
            success = conn.execute("SELECT COUNT(*) FROM nomenclature_results WHERE success = 1").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM nomenclature_results WHERE success = 0").fetchone()[0]
            changed = conn.execute(
                "SELECT COUNT(*) FROM nomenclature_results WHERE created_at != updated_at"
            ).fetchone()[0]
            by_match_type = {}
            cursor = conn.execute(
                "SELECT match_type, COUNT(*) as cnt FROM nomenclature_results GROUP BY match_type"
            )
            for row in cursor.fetchall():
                by_match_type[row[0] or "unknown"] = row[1]

            return {
                "total": total,
                "success": success,
                "failed": failed,
                "changed_after_insert": changed,
                "by_match_type": by_match_type,
                "success_rate": round(success / total, 3) if total else 0.0,
            }

    def export_to_excel(
        self,
        output_path: str,
        source_df=None,
        source_path: Optional[str] = None,
        article_col: str = "Артикул",
        name_col: str = "наименование",
        extra_columns: Optional[List[str]] = None,
    ) -> str:
        """
        Экспорт результатов в Excel.
        Если source_df/source_path переданы — обогащаем их колонками из БД.
        Иначе экспортируем все записи из БД.
        """
        import pandas as pd

        if source_df is not None:
            df = source_df.copy()
        elif source_path:
            df = pd.read_excel(source_path)
        else:
            # Экспорт всех записей из БД
            records = self.get_all_results()
            if not records:
                logger.warning("No records to export")
                return output_path
            df = pd.DataFrame(records)
            # Убираем служебные колонки
            drop_cols = ["id", "params", "ens_params", "ens_params_mask", "details",
                         "mask_pattern", "mask_pattern_hash", "created_at"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            df.to_excel(output_path, index=False)
            logger.info("Exported %d records to %s", len(df), output_path)
            return output_path

        # Обогащаем исходный DataFrame данными из БД
        # Создаем lookup по name
        lookup: Dict[str, Dict] = {}
        for record in self.get_all_results():
            key = str(record.get("name", "")).strip()
            lookup[key] = record

        # Добавляем колонки
        default_cols = [
            "ens_code", "ens_name", "level", "success", "confidence",
            "match_type_ru", "coating_substitution", "fuzzy_mismatched_params"
        ]
        cols = extra_columns or default_cols

        for col in cols:
            df[col] = None

        for idx, row in df.iterrows():
            nam = str(row.get(name_col, "")).strip()
            rec = lookup.get(nam)
            if rec:
                for col in cols:
                    val = rec.get(col)
                    if col in ("coating_substitution", "fuzzy_mismatched_params"):
                        val = _json_dumps(val) if val else None
                    elif col == "success":
                        val = "Да" if val else "Нет"
                    elif col == "confidence" and val is not None:
                        val = round(float(val), 3)
                    df.at[idx, col] = val

        df.to_excel(output_path, index=False)
        logger.info("Enriched Excel exported to %s (%d rows)", output_path, len(df))
        return output_path

    def _deserialize_row(self, row: Dict) -> Dict:
        """Десериализация JSON-полей из строки БД."""
        json_fields = [
            "params", "ens_params", "ens_params_mask", "coating_substitution",
            "fuzzy_mismatched_params", "details"
        ]
        for field in json_fields:
            if field in row and row[field] is not None:
                row[field] = _json_loads(row[field])
        if "success" in row:
            row["success"] = bool(row["success"])
        return row