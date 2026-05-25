"""
Mask Database Module
SQLite-based storage for regex masks with auto-validation support.
"""
# =============================================================================
# FIX 2026-05-22 11:09 UTC+3:
# 1. _compute_pattern_hash now includes standard+item_type to avoid collisions.
# 2. save_mask: replace_existing uses UPDATE instead of DELETE+INSERT.
# 3. save_mask: INSERT uses ON CONFLICT(pattern_hash) DO UPDATE for upsert.
# 4. Added replace_existing_fallback: if exact (std+type) not found,
#    delete any mask with conflicting pattern_hash before insert.
# =============================================================================

import sqlite3
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils.standard_utils import canonicalize_standard

logger = logging.getLogger(__name__)


@dataclass
class MaskRecord:
    """Запись маски в БД."""
    id: Optional[int] = None
    standard: str = ""
    item_type: str = ""
    pattern: str = ""
    params: List[str] = field(default_factory=list)
    required: List[str] = field(default_factory=list)
    auto_score: float = 0.0
    is_active: bool = False
    source: str = ""
    test_examples: int = 0
    pattern_hash: str = ""
    created_at: Optional[str] = None
    last_used: Optional[str] = None
    usage_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'standard': self.standard,
            'item_type': self.item_type,
            'pattern': self.pattern,
            'params': self.params,
            'required': self.required,
            'auto_score': self.auto_score,
            'is_active': self.is_active,
            'source': self.source,
            'test_examples': self.test_examples,
            'pattern_hash': self.pattern_hash,
            'created_at': self.created_at or datetime.utcnow().isoformat(),
            'last_used': self.last_used,
            'usage_count': self.usage_count
        }


class MaskDatabase:
    """База данных масок SQLite."""

    def __init__(self, db_path: str = "cache/masks.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Инициализация таблиц БД."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS masks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    standard TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    params TEXT,
                    required TEXT,
                    auto_score REAL DEFAULT 0.0,
                    is_active BOOLEAN DEFAULT 0,
                    source TEXT DEFAULT '',
                    test_examples INTEGER DEFAULT 0,
                    pattern_hash TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_masks_standard_item_type
                ON masks(standard, item_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_masks_pattern_hash
                ON masks(pattern_hash)
            """)
            conn.commit()
            logger.info(f"[MaskDB] Initialized: {self.db_path}")
            self._migrate_standards()

    def _migrate_standards(self):
        """Миграция: привести все существующие standard к каноническому виду."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, standard FROM masks")
            rows = cursor.fetchall()
            updated = 0
            for row_id, std in rows:
                canon = canonicalize_standard(std)
                if canon != std:
                    cursor.execute(
                        "UPDATE masks SET standard = ? WHERE id = ?",
                        (canon, row_id)
                    )
                    updated += 1
            conn.commit()
            if updated:
                logger.info("[MaskDB] Migrated %d standards to canonical form", updated)

    def _compute_pattern_hash(self, pattern: str, standard: str = "", item_type: str = "") -> str:
        """Вычисление хеша паттерна для дедупликации.

        FIX 2026-05-22: включает standard и item_type, чтобы маски
        для разных стандартов с похожими паттернами не конфликтовали.
        """
        key = f"{standard}:{item_type}:{pattern}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def get_mask(self, standard: str, item_type: str) -> Optional[MaskRecord]:
        """Получение маски по стандарту и типу (standard канонизируется)."""
        canon_std = canonicalize_standard(standard)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, standard, item_type, pattern, params, required,
                       auto_score, is_active, source, test_examples, pattern_hash,
                       created_at, last_used, usage_count
                FROM masks
                WHERE standard = ? AND item_type = ?
                ORDER BY auto_score DESC, created_at DESC
                LIMIT 1
            """, (canon_std, item_type))
            row = cursor.fetchone()
            if row:
                return MaskRecord(
                    id=row[0],
                    standard=row[1],
                    item_type=row[2],
                    pattern=row[3],
                    params=row[4].split(',') if row[4] else [],
                    required=row[5].split(',') if row[5] else [],
                    auto_score=row[6],
                    is_active=bool(row[7]),
                    source=row[8],
                    test_examples=row[9],
                    pattern_hash=row[10],
                    created_at=row[11],
                    last_used=row[12],
                    usage_count=row[13]
                )
        return None

    def get_mask_by_id(self, mask_id: int) -> Optional[MaskRecord]:
        """Получение маски по ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, standard, item_type, pattern, params, required,
                       auto_score, is_active, source, test_examples, pattern_hash,
                       created_at, last_used, usage_count
                FROM masks
                WHERE id = ?
            """, (mask_id,))
            row = cursor.fetchone()
            if row:
                return MaskRecord(
                    id=row[0],
                    standard=row[1],
                    item_type=row[2],
                    pattern=row[3],
                    params=row[4].split(',') if row[4] else [],
                    required=row[5].split(',') if row[5] else [],
                    auto_score=row[6],
                    is_active=bool(row[7]),
                    source=row[8],
                    test_examples=row[9],
                    pattern_hash=row[10],
                    created_at=row[11],
                    last_used=row[12],
                    usage_count=row[13]
                )
        return None

    def save_mask(self, mask: MaskRecord, auto_activate: bool = True,
                  replace_existing: bool = False) -> Optional[int]:
        """
        Сохранение маски в БД.

        FIX 2026-05-22: pattern_hash включает standard+item_type.
        При replace_existing=True используется UPDATE вместо DELETE+INSERT
        для предотвращения UNIQUE constraint failed.
        """
        import json
        canon_std = canonicalize_standard(mask.standard)
        mask.standard = canon_std
        pattern_hash = self._compute_pattern_hash(mask.pattern, canon_std, mask.item_type)
        mask.pattern_hash = pattern_hash

        params_json = json.dumps(mask.params) if mask.params else ''
        required_json = json.dumps(mask.required) if mask.required else ''

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # --- FIX: replace_existing via UPDATE, not DELETE+INSERT ---
            if replace_existing:
                cursor.execute(
                    """SELECT id FROM masks
                       WHERE standard = ? AND item_type = ?
                       ORDER BY created_at DESC LIMIT 1""",
                    (canon_std, mask.item_type)
                )
                existing = cursor.fetchone()
                if existing:
                    existing_id = existing[0]
                    cursor.execute(
                        """UPDATE masks SET
                           pattern = ?, params = ?, required = ?,
                           pattern_hash = ?, is_active = ?,
                           auto_score = ?, source = ?,
                           last_used = CURRENT_TIMESTAMP
                           WHERE id = ?""",
                        (mask.pattern, params_json, required_json,
                         pattern_hash, 1 if auto_activate else 0,
                         mask.auto_score, mask.source, existing_id)
                    )
                    conn.commit()
                    logger.info(
                        "[MaskDB] Replaced mask #%d for %s/%s (hash=%s)",
                        existing_id, canon_std, mask.item_type, pattern_hash[:8]
                    )
                    return existing_id

            # --- Если replace_existing=False или маска не найдена — UPSERT ---
            try:
                cursor.execute(
                    """INSERT INTO masks
                       (standard, item_type, pattern, params, required,
                        pattern_hash, is_active, auto_score, source,
                        test_examples, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(pattern_hash) DO UPDATE SET
                           standard = excluded.standard,
                           item_type = excluded.item_type,
                           pattern = excluded.pattern,
                           params = excluded.params,
                           required = excluded.required,
                           is_active = excluded.is_active,
                           auto_score = excluded.auto_score,
                           source = excluded.source,
                           last_used = CURRENT_TIMESTAMP
                       RETURNING id""",
                    (canon_std, mask.item_type, mask.pattern,
                     params_json, required_json, pattern_hash,
                     1 if auto_activate else 0, mask.auto_score,
                     mask.source, mask.test_examples)
                )
                row = cursor.fetchone()
                conn.commit()
                if row:
                    new_id = row[0]
                    logger.info(
                        "[MaskDB] Saved mask #%d for %s/%s (hash=%s)",
                        new_id, canon_std, mask.item_type, pattern_hash[:8]
                    )
                    return new_id
            except sqlite3.IntegrityError as e:
                logger.error(
                    "[MaskDB] UNIQUE conflict on pattern_hash=%s for %s/%s: %s",
                    pattern_hash[:8], canon_std, mask.item_type, e
                )
                # Fallback: delete conflicting mask and retry
                cursor.execute(
                    "DELETE FROM masks WHERE pattern_hash = ?",
                    (pattern_hash,)
                )
                cursor.execute(
                    """INSERT INTO masks
                       (standard, item_type, pattern, params, required,
                        pattern_hash, is_active, auto_score, source,
                        test_examples, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                       RETURNING id""",
                    (canon_std, mask.item_type, mask.pattern,
                     params_json, required_json, pattern_hash,
                     1 if auto_activate else 0, mask.auto_score,
                     mask.source, mask.test_examples)
                )
                row = cursor.fetchone()
                conn.commit()
                if row:
                    new_id = row[0]
                    logger.warning(
                        "[MaskDB] Fallback save mask #%d for %s/%s (deleted conflict)",
                        new_id, canon_std, mask.item_type
                    )
                    return new_id
                raise
        return None

    def update_usage(self, mask_id: int):
        """Обновление счётчика использования маски."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE masks
                   SET usage_count = usage_count + 1,
                       last_used = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (mask_id,)
            )
            conn.commit()

    def cleanup_low_score_masks(self, min_score: float = 0.5):
        """Удаление масок с низким auto_score."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM masks WHERE auto_score < ? AND is_active = 0",
                (min_score,)
            )
            deleted = cursor.rowcount
            conn.commit()
            if deleted:
                logger.info("[MaskDB] Cleaned up %d low-score masks", deleted)
            return deleted

    def get_all_masks(self, standard: Optional[str] = None,
                      item_type: Optional[str] = None) -> List[MaskRecord]:
        """Получение всех масок (с фильтрами)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if standard and item_type:
                cursor.execute(
                    """SELECT id, standard, item_type, pattern, params, required,
                           auto_score, is_active, source, test_examples, pattern_hash,
                           created_at, last_used, usage_count
                    FROM masks WHERE standard = ? AND item_type = ?
                    ORDER BY auto_score DESC""",
                    (standard, item_type)
                )
            elif standard:
                cursor.execute(
                    """SELECT id, standard, item_type, pattern, params, required,
                           auto_score, is_active, source, test_examples, pattern_hash,
                           created_at, last_used, usage_count
                    FROM masks WHERE standard = ?
                    ORDER BY auto_score DESC""",
                    (standard,)
                )
            else:
                cursor.execute(
                    """SELECT id, standard, item_type, pattern, params, required,
                           auto_score, is_active, source, test_examples, pattern_hash,
                           created_at, last_used, usage_count
                    FROM masks ORDER BY auto_score DESC"""
                )
            rows = cursor.fetchall()
            return [
                MaskRecord(
                    id=r[0], standard=r[1], item_type=r[2], pattern=r[3],
                    params=r[4].split(',') if r[4] else [],
                    required=r[5].split(',') if r[5] else [],
                    auto_score=r[6], is_active=bool(r[7]),
                    source=r[8], test_examples=r[9],
                    pattern_hash=r[10], created_at=r[11],
                    last_used=r[12], usage_count=r[13]
                )
                for r in rows
            ]

    def get_mask_stats(self) -> Dict[str, Any]:
        """Статистика по маскам."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM masks")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM masks WHERE is_active = 1")
            active = cursor.fetchone()[0]
            cursor.execute("SELECT AVG(auto_score) FROM masks")
            avg_score = cursor.fetchone()[0] or 0.0
            return {
                "total": total,
                "active": active,
                "inactive": total - active,
                "average_score": round(avg_score, 3)
            }