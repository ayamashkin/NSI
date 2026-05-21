"""
Mask Database Module
SQLite-based storage for regex masks with auto-validation support.

LAST_FIX: 2026-05-20 2026-05-20 14:55 UTC+3 UTC+3 — save_mask: created_at обновляется
  при replace_existing и ON CONFLICT (ранее оставался старый timestamp,
  что приводило к stale cache в result.db).
"""
import sqlite3
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

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

    def _compute_pattern_hash(self, pattern: str) -> str:
        """Вычисление хеша паттерна для дедупликации."""
        return hashlib.md5(pattern.encode('utf-8')).hexdigest()

    def get_mask(self, standard: str, item_type: str) -> Optional[MaskRecord]:
        """Получение маски по стандарту и типу."""
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
            """, (standard, item_type))
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

    def save_mask(
        self,
        mask: MaskRecord,
        auto_activate: bool = True,
        replace_existing: bool = False
    ) -> Optional[int]:
        """Сохранение маски в БД.

        Args:
            mask: запись маски
            auto_activate: автоматически активировать если score >= threshold
            replace_existing: заменить существующую маску для той же пары (standard, item_type)

        Returns:
            ID сохраненной маски или None
        """
        pattern_hash = self._compute_pattern_hash(mask.pattern)
        mask_data = {
            'standard': mask.standard,
            'item_type': mask.item_type.upper(),
            'pattern': mask.pattern,
            'params': ','.join(mask.params),
            'required': ','.join(mask.required),
            'auto_score': mask.auto_score,
            'is_active': mask.is_active,
            'source': mask.source,
            'test_examples': mask.test_examples,
            'pattern_hash': pattern_hash
        }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Проверяем существующую маску для той же пары (standard, item_type)
            if replace_existing:
                cursor.execute("""
                    SELECT id FROM masks
                    WHERE standard = :standard AND item_type = :item_type
                    ORDER BY created_at DESC LIMIT 1
                """, mask_data)
                existing = cursor.fetchone()
                if existing:
                    existing_id = existing[0]
                    logger.info(f"[MaskDB] Replacing mask #{existing_id} for {mask.standard}/{mask.item_type}")
                    cursor.execute("""
                        UPDATE masks SET
                        pattern = :pattern,
                        params = :params,
                        required = :required,
                        auto_score = :auto_score,
                        is_active = :is_active,
                        source = :source,
                        test_examples = :test_examples,
                        pattern_hash = :pattern_hash,
                        created_at = CURRENT_TIMESTAMP,
                        last_used = CURRENT_TIMESTAMP
                        WHERE id = :existing_id
                    """, {**mask_data, 'existing_id': existing_id})
                    conn.commit()
                    logger.info(f"[MaskDB] Mask #{existing_id} replaced (created_at updated)")
                    return existing_id

            # INSERT или UPDATE по pattern_hash
            cursor.execute("""
                INSERT INTO masks (
                    standard, item_type, pattern, params, required,
                    auto_score, is_active, source, test_examples, pattern_hash
                ) VALUES (
                    :standard, :item_type, :pattern, :params, :required,
                    :auto_score, :is_active, :source, :test_examples, :pattern_hash
                )
                ON CONFLICT(pattern_hash) DO UPDATE SET
                    auto_score = excluded.auto_score,
                    is_active = excluded.is_active,
                    source = excluded.source,
                    test_examples = excluded.test_examples,
                    created_at = CURRENT_TIMESTAMP,
                    last_used = CURRENT_TIMESTAMP
                RETURNING id
            """, mask_data)
            result = cursor.fetchone()
            conn.commit()
            if result:
                mask_id = result[0]
                logger.info(f"[MaskDB] Mask saved: ID={mask_id}, {mask.standard}/{mask.item_type}, score={mask.auto_score:.2f}")
                return mask_id
            return None

    def update_usage(self, mask_id: int):
        """Обновление счетчика использования маски."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE masks
                SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (mask_id,))
            conn.commit()

    def cleanup_low_score_masks(self, threshold: float = 0.5) -> int:
        """Удаление масок с низким score."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM masks
                WHERE auto_score < ? AND is_active = 0
            """, (threshold,))
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"[MaskDB] Cleaned up {deleted} low-score masks")
            return deleted

    def get_all_masks(self) -> List[MaskRecord]:
        """Получение всех масок."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, standard, item_type, pattern, params, required,
                       auto_score, is_active, source, test_examples, pattern_hash,
                       created_at, last_used, usage_count
                FROM masks
                ORDER BY standard, item_type
            """)
            rows = cursor.fetchall()
            return [
                MaskRecord(
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
                for row in rows
            ]

    def get_mask_stats(self) -> Dict[str, Any]:
        """Статистика по маскам."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active,
                    AVG(auto_score) as avg_score
                FROM masks
            """)
            row = cursor.fetchone()
            return {
                'total': row[0],
                'active': row[1],
                'avg_score': round(row[2], 3) if row[2] else 0.0
            }