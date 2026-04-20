"""
Mask Database Module
SQLite-based storage for regex masks with auto-validation support.
"""

import sqlite3
import json
import logging
import hashlib
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
from queue import Queue, Empty

logger = logging.getLogger(__name__)


@dataclass
class MaskRecord:
    """Запись маски в базе данных."""
    id: Optional[int] = None
    standard: str = ""
    item_type: str = ""
    pattern: str = ""
    params: List[str] = field(default_factory=list)
    required: List[str] = field(default_factory=list)
    auto_score: float = 0.0
    is_active: bool = False
    source: str = "llm"  # 'llm', 'default', 'manual'
    usage_count: int = 0
    test_examples: List[Dict] = field(default_factory=list)
    created_at: Optional[str] = None
    last_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            'id': self.id,
            'standard': self.standard,
            'item_type': self.item_type,
            'pattern': self.pattern,
            'params': json.dumps(self.params, ensure_ascii=False),
            'required': json.dumps(self.required, ensure_ascii=False),
            'auto_score': self.auto_score,
            'is_active': self.is_active,
            'source': self.source,
            'usage_count': self.usage_count,
            'test_examples': json.dumps(self.test_examples, ensure_ascii=False),
            'created_at': self.created_at or datetime.utcnow().isoformat(),
            'last_used': self.last_used
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> 'MaskRecord':
        """Создание из строки БД."""
        return cls(
            id=row['id'],
            standard=row['standard'],
            item_type=row['item_type'],
            pattern=row['pattern'],
            params=json.loads(row['params']) if row['params'] else [],
            required=json.loads(row['required']) if row['required'] else [],
            auto_score=row['auto_score'],
            is_active=bool(row['is_active']),
            source=row['source'],
            usage_count=row['usage_count'],
            test_examples=json.loads(row['test_examples']) if row['test_examples'] else [],
            created_at=row['created_at'],
            last_used=row['last_used']
        )

    @property
    def pattern_hash(self) -> str:
        """Уникальный хеш паттерна."""
        content = f"{self.pattern}:{self.standard}:{self.item_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ConnectionPool:
    """Пул соединений SQLite с WAL mode."""

    def __init__(self, db_path: str, max_connections: int = 5, timeout: float = 30.0):
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: Queue = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._connections_created = 0
        self._initialize_pool()

    def _initialize_pool(self):
        """Инициализация пула соединений."""
        for _ in range(self.max_connections):
            conn = self._create_connection()
            self._pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Создание нового соединения с WAL mode."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
            isolation_level=None
        )
        conn.row_factory = sqlite3.Row

        # WAL mode и оптимизации
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA busy_timeout = 5000")
        cursor.execute("PRAGMA cache_size = -64000")  # 64MB
        cursor.execute("PRAGMA foreign_keys = ON")

        with self._lock:
            self._connections_created += 1

        return conn

    def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        """Проверка валидности соединения."""
        try:
            conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def get_connection(self) -> sqlite3.Connection:
        """Получение соединения из пула."""
        try:
            conn = self._pool.get(timeout=self.timeout)
            if not self._validate_connection(conn):
                conn.close()
                conn = self._create_connection()
            return conn
        except Empty:
            raise TimeoutError(f"Could not acquire connection within {self.timeout}s")

    def release_connection(self, conn: sqlite3.Connection):
        """Возврат соединения в пул."""
        try:
            conn.rollback()
            self._pool.put_nowait(conn)
        except:
            conn.close()

    @contextmanager
    def connection(self):
        """Контекстный менеджер для соединения."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.release_connection(conn)

    def close_all(self):
        """Закрытие всех соединений."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break


class MaskDatabase:
    """
    База данных масок с авто-валидацией.

    Features:
    - SQLite с WAL mode
    - Connection pooling
    - Уникальность по hash(pattern + standard)
    - Авто-активация при score >= 0.85
    """

    def __init__(self, db_path: str = "masks.db", max_connections: int = 5):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pool = ConnectionPool(str(self.db_path), max_connections)
        self._init_db()

    def _init_db(self):
        """Инициализация схемы БД."""
        with self.pool.connection() as conn:
            cursor = conn.cursor()

            # Основная таблица масок
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS masks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    standard TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    params TEXT,  -- JSON array
                    required TEXT,  -- JSON array
                    auto_score REAL DEFAULT 0.0,
                    is_active INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'llm',
                    usage_count INTEGER DEFAULT 0,
                    test_examples TEXT,  -- JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    pattern_hash TEXT UNIQUE,

                    -- Индексы для быстрого поиска
                    UNIQUE(standard, item_type, pattern_hash)
                )
            """)

            # Индексы
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_masks_standard_type 
                ON masks(standard, item_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_masks_active 
                ON masks(is_active, auto_score DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_masks_source 
                ON masks(source)
            """)

            # Таблица для логов валидации
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mask_id INTEGER,
                    test_count INTEGER,
                    success_count INTEGER,
                    score REAL,
                    details TEXT,  -- JSON
                    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (mask_id) REFERENCES masks(id)
                )
            """)

            conn.commit()
            logger.info(f"Database initialized: {self.db_path}")

    def save_mask(self, mask: MaskRecord, auto_activate: bool = True) -> Optional[int]:
        """
        Сохранение маски.

        Args:
            mask: Запись маски
            auto_activate: Автоматически активировать если score >= 0.85

        Returns:
            ID сохраненной маски или None при ошибке
        """
        try:
            with self.pool.connection() as conn:
                cursor = conn.cursor()

                # Проверяем auto-activation
                if auto_activate and mask.auto_score >= 0.85:
                    mask.is_active = True
                    logger.info(f"Auto-activating mask for {mask.standard} (score: {mask.auto_score:.2f})")

                mask_data = mask.to_dict()
                mask_data['pattern_hash'] = mask.pattern_hash

                cursor.execute("""
                    INSERT INTO masks 
                    (standard, item_type, pattern, params, required, auto_score, 
                     is_active, source, usage_count, test_examples, created_at, 
                     last_used, pattern_hash)
                    VALUES 
                    (:standard, :item_type, :pattern, :params, :required, :auto_score,
                     :is_active, :source, :usage_count, :test_examples, :created_at,
                     :last_used, :pattern_hash)
                    ON CONFLICT(pattern_hash) DO UPDATE SET
                        auto_score = excluded.auto_score,
                        is_active = excluded.is_active,
                        test_examples = excluded.test_examples,
                        last_used = CURRENT_TIMESTAMP
                    RETURNING id
                """, mask_data)

                result = cursor.fetchone()
                conn.commit()

                if result:
                    mask.id = result[0]
                    logger.debug(f"Mask saved with ID: {mask.id}")
                    return mask.id

        except Exception as e:
            logger.error(f"Failed to save mask: {e}")
            return None

        return None

    def get_mask(self, standard: str, item_type: str, prefer_active: bool = True) -> Optional[MaskRecord]:
        """
        Получение лучшей маски для (standard, type).

        Args:
            standard: Стандарт (ГОСТ, ОСТ, etc.)
            item_type: Тип изделия (болт, гайка, etc.)
            prefer_active: Предпочитать активные маски

        Returns:
            MaskRecord или None
        """
        try:
            with self.pool.connection() as conn:
                cursor = conn.cursor()

                if prefer_active:
                    # Сначала ищем активные с лучшим score
                    cursor.execute("""
                        SELECT * FROM masks 
                        WHERE standard = ? AND item_type = ? AND is_active = 1
                        ORDER BY auto_score DESC, usage_count DESC
                        LIMIT 1
                    """, (standard, item_type))
                else:
                    cursor.execute("""
                        SELECT * FROM masks 
                        WHERE standard = ? AND item_type = ?
                        ORDER BY auto_score DESC, usage_count DESC
                        LIMIT 1
                    """, (standard, item_type))

                row = cursor.fetchone()

                if row:
                    # Обновляем usage_count и last_used
                    cursor.execute("""
                        UPDATE masks 
                        SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (row['id'],))
                    conn.commit()

                    return MaskRecord.from_row(row)

        except Exception as e:
            logger.error(f"Failed to get mask: {e}")

        return None

    def get_mask_by_pattern(self, pattern: str, standard: str) -> Optional[MaskRecord]:
        """Получение маски по паттерну."""
        temp_mask = MaskRecord(pattern=pattern, standard=standard, item_type="")

        try:
            with self.pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM masks WHERE pattern_hash = ?
                """, (temp_mask.pattern_hash,))

                row = cursor.fetchone()
                if row:
                    return MaskRecord.from_row(row)
        except Exception as e:
            logger.error(f"Failed to get mask by pattern: {e}")

        return None

    def list_masks(
        self, 
        standard: Optional[str] = None,
        item_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        source: Optional[str] = None,
        min_score: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[MaskRecord]:
        """Список масок с фильтрацией."""
        query = "SELECT * FROM masks WHERE 1=1"
        params = []

        if standard:
            query += " AND standard = ?"
            params.append(standard)
        if item_type:
            query += " AND item_type = ?"
            params.append(item_type)
        if is_active is not None:
            query += " AND is_active = ?"
            params.append(1 if is_active else 0)
        if source:
            query += " AND source = ?"
            params.append(source)
        if min_score is not None:
            query += " AND auto_score >= ?"
            params.append(min_score)

        query += " ORDER BY auto_score DESC, usage_count DESC"

        if limit:
            query += f" LIMIT {int(limit)}"

        try:
            with self.pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [MaskRecord.from_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list masks: {e}")
            return []

    def update_mask_score(self, mask_id: int, new_score: float, test_examples: List[Dict]):
        """Обновление score маски после валидации."""
        try:
            with self.pool.connection() as conn:
                cursor = conn.cursor()

                is_active = new_score >= 0.85

                cursor.execute("""
                    UPDATE masks 
                    SET auto_score = ?, is_active = ?, test_examples = ?
                    WHERE id = ?
                """, (new_score, is_active, json.dumps(test_examples, ensure_ascii=False), mask_id))

                conn.commit()
                logger.info(f"Updated mask {mask_id}: score={new_score:.2f}, active={is_active}")
                return True
        except Exception as e:
            logger.error(f"Failed to update mask score: {e}")
            return False

    def log_validation(self, mask_id: int, test_count: int, success_count: int, 
                       score: float, details: Dict):
        """Логирование валидации."""
        try:
            with self.pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO validation_logs 
                    (mask_id, test_count, success_count, score, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (mask_id, test_count, success_count, score, json.dumps(details, ensure_ascii=False)))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log validation: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика по маскам."""
        try:
            with self.pool.connection() as conn:
                cursor = conn.cursor()

                stats = {
                    'total': cursor.execute("SELECT COUNT(*) FROM masks").fetchone()[0],
                    'active': cursor.execute("SELECT COUNT(*) FROM masks WHERE is_active = 1").fetchone()[0],
                    'by_source': {},
                    'by_standard': {},
                    'avg_score': cursor.execute("SELECT AVG(auto_score) FROM masks").fetchone()[0] or 0.0
                }

                cursor.execute("SELECT source, COUNT(*) FROM masks GROUP BY source")
                for row in cursor.fetchall():
                    stats['by_source'][row[0]] = row[1]

                cursor.execute("SELECT standard, COUNT(*) FROM masks GROUP BY standard")
                for row in cursor.fetchall():
                    stats['by_standard'][row[0]] = row[1]

                return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def delete_mask(self, mask_id: int) -> bool:
        """Удаление маски."""
        try:
            with self.pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM masks WHERE id = ?", (mask_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete mask: {e}")
            return False

    def cleanup_low_score_masks(self, threshold: float = 0.5) -> int:
        """Очистка масок с низким score."""
        try:
            with self.pool.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM masks 
                    WHERE auto_score < ? AND is_active = 0 AND source = 'llm'
                """, (threshold,))
                conn.commit()
                deleted = cursor.rowcount
                logger.info(f"Cleaned up {deleted} low-score masks")
                return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup masks: {e}")
            return 0

    def close(self):
        """Закрытие пула соединений."""
        self.pool.close_all()
