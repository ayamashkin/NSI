"""
Database Manager Module
SQLite-based storage with UPSERT semantics for processing results.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from datetime import datetime
from dataclasses import asdict

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Менеджер базы данных SQLite для хранения результатов обработки номенклатуры.

    Features:
    - UPSERT семантика: (article, prompt_id) - уникальный ключ
    - Автоматическое создание индексов
    - Поддержка JSON полей для параметров
    - Потокобезопасные операции
    """

    def __init__(self, db_path: str = "results.db"):
        """
        Инициализация менеджера базы данных.

        Args:
            db_path: Путь к файлу SQLite базы данных
        """
        self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """
        Контекстный менеджер для получения соединения с БД.
        Автоматически коммитит изменения при успехе.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """Инициализация схемы базы данных."""
        with self._get_connection() as conn:
            # Основная таблица результатов
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article TEXT NOT NULL,
                    name TEXT NOT NULL,
                    guid TEXT NOT NULL,
                    prompt_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    display_name TEXT,
                    params TEXT,  -- JSON array of parameters
                    raw_response TEXT,
                    error_message TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_used TEXT,
                    api_source TEXT,
                    -- Уникальный индекс для UPSERT семантики
                    UNIQUE(article, prompt_id) ON CONFLICT REPLACE
                )
            """)

            # Индексы для быстрого поиска
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_article_prompt 
                ON results(article, prompt_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category 
                ON results(category)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON results(status)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_guid 
                ON results(guid)
            """)

            logger.info(f"Database initialized: {self.db_path}")

    def upsert_result(self, result: Dict[str, Any]) -> bool:
        """
        Сохранение или обновление результата обработки.

        UPSERT логика:
        - Если запись с (article, prompt_id) существует - обновляет
        - Если не существует - создает новую

        Args:
            result: Словарь с данными результата обработки

        Returns:
            True при успехе, False при ошибке
        """
        try:
            with self._get_connection() as conn:
                params_json = json.dumps(
                    result.get('params', []), 
                    ensure_ascii=False
                ) if result.get('params') else None

                conn.execute("""
                    INSERT INTO results 
                    (article, name, guid, prompt_id, category, status, 
                     display_name, params, raw_response, error_message,
                     processed_at, model_used, api_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(article, prompt_id) DO UPDATE SET
                        name = excluded.name,
                        guid = excluded.guid,
                        category = excluded.category,
                        status = excluded.status,
                        display_name = excluded.display_name,
                        params = excluded.params,
                        raw_response = excluded.raw_response,
                        error_message = excluded.error_message,
                        processed_at = excluded.processed_at,
                        model_used = excluded.model_used,
                        api_source = excluded.api_source
                """, (
                    result.get('article'),
                    result.get('name'),
                    result.get('guid'),
                    result.get('prompt_id'),
                    result.get('category'),
                    result.get('status'),
                    result.get('display_name'),
                    params_json,
                    result.get('raw_response'),
                    result.get('error_message'),
                    result.get('processed_at', datetime.utcnow().isoformat()),
                    result.get('model_used'),
                    result.get('api_source')
                ))
            return True
        except Exception as e:
            logger.error(f"Database upsert error for {result.get('article')}: {e}")
            return False

    def get_result(self, article: str, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение результата по артикулу и ID промпта.

        Args:
            article: Артикул изделия
            prompt_id: Идентификатор промпта

        Returns:
            Словарь с данными результата или None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT * FROM results 
                   WHERE article = ? AND prompt_id = ?""",
                (article, prompt_id)
            ).fetchone()

            if row:
                return self._row_to_dict(row)
        return None

    def get_results_by_article(self, article: str) -> List[Dict[str, Any]]:
        """
        Получение всех результатов для конкретного артикула.

        Args:
            article: Артикул изделия

        Returns:
            Список результатов обработки разными промптами
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM results WHERE article = ?",
                (article,)
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_all_results(
        self, 
        category: Optional[str] = None,
        status: Optional[str] = None,
        prompt_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение результатов с фильтрацией.

        Args:
            category: Фильтр по категории
            status: Фильтр по статусу (completed, ignored, error)
            prompt_id: Фильтр по промпту
            limit: Ограничение количества записей

        Returns:
            Список результатов
        """
        query = "SELECT * FROM results WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)
        if status:
            query += " AND status = ?"
            params.append(status)
        if prompt_id:
            query += " AND prompt_id = ?"
            params.append(prompt_id)

        query += " ORDER BY processed_at DESC"

        if limit:
            query += f" LIMIT {int(limit)}"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Конвертация строки БД в словарь."""
        result = dict(row)
        if result.get('params'):
            try:
                result['params'] = json.loads(result['params'])
            except json.JSONDecodeError:
                result['params'] = []
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по обработке.

        Returns:
            Словарь с агрегированной статистикой
        """
        with self._get_connection() as conn:
            stats = {
                'total': conn.execute("SELECT COUNT(*) FROM results").fetchone()[0],
                'by_status': {},
                'by_category': {},
                'by_prompt': {},
                'by_api': {}
            }

            # По статусам
            for row in conn.execute(
                "SELECT status, COUNT(*) FROM results GROUP BY status"
            ):
                stats['by_status'][row[0]] = row[1]

            # По категориям
            for row in conn.execute(
                "SELECT category, COUNT(*) FROM results GROUP BY category"
            ):
                stats['by_category'][row[0]] = row[1]

            # По промптам
            for row in conn.execute(
                "SELECT prompt_id, COUNT(*) FROM results GROUP BY prompt_id"
            ):
                stats['by_prompt'][row[0]] = row[1]

            # По API источникам
            for row in conn.execute(
                "SELECT api_source, COUNT(*) FROM results GROUP BY api_source"
            ):
                if row[0]:
                    stats['by_api'][row[0]] = row[1]

            return stats

    def delete_result(self, article: str, prompt_id: str) -> bool:
        """
        Удаление конкретного результата.

        Args:
            article: Артикул изделия
            prompt_id: Идентификатор промпта

        Returns:
            True если запись была удалена
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM results WHERE article = ? AND prompt_id = ?",
                (article, prompt_id)
            )
            return cursor.rowcount > 0

    def clear_all(self, confirm: bool = False) -> int:
        """
        Очистка всех результатов (опасная операция).

        Args:
            confirm: Подтверждение очистки

        Returns:
            Количество удаленных записей
        """
        if not confirm:
            logger.warning("Clear all cancelled - confirmation required")
            return 0

        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM results")
            count = cursor.rowcount
            logger.info(f"Cleared {count} records from database")
            return count

    def export_to_json(
        self, 
        output_path: str, 
        structure: str = "flat"
    ) -> str:
        """
        Экспорт результатов в JSON файл.

        Args:
            output_path: Путь для сохранения файла
            structure: Формат структуры ('flat' или 'by_code')

        Returns:
            Путь к созданному файлу
        """
        results = self.get_all_results()

        if structure == "by_code":
            # Группировка по артикулам
            data = {}
            for r in results:
                article = r['article']
                if article not in data:
                    data[article] = {
                        "article": article,
                        "name": r['name'],
                        "guid": r['guid'],
                        "prompts": {}
                    }
                data[article]["prompts"][r['prompt_id']] = {
                    "status": r['status'],
                    "category": r['category'],
                    "display_name": r['display_name'],
                    "params": r.get('params', []),
                    "processed_at": r['processed_at'],
                    "model_used": r.get('model_used'),
                    "api_source": r.get('api_source')
                }
        else:
            # Плоский список
            data = results

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(results)} records to {output_path}")
        return str(output_path)

    def get_pending_items(
        self, 
        all_articles: List[str], 
        prompt_id: str
    ) -> List[str]:
        """
        Получение списка артикулов, которые еще не обработаны конкретным промптом.

        Args:
            all_articles: Полный список артикулов для обработки
            prompt_id: ID промпта

        Returns:
            Список артикулов без готовых результатов
        """
        with self._get_connection() as conn:
            # Получаем уже обработанные
            placeholders = ','.join(['?' for _ in all_articles])
            query = f"""
                SELECT article FROM results 
                WHERE prompt_id = ? 
                AND article IN ({placeholders})
                AND status IN ('completed', 'ignored')
            """
            processed = set(
                row[0] for row in conn.execute(query, [prompt_id] + all_articles)
            )

            # Возвращаем не обработанные
            return [a for a in all_articles if a not in processed]
