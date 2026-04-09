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
            # Проверяем, существует ли таблица и нужна ли миграция
            cursor = conn.execute("PRAGMA table_info(results)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            # Если таблица не существует — создаем с нуля
            if not existing_columns:
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
                        params TEXT,
                        raw_response TEXT,
                        error_message TEXT,
                        prompt_text TEXT,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_used TEXT,
                        api_source TEXT,
                        UNIQUE(article, prompt_id) ON CONFLICT REPLACE
                    )
                """)
            else:
                # Миграция: добавляем колонку prompt_text если её нет
                if 'prompt_text' not in existing_columns:
                    conn.execute("ALTER TABLE results ADD COLUMN prompt_text TEXT")
                    logger.info("Migrated database: added prompt_text column")

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
                     display_name, params, raw_response, error_message, prompt_text,
                     processed_at, model_used, api_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(article, prompt_id) DO UPDATE SET
                        name = excluded.name,
                        guid = excluded.guid,
                        category = excluded.category,
                        status = excluded.status,
                        display_name = excluded.display_name,
                        params = excluded.params,
                        raw_response = excluded.raw_response,
                        error_message = excluded.error_message,
                        prompt_text = excluded.prompt_text,
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
                    result.get('prompt_text'),
                    result.get('processed_at', datetime.utcnow().isoformat()),
                    result.get('model_used'),
                    result.get('api_source')
                ))
            return True
        except Exception as e:
            logger.error(f"Database upsert error for {result.get('article')}: {e}")
            return False

    def get_result(self, article: str, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Получение результата по артикулу и ID промпта."""
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
        """Получение всех результатов для конкретного артикула."""
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
        """Получение результатов с фильтрацией."""
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

        # Парсим JSON-поля
        for field in ['params']:
            if result.get(field) and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = []

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по обработке."""
        with self._get_connection() as conn:
            stats = {
                'total': conn.execute("SELECT COUNT(*) FROM results").fetchone()[0],
                'by_status': {},
                'by_category': {},
                'by_prompt': {},
                'by_api': {}
            }

            for row in conn.execute("SELECT status, COUNT(*) FROM results GROUP BY status"):
                stats['by_status'][row[0]] = row[1]

            for row in conn.execute("SELECT category, COUNT(*) FROM results GROUP BY category"):
                stats['by_category'][row[0]] = row[1]

            for row in conn.execute("SELECT prompt_id, COUNT(*) FROM results GROUP BY prompt_id"):
                stats['by_prompt'][row[0]] = row[1]

            for row in conn.execute("SELECT api_source, COUNT(*) FROM results GROUP BY api_source"):
                if row[0]:
                    stats['by_api'][row[0]] = row[1]

            return stats

    def delete_result(self, article: str, prompt_id: str) -> bool:
        """Удаление конкретного результата."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM results WHERE article = ? AND prompt_id = ?",
                (article, prompt_id)
            )
            return cursor.rowcount > 0

    def clear_all(self, confirm: bool = False) -> int:
        """Очистка всех результатов."""
        if not confirm:
            logger.warning("Clear all cancelled - confirmation required")
            return 0

        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM results")
            count = cursor.rowcount
            logger.info(f"Cleared {count} records from database")
            return count

    def export_filtered_to_json(
            self,
            output_path: str,
            results: List[Dict],
            structure: str = 'flat',
            include_raw: bool = False,
            include_prompt: bool = False
    ) -> str:
        """Экспорт отфильтрованных результатов в JSON с десериализацией полей."""
        # Преобразуем JSON-строки в объекты и обрабатываем prompt_text
        serialized_results = [self._serialize_row(r) for r in results]

        # Очищаем результаты согласно флагам
        for r in serialized_results:
            if not include_raw:
                r.pop('raw_response', None)
                r.pop('error_message', None)
            if not include_prompt:
                r.pop('prompt_text', None)

        if structure == 'by_code':
            data = self._structure_by_code(serialized_results)
        elif structure == 'by_category':
            data = self._structure_by_category(serialized_results)
        elif structure == 'by_prompt':
            data = self._structure_by_prompt(serialized_results)
        else:
            data = serialized_results

        # Записываем с правильной кодировкой и форматированием
        # Используем ensure_ascii=False и не экранируем переносы строк
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,  # Не экранировать Unicode (русские буквы)
                indent=2,  # Красивое форматирование
                separators=(',', ': ')  # Пробел после двоеточия для читаемости
            )

        return output_path

    def get_pending_items(self, all_articles: List[str], prompt_id: str) -> List[str]:
        """Получение списка артикулов, которые еще не обработаны."""
        with self._get_connection() as conn:
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
            return [a for a in all_articles if a not in processed]

    def get_filtered_results(self, prompt_id: str = None, status: str = None, limit: int = None) -> List[Dict]:
        """Получение результатов с фильтрацией."""
        query = "SELECT * FROM results WHERE 1=1"
        params = []

        if prompt_id:
            query += " AND prompt_id = ?"
            params.append(prompt_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY processed_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def _structure_by_code(self, results: List[Dict]) -> Dict:
        """Группировка результатов по артикулам."""
        data = {}
        for r in results:
            article = r.get('article', 'unknown')
            if article not in data:
                data[article] = {
                    "article": article,
                    "name": r.get('name', ''),
                    "guid": r.get('guid', ''),
                    "prompts": {}
                }
            data[article]["prompts"][r.get('prompt_id', 'unknown')] = {
                "status": r.get('status'),
                "category": r.get('category'),
                "display_name": r.get('display_name'),
                "params": r.get('params', []),
                "processed_at": r.get('processed_at')
            }
        return data

    def _structure_by_category(self, results: List[Dict]) -> Dict:
        """Группировка результатов по категориям."""
        grouped = {}
        for r in results:
            cat = r.get('category', 'unknown')
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(r)
        return grouped

    def _structure_by_prompt(self, results: List[Dict]) -> Dict:
        """Группировка результатов по промптам."""
        grouped = {}
        for r in results:
            pid = r.get('prompt_id', 'unknown')
            if pid not in grouped:
                grouped[pid] = []
            grouped[pid].append(r)
        return grouped

    def _serialize_row(self, row: Dict) -> Dict:
        """Преобразует строку БД в корректный формат для экспорта."""
        result = dict(row)

        # Поля, которые нужно распарсить из JSON-строк
        json_fields = ['params', 'raw_response', 'usage']

        for field in json_fields:
            if field in result and result[field] is not None:
                if isinstance(result[field], str):
                    try:
                        result[field] = json.loads(result[field])
                    except json.JSONDecodeError:
                        pass

        # Обрабатываем prompt_text - исправляем кодировку и escape-последовательности
        if 'prompt_text' in result and isinstance(result['prompt_text'], str):
            text = result['prompt_text']

            # Шаг 1: Исправляем mojibake (Latin-1 -> UTF-8)
            # Проверяем, есть ли типичные признаки mojibake (кириллица в Latin-1)
            if text and any(ord(c) > 127 for c in text[:100] if c.isalpha()):
                try:
                    text = text.encode('latin-1').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass  # Если не получилось, оставляем как есть

            # Шаг 2: Декодируем escape-последовательности
            # ВАЖНО: заменяем \n (два символа) на реальный перенос строки (один символ)
            # Сначала обрабатываем двойное экранирование
            text = text.replace('\\\\n', '\\x00')  # временный маркер
            text = text.replace('\\\\t', '\\x01')  # временный маркер
            text = text.replace('\\\\r', '\\x02')  # временный маркер

            # Затем одинарное экранирование
            text = text.replace('\\n', '\n')
            text = text.replace('\\t', '\t')
            text = text.replace('\\r', '\r')

            # Возвращаем временные маркеры
            text = text.replace('\\x00', '\\n')
            text = text.replace('\\x01', '\\t')
            text = text.replace('\\x02', '\\r')

            # Убираем экранирование кавычек
            text = text.replace('\\"', '"')
            text = text.replace("\\'", "'")
            text = text.replace('\\\\', '\\')

            result['prompt_text'] = text

        return result