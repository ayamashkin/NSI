"""
JSON Export Module
Экспорт результатов обработки в различные JSON форматы.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class JSONExporter:
    """
    Экспортер результатов в JSON формат.

    Поддерживаемые форматы:
    - flat: Плоский список всех результатов
    - by_code: Группировка по артикулам
    - by_category: Группировка по категориям
    - by_prompt: Группировка по промптам
    """

    def __init__(self, results: List[Dict[str, Any]]):
        """
        Инициализация экспортера.

        Args:
            results: Список результатов обработки
        """
        self.results = results

    def export(
        self, 
        output_path: str, 
        structure: str = "flat",
        include_raw: bool = False
    ) -> str:
        """
        Экспорт результатов в JSON файл.

        Args:
            output_path: Путь для сохранения
            structure: Формат структуры (flat, by_code, by_category, by_prompt)
            include_raw: Включать ли raw_response в вывод

        Returns:
            Путь к созданному файлу
        """
        # Фильтрация полей
        data = self._prepare_data(structure, include_raw)

        # Сохранение
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Exported {len(self.results)} records to {output_path}")
        return str(output_path)

    def _prepare_data(
        self, 
        structure: str, 
        include_raw: bool
    ) -> Any:
        """Подготовка данных в нужной структуре."""

        # Очистка результатов
        cleaned = []
        for r in self.results:
            item = {
                'article': r.get('article'),
                'name': r.get('name'),
                'guid': r.get('guid'),
                'prompt_id': r.get('prompt_id'),
                'category': r.get('category'),
                'status': r.get('status'),
                'display_name': r.get('display_name'),
                'params': r.get('params', []),
                'processed_at': r.get('processed_at'),
                'model_used': r.get('model_used'),
                'api_source': r.get('api_source')
            }

            if include_raw:
                item['raw_response'] = r.get('raw_response')
                item['error_message'] = r.get('error_message')

            cleaned.append(item)

        # Форматирование по структуре
        if structure == "by_code":
            return self._group_by_code(cleaned)
        elif structure == "by_category":
            return self._group_by_category(cleaned)
        elif structure == "by_prompt":
            return self._group_by_prompt(cleaned)
        else:  # flat
            return cleaned

    def _group_by_code(self, items: List[Dict]) -> Dict[str, Any]:
        """Группировка по артикулам."""
        result = {}
        for item in items:
            article = item['article']
            if article not in result:
                result[article] = {
                    'article': article,
                    'name': item['name'],
                    'guid': item['guid'],
                    'prompts': {}
                }
            result[article]['prompts'][item['prompt_id']] = {
                'status': item['status'],
                'category': item['category'],
                'display_name': item['display_name'],
                'params': item['params'],
                'processed_at': item['processed_at']
            }
        return result

    def _group_by_category(self, items: List[Dict]) -> Dict[str, Any]:
        """Группировка по категориям."""
        result = {}
        for item in items:
            category = item['category']
            if category not in result:
                result[category] = []
            result[category].append(item)
        return result

    def _group_by_prompt(self, items: List[Dict]) -> Dict[str, Any]:
        """Группировка по промптам."""
        result = {}
        for item in items:
            prompt_id = item['prompt_id']
            if prompt_id not in result:
                result[prompt_id] = []
            result[prompt_id].append(item)
        return result

    def export_summary(self, output_path: str) -> str:
        """
        Экспорт сводной статистики.

        Args:
            output_path: Путь для сохранения

        Returns:
            Путь к созданному файлу
        """
        stats = {
            'export_date': datetime.utcnow().isoformat(),
            'total_records': len(self.results),
            'by_status': {},
            'by_category': {},
            'by_prompt': {}
        }

        for item in self.results:
            # По статусам
            status = item.get('status', 'unknown')
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

            # По категориям
            category = item.get('category', 'unknown')
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1

            # По промптам
            prompt_id = item.get('prompt_id', 'unknown')
            stats['by_prompt'][prompt_id] = stats['by_prompt'].get(prompt_id, 0) + 1

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported summary to {output_path}")
        return str(output_path)


def export_results(
    results: List[Dict[str, Any]],
    output_path: str,
    structure: str = "flat"
) -> str:
    """
    Упрощенная функция экспорта.

    Args:
        results: Список результатов
        output_path: Путь для сохранения
        structure: Формат структуры

    Returns:
        Путь к созданному файлу
    """
    exporter = JSONExporter(results)
    return exporter.export(output_path, structure)
