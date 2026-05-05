"""
Модуль анализа качества распознавания параметрических данных.

Группирует результаты по паре (item_type + standard) и выводит статистику:
- Общее количество строк
- Количество с определенным ens_code
- Количество с распознанными params
- Количество с распознанными ens_params
- Проценты распознавания
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.automated_processor import AutomatedParametricProcessor
from utils.excel_loader import ExcelLoader

logger = logging.getLogger(__name__)


@dataclass
class QualityStats:
    """Статистика по группе (item_type + standard)."""
    item_type: str = ""
    standard: str = ""
    total: int = 0
    ens_code_found: int = 0          # ens_code определен
    ens_code_percent: float = 0.0
    params_parsed: int = 0           # params распознаны (парсинг text)
    params_percent: float = 0.0
    ens_params_parsed: int = 0       # ens_params распознаны (из модели ENS)
    ens_params_percent: float = 0.0
    both_params_parsed: int = 0      # и params, и ens_params
    both_params_percent: float = 0.0

    def recalculate(self):
        """Пересчитать проценты."""
        if self.total > 0:
            self.ens_code_percent = round(self.ens_code_found / self.total * 100, 2)
            self.params_percent = round(self.params_parsed / self.total * 100, 2)
            self.ens_params_percent = round(self.ens_params_parsed / self.total * 100, 2)
            self.both_params_percent = round(self.both_params_parsed / self.total * 100, 2)


class QualityAnalyzer:
    """Анализатор качества распознавания параметрических данных."""

    def __init__(self, processor: AutomatedParametricProcessor):
        self.processor = processor
        self.loader = ExcelLoader()
        self._last_detail_results: List[Dict[str, Any]] = []

    def analyze_file(self, excel_path: str) -> Dict[str, QualityStats]:
        """
        Анализ Excel-файла: собрать статистику по (item_type, standard).

        Args:
            excel_path: Путь к Excel-файлу

        Returns:
            Словарь: ключ "item_type|standard" → QualityStats
        """
        logger.info(f"[ANALYZE] Загрузка файла: {excel_path}")
        texts = self.loader.load(excel_path)
        logger.info(f"[ANALYZE] Загружено {len(texts)} строк")

        # Группировка статистики
        stats_by_group: Dict[str, QualityStats] = {}
        # Детальные результаты для JSON
        detail_results: List[Dict[str, Any]] = []

        start_time = time.time()

        for idx, text in enumerate(texts):
            try:
                result = self.processor.process(text)

                # Ключ группировки
                item_type = result.item_type if result.item_type else "(не определен)"
                standard = result.standard if result.standard else "(не определен)"
                group_key = f"{item_type}|{standard}"

                if group_key not in stats_by_group:
                    stats_by_group[group_key] = QualityStats(
                        item_type=item_type,
                        standard=standard
                    )

                stats = stats_by_group[group_key]
                stats.total += 1

                # ens_code определен?
                if result.ens_code:
                    stats.ens_code_found += 1

                # params распознаны (парсинг text)?
                has_params = bool(result.params)
                if has_params:
                    stats.params_parsed += 1

                # ens_params распознаны (из модели ENS)?
                has_ens_params = bool(result.ens_params)
                if has_ens_params:
                    stats.ens_params_parsed += 1

                # И то, и другое
                if has_params and has_ens_params:
                    stats.both_params_parsed += 1

                # Детали
                detail_results.append({
                    "text": text,
                    "item_type": item_type,
                    "standard": standard,
                    "ens_code": result.ens_code,
                    "has_params": has_params,
                    "has_ens_params": has_ens_params,
                    "params": result.params,
                    "ens_params": result.ens_params,
                    "level": result.level,
                    "success": result.success
                })

            except Exception as e:
                logger.warning(f"[ANALYZE] Ошибка обработки строки {idx}: {e}")
                group_key = "(ошибка)|(ошибка)"
                if group_key not in stats_by_group:
                    stats_by_group[group_key] = QualityStats(
                        item_type="(ошибка)",
                        standard="(ошибка)"
                    )
                stats_by_group[group_key].total += 1

        # Пересчитать проценты
        for stats in stats_by_group.values():
            stats.recalculate()

        elapsed = time.time() - start_time
        logger.info(f"[ANALYZE] Анализ завершен за {elapsed:.1f} сек, "
                    f"{len(stats_by_group)} групп")

        self._last_detail_results = detail_results
        return stats_by_group

    def format_report_json(self, stats: Dict[str, QualityStats]) -> dict:
        """
        Форматировать отчет в виде JSON-структуры.

        Args:
            stats: Результат analyze_file()

        Returns:
            dict с полной статистикой
        """
        sorted_items = sorted(stats.items(), key=lambda x: (x[1].item_type, x[1].standard))

        total_all = sum(s.total for s in stats.values())
        total_ens_code = sum(s.ens_code_found for s in stats.values())
        total_params = sum(s.params_parsed for s in stats.values())
        total_ens_params = sum(s.ens_params_parsed for s in stats.values())
        total_both = sum(s.both_params_parsed for s in stats.values())

        groups = []
        for key, s in sorted_items:
            groups.append({
                "item_type": s.item_type,
                "standard": s.standard,
                "total": s.total,
                "ens_code": {
                    "found": s.ens_code_found,
                    "percent": s.ens_code_percent
                },
                "params": {
                    "found": s.params_parsed,
                    "percent": s.params_percent
                },
                "ens_params": {
                    "found": s.ens_params_parsed,
                    "percent": s.ens_params_percent
                },
                "both": {
                    "found": s.both_params_parsed,
                    "percent": s.both_params_percent
                }
            })

        report = {
            "summary": {
                "total": total_all,
                "ens_code": {
                    "found": total_ens_code,
                    "percent": round(total_ens_code / total_all * 100, 2) if total_all else 0
                },
                "params": {
                    "found": total_params,
                    "percent": round(total_params / total_all * 100, 2) if total_all else 0
                },
                "ens_params": {
                    "found": total_ens_params,
                    "percent": round(total_ens_params / total_all * 100, 2) if total_all else 0
                },
                "both": {
                    "found": total_both,
                    "percent": round(total_both / total_all * 100, 2) if total_all else 0
                }
            },
            "groups": groups
        }

        return report

    def save_json(self, stats: Dict[str, QualityStats], output_path: str):
        """Сохранить статистику и детали в JSON."""
        data = {
            "summary": {
                k: {f: getattr(v, f) for f in v.__dataclass_fields__}
                for k, v in stats.items()
            },
            "details": getattr(self, '_last_detail_results', [])
        }
        Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                     encoding='utf-8')
        logger.info(f"[ANALYZE] JSON сохранен: {output_path}")


def analyze_quality(
    excel_path: str,
    mask_db_path: str = "cache/masks.db",
    ens_index_path: str = "models/hardware/ens_hardware.pkl",
    output_json: Optional[str] = None,
    ocr_service_url: Optional[str] = None,
    max_workers: int = 4
) -> dict:
    """
    Точка входа для CLI: анализ качества распознавания.

    Returns:
        JSON-структура с отчетом
    """
    analyzer = QualityAnalyzer(
        mask_db_path=mask_db_path,
        ens_index_path=ens_index_path,
        ocr_service_url=ocr_service_url,
        max_workers=max_workers
    )

    stats = analyzer.analyze_file(excel_path)
    report = analyzer.format_report_json(stats)

    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"[ANALYZE] JSON отчет сохранен: {output_json}")

    return report