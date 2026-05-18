"""
Модуль анализа качества распознавания параметрических данных.

Группирует результаты по паре (item_type + standard) и выводит статистику.

VERSION: 2025-05-06-fix7 (double-dollar-fix)
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.automated_processor import AutomatedParametricProcessor

logger = logging.getLogger(__name__)

@dataclass
class QualityStats:
    """Статистика по группе (item_type + standard)."""
    item_type: str = ""
    standard: str = ""
    total: int = 0
    ens_code_found: int = 0  # ens_code определен
    ens_code_percent: float = 0.0
    params_parsed: int = 0  # params распознаны (парсинг text)
    params_percent: float = 0.0
    ens_params_parsed: int = 0  # ens_params распознаны (из модели ENS)
    ens_params_percent: float = 0.0
    both_params_parsed: int = 0  # и params, и ens_params
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

        import pandas as pd
        df = pd.read_excel(excel_path)

        # FIX: поддержка разных вариантов названий колонки
        name_col = None
        for candidate in ['Наименование', 'Краткое наименование', 'Full name', 'Name']:
            if candidate in df.columns:
                name_col = candidate
                break

        if name_col is None:
            name_cols = [c for c in df.columns if 'наимен' in str(c).lower() or 'name' in str(c).lower()]
            if name_cols:
                name_col = name_cols[0]
            else:
                raise ValueError("Колонка с наименованием не найдена")

        texts = df[name_col].astype(str).tolist()
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
                if result.ens_match and result.ens_match.get('code'):
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
                    "ens_code": result.ens_match.get('code') if result.ens_match else None,
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

    def format_report(self, stats: Dict[str, QualityStats]) -> str:
        """
        Форматировать отчет в виде текстовой таблицы.

        Args:
            stats: Результат analyze_file()

        Returns:
            Строка с таблицей
        """
        lines = []
        lines.append("=" * 120)
        lines.append("АНАЛИЗ КАЧЕСТВА РАСПОЗНАВАНИЯ")
        lines.append("=" * 120)

        # Сортировка: сначала по item_type, затем по standard
        sorted_items = sorted(stats.items(), key=lambda x: (x[1].item_type, x[1].standard))

        # Заголовок таблицы
        header = (
            f"{'Тип':<12} {'Стандарт':<30} {'Всего':>6} "
            f"{'ENS код':>8} {'%':>6} "
            f"{'Params':>8} {'%':>6} "
            f"{'ENS params':>10} {'%':>6} "
            f"{'Оба':>8} {'%':>6}"
        )
        lines.append(header)
        lines.append("-" * 120)

        # Итоги
        total_all = 0
        total_ens_code = 0
        total_params = 0
        total_ens_params = 0
        total_both = 0

        for key, s in sorted_items:
            line = (
                f"{s.item_type:<12} {s.standard:<30} {s.total:>6} "
                f"{s.ens_code_found:>8} {s.ens_code_percent:>5.1f}% "
                f"{s.params_parsed:>8} {s.params_percent:>5.1f}% "
                f"{s.ens_params_parsed:>10} {s.ens_params_percent:>5.1f}% "
                f"{s.both_params_parsed:>8} {s.both_params_percent:>5.1f}%"
            )
            lines.append(line)

            total_all += s.total
            total_ens_code += s.ens_code_found
            total_params += s.params_parsed
            total_ens_params += s.ens_params_parsed
            total_both += s.both_params_parsed

        # Итоговая строка
        lines.append("-" * 120)
        if total_all > 0:
            total_line = (
                f"{'ИТОГО':<12} {'':<30} {total_all:>6} "
                f"{total_ens_code:>8} {total_ens_code/total_all*100:>5.1f}% "
                f"{total_params:>8} {total_params/total_all*100:>5.1f}% "
                f"{total_ens_params:>10} {total_ens_params/total_all*100:>5.1f}% "
                f"{total_both:>8} {total_both/total_all*100:>5.1f}%"
            )
            lines.append(total_line)

        lines.append("=" * 120)
        lines.append("")
        lines.append("Легенда:")
        lines.append(" ENS код — определен ens_code (по TF-IDF или маске)")
        lines.append(" Params — распознаны params из текста (парсинг)")
        lines.append(" ENS params — распознаны ens_params из модели ENS")
        lines.append(" Оба — и params, и ens_params распознаны")

        return "\n".join(lines)

    def save_excel(self, stats: Dict[str, QualityStats], output_path: str):
        """Сохранить отчет в Excel."""
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas не установлен — Excel не сохранен")
            return

        rows = []
        for key, s in sorted(stats.items(), key=lambda x: (x[1].item_type, x[1].standard)):
            rows.append({
                "Тип": s.item_type,
                "Стандарт": s.standard,
                "Всего": s.total,
                "ENS код (шт)": s.ens_code_found,
                "ENS код (%)": s.ens_code_percent,
                "Params (шт)": s.params_parsed,
                "Params (%)": s.params_percent,
                "ENS params (шт)": s.ens_params_parsed,
                "ENS params (%)": s.ens_params_percent,
                "Оба (шт)": s.both_params_parsed,
                "Оба (%)": s.both_params_percent,
            })

        df = pd.DataFrame(rows)

        # Итоговая строка
        total_all = sum(s.total for s in stats.values())
        total_ens = sum(s.ens_code_found for s in stats.values())
        total_par = sum(s.params_parsed for s in stats.values())
        total_ens_par = sum(s.ens_params_parsed for s in stats.values())
        total_both = sum(s.both_params_parsed for s in stats.values())

        if total_all > 0:
            totals = pd.DataFrame([{
                "Тип": "ИТОГО",
                "Стандарт": "",
                "Всего": total_all,
                "ENS код (шт)": total_ens,
                "ENS код (%)": round(total_ens / total_all * 100, 2),
                "Params (шт)": total_par,
                "Params (%)": round(total_par / total_all * 100, 2),
                "ENS params (шт)": total_ens_par,
                "ENS params (%)": round(total_ens_par / total_all * 100, 2),
                "Оба (шт)": total_both,
                "Оба (%)": round(total_both / total_all * 100, 2),
            }])
            df = pd.concat([df, totals], ignore_index=True)

        df.to_excel(output_path, index=False, engine='openpyxl')
        logger.info(f"[ANALYZE] Excel сохранен: {output_path}")

    def save_json(self, stats: Dict[str, QualityStats], output_path: str):
        """Сохранить статистику и детали в JSON."""
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

        data = {
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
            "groups": groups,
            "details": self._last_detail_results
        }

        def _json_default(obj):
            """JSON serializer for non-standard types."""
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
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
