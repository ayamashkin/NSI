# =============================================================================
# ФАЙЛ: core/batch_service.py
# ПОСЛЕДНИЕ 5 ИЗМЕНЕНИЙ (МСК, UTC+3), от новых к старым:
# 2026-06-04 15:00:00 — FEAT: BatchService — вынос логики batch processing из cli.py.
#   Чистые функции для web-интерфейса: process_batch, process_excel, results_to_excel_rows.
# =============================================================================

"""
Batch Processing Service — чистые функции для пакетной обработки.
Выделено из cli.py для использования в web-интерфейсе.
"""

import logging
import os
import sys
import threading
import json
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _find_name_column(df: pd.DataFrame) -> Optional[str]:
    """Поиск колонки с наименованием."""
    keywords = ['наименование', 'номенклатура', 'name', 'наименов', 'наим.', 'краткое наименование']
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def _format_params_cell(params_dict: dict, max_items: int = 20) -> str:
    """Форматировать dict параметров в многострочный текст."""
    if not params_dict:
        return ''
    # FIX 2026-06-04: защита от list (ens_params_mask иногда list, не dict)
    if isinstance(params_dict, list):
        lines = [str(item) for item in params_dict if item is not None]
        return '\n'.join(lines[:max_items]) if lines else ''
    if not isinstance(params_dict, dict):
        return str(params_dict)[:1000]
    lines = []
    for k, v in params_dict.items():
        if v is not None:
            lines.append(f"{k}={v}")
            if len(lines) >= max_items:
                lines.append(f"... ({len(params_dict)} всего)")
                break
    return '\n'.join(lines)


def _format_top_candidates(details_dict: dict) -> str:
    """Форматировать top-5 кандидатов из details в читаемый текст."""
    if not details_dict:
        return ''
    top = details_dict.get('top_candidates', []) or details_dict.get('debug_candidates', [])
    if not top:
        return ''
    lines = []
    for i, cd in enumerate(top, 1):
        name = cd.get('name', 'N/A')[:40]
        code = cd.get('ens_code', 'N/A')
        score = cd.get('score', 0)
        lines.append(f"{i}. [{code}] {name} (score={score})")
        comp = cd.get('params_comparison', {})
        for pk, pv in comp.items():
            status = pv.get('status', '?') if isinstance(pv, dict) else str(pv)
            extracted = pv.get('extracted', '?') if isinstance(pv, dict) else '?'
            ens_val = pv.get('ens_value', '?') if isinstance(pv, dict) else '?'
            if status == 'exact':
                lines.append(f"   {pk}: {extracted}={ens_val}")
            elif status == 'exact (in name)':
                lines.append(f"   {pk}: {extracted}~{ens_val} (in name)")
            elif 'token' in status:
                lines.append(f"   {pk}: {extracted}~{ens_val}")
            else:
                lines.append(f"   {pk}: {extracted}!={ens_val}")
    return '\n'.join(lines)


class BatchService:
    """Сервис пакетной обработки номенклатуры."""

    def __init__(
        self,
        db_path: str = 'cache/masks.db',
        ens_index_path: Optional[str] = None,
        result_db_path: str = 'cache/result.db',
        domain: str = 'hardware',
        workers: int = 4,
        use_llm: bool = False,
        no_cache: bool = False,
        include_details: bool = True,
    ):
        self.db_path = db_path
        self.ens_index_path = ens_index_path
        self.result_db_path = result_db_path
        self.domain = domain
        self.workers = workers
        self.use_llm = use_llm
        self.no_cache = no_cache
        self.include_details = include_details
        self._processor = None
        self._mask_db = None
        self._init_processor()

    def _init_processor(self):
        """Инициализация процессора."""
        from core.mask_database import MaskDatabase
        from core.automated_processor import AutomatedParametricProcessor
        from core.settings import get_settings
        from core.domain_config import DomainConfig

        # Автоопределение пути к индексу из доменного конфига
        ens_index = self.ens_index_path
        if not ens_index and self.domain:
            try:
                cfg = DomainConfig.load(self.domain)
                if cfg.index_path:
                    ens_index = cfg.index_path
                    logger.info(f"📂 Индекс из домена '{self.domain}': {ens_index}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить домен '{self.domain}': {e}")

        if not ens_index:
            raise ValueError(
                f"Не указан путь к ENS-индексу. "
                f"Укажите ens_index_path или настройте домен '{self.domain}'"
            )

        self.ens_index_path = ens_index

        settings = get_settings()
        llm_clients = {}
        if self.use_llm:
            from utils.llm_utils import init_llm_clients
            llm_clients = init_llm_clients(settings, all_services=False)
            if llm_clients:
                logger.info("🤖 LLM клиенты инициализированы")

        self._mask_db = MaskDatabase(db_path=self.db_path)
        self._processor = AutomatedParametricProcessor(
            mask_db=self._mask_db,
            llm_clients=llm_clients if self.use_llm else None,
            ens_index_path=self.ens_index_path,
            use_llm_generation=self.use_llm,
            settings=settings,
            result_db_path=self.result_db_path,
            no_cache=self.no_cache,
            domain=self.domain,
        )
        logger.info("[BATCH_SERVICE] Процессор инициализирован (domain=%s, workers=%d)",
                     self.domain, self.workers)

    def process_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int, dict], None]] = None,
    ) -> Tuple[List[Any], Dict[str, int]]:
        """
        Пакетная обработка списка текстов.

        Args:
            texts: список наименований
            progress_callback: функция (current, total, stats) для прогресса

        Returns:
            (results, stats) — список ProcessingResult и статистика
        """
        results = [None] * len(texts)
        stats = {'total': 0, 'success': 0, 'failed': 0}
        stats_lock = threading.Lock()
        processed = [0]
        processed_lock = threading.Lock()

        def _process_one(idx_text):
            idx, text = idx_text
            try:
                result = self._processor.process(text)
                with stats_lock:
                    stats['total'] += 1
                    if result.success:
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                # Сохраняем в result.db
                if self.result_db_path:
                    try:
                        self._save_result(result)
                    except Exception as e:
                        logger.warning("[BATCH_SERVICE] Failed to save result: %s", e)
                # Прогресс
                with processed_lock:
                    processed[0] += 1
                    if progress_callback:
                        progress_callback(processed[0], len(texts), dict(stats))
                return idx, result
            except Exception as e:
                logger.error("[BATCH_SERVICE] Error processing item %d: %s", idx, e)
                with stats_lock:
                    stats['total'] += 1
                    stats['failed'] += 1
                with processed_lock:
                    processed[0] += 1
                    if progress_callback:
                        progress_callback(processed[0], len(texts), dict(stats))
                return idx, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(_process_one, (i, t)): i for i, t in enumerate(texts)}
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                if result is not None:
                    results[idx] = result

        valid_results = [r for r in results if r is not None]
        logger.info("[BATCH_SERVICE] Обработка завершена: %s", stats)
        return valid_results, stats

    def _save_result(self, result):
        """Сохранить результат в result.db."""
        from core.result_database import ResultDatabaseManager
        manager = ResultDatabaseManager(db_path=self.result_db_path)
        changed, reason = manager.upsert_result(
            name=result.text,
            article=None,
            item_type=result.item_type,
            standard=result.standard,
            ens_code=result.ens_code,
            ens_name=result.ens_name,
            success=result.success,
            confidence=result.confidence,
            params=result.params,
            ens_params=result.ens_params,
            ens_params_mask=result.ens_params_mask,
            match_type=result.match_type,
            match_type_ru=result.match_type_ru,
            coating_substitution=result.coating_substitution,
            fuzzy_mismatched_params=result.fuzzy_mismatched_params,
            mask_id=result.mask_id,
            mask_pattern=result.mask_pattern,
            details=result.details,
            processing_time_ms=result.processing_time_ms,
        )
        logger.debug("[BATCH_SERVICE] Saved: changed=%s reason=%s", changed, reason)

    def process_excel(
        self,
        file_path: str,
        progress_callback: Optional[Callable[[int, int, dict], None]] = None,
    ) -> Tuple[List[Any], Dict[str, int], pd.DataFrame, str]:
        """
        Обработка Excel-файла.

        Returns:
            (results, stats, original_df, name_col)
        """
        df = pd.read_excel(file_path)
        name_col = _find_name_column(df)
        if name_col is None:
            raise ValueError(
                f"В файле отсутствует колонка с наименованием. "
                f"Доступные колонки: {list(df.columns)}"
            )
        texts = df[name_col].astype(str).tolist()
        logger.info("[BATCH_SERVICE] Excel загружен: %d строк, колонка '%s'",
                     len(texts), name_col)
        results, stats = self.process_batch(texts, progress_callback)
        return results, stats, df, name_col

    def results_to_excel_rows(
        self,
        results: List[Any],
        original_df: pd.DataFrame,
        name_col: str,
    ) -> List[Dict]:
        """Преобразовать результаты в строки для Excel-экспорта."""
        excel_rows = []
        for idx, result in enumerate(results):
            out_row = {}
            # Копируем оригинальные колонки
            for col in original_df.columns:
                val = original_df.iloc[idx][col]
                if pd.isna(val):
                    out_row[str(col)] = None
                else:
                    out_row[str(col)] = val
            # Добавляем результаты обработки
            out_row['Код ЕНС'] = str(result.ens_code)[:50] if result.ens_code else ''
            out_row['Наименование ЕНС'] = str(result.ens_name)[:500] if result.ens_name else ''
            out_row['Уровень'] = str(result.level.value if hasattr(result.level, 'value') else result.level) if result.level else ''
            out_row['Распознано'] = 'Да' if result.success else 'Нет'
            out_row['Уверенность'] = round(float(result.confidence or 0.0), 3)
            out_row['Тип сопоставления'] = str(result.match_type_ru) if result.match_type_ru else 'Не определено'
            sub = result.coating_substitution
            if sub:
                clean_sub = {
                    'original': sub.get('original'),
                    'corrected': sub.get('corrected'),
                    'material': sub.get('material'),
                    'reason': sub.get('reason'),
                }
                out_row['Подстановка покрытия'] = json.dumps(clean_sub, ensure_ascii=False)
            else:
                out_row['Подстановка покрытия'] = None
            mism = result.fuzzy_mismatched_params
            out_row['Несовпавшие параметры'] = json.dumps(mism, ensure_ascii=False) if mism else None
            out_row['маска'] = str(result.mask_pattern)[:1000] if result.mask_pattern else ''
            out_row['params'] = _format_params_cell(result.params)
            out_row['ens_params'] = _format_params_cell(result.ens_params)
            out_row['ens_params_mask'] = _format_params_cell(result.ens_params_mask)
            out_row['стандарт'] = str(result.standard) if result.standard else ''
            out_row['тип'] = str(result.item_type) if result.item_type else ''
            has_mask = False
            if result.standard and result.item_type:
                try:
                    m = self._mask_db.get_mask(result.standard, result.item_type.upper())
                    if m is None:
                        m = self._mask_db.get_mask(result.standard, result.item_type)
                    has_mask = m is not None
                except Exception:
                    pass
            out_row['маски_в_бд'] = 'Да' if has_mask else 'Нет'
            if self.include_details and result.details:
                if result.confidence is not None and result.confidence < 1.0:
                    top_text = _format_top_candidates(result.details)
                    if top_text:
                        out_row['детали'] = top_text
                    else:
                        out_row['детали'] = json.dumps(result.details, ensure_ascii=False, default=str)
                else:
                    out_row['детали'] = json.dumps(result.details, ensure_ascii=False, default=str)
            excel_rows.append(out_row)
        return excel_rows

    @staticmethod
    def get_available_domains() -> List[str]:
        """Получить список доступных доменов."""
        domains = []
        config_dir = Path('config')
        if config_dir.exists():
            for f in config_dir.iterdir():
                if f.suffix in ('.yaml', '.yml') and f.name != 'config.yaml':
                    domains.append(f.stem)
        return domains if domains else ['hardware']
