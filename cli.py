# =============================================================================
# Nomenclature Processor CLI
# Параметрический процессор сопоставления номенклатуры с ЕНС (LLM + Parametric modes)
#
# 2026-05-27 17:00:00 — Добавлена команда build-index с доменной архитектурой
# 2026-05-27 14:25:00 — generate-masks, batch, diagnose обновлены с --domain
# 2026-05-27 14:25:00 — Добавлен --auto-domain для multi-domain сопоставления
# 2026-05-27 14:25:00 — ens build-index теперь использует ENSIndexBuilder
# 2026-05-27 14:25:00 — process_parametric и analyze-quality поддерживают --domain
# =============================================================================

import click
import logging
import threading
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from utils.standard_utils import canonicalize_standard
from core.settings import setup_logging
from core.domain_config import DomainConfig
from core.ens_index_builder import ENSIndexBuilder

logger = logging.getLogger(__name__)

def _sanitize_for_json(obj):
    """Рекурсивная очистка объекта от NaN/Infinity для JSON-сериализации."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif pd.isna(obj):
        return None
    else:
        return obj

def _find_name_column(df):
    """Поиск колонки с наименованием."""
    keywords = ['наименование', 'номенклатура', 'name', 'наименов', 'наим.', 'краткое наименование']
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None

def _truncate_dataframe_cells(df, max_length=1000):
    """Обрезка длинных строковых значений."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: str(x)[:max_length] if pd.notna(x) and len(str(x)) > max_length else x
            )
    return df

@click.group()
@click.option('--config', '-c', default='config/config.yaml', help='Путь к конфигурации')
@click.pass_context
def cli(ctx, config):
    """Nomenclature Processor - параметрический процессор сопоставления с ЕНС"""
    ctx.ensure_object(dict)
    config_path = Path(config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            ctx.obj['config'] = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found: {config}")
        ctx.obj['config'] = {}
    try:
        setup_logging(str(config_path))
    except Exception as e:
        logger.warning(f"Failed to setup logging from config: {e}")

# ============================
# LEGACY COMMANDS (LLM Mode)
# ============================

@cli.command()
def prompts():
    """Вывод списка доступных промптов"""
    from core.settings import get_settings
    settings = get_settings()
    click.echo("📋 Доступные промпты:")
    for pid, cfg in settings.prompts.items():
        click.echo(f"\n🔹 {pid}")
        click.echo(f"  Название: {cfg.name}")
        click.echo(f"  Категория: {cfg.category}")
        click.echo(f"  Сервис: {cfg.resolve_service(settings)}")
        click.echo(f"  Модель: {cfg.resolve_model(settings)}")
        click.echo(f"  Ключевые слова: {', '.join(cfg.keywords[:5])}...")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--prompt', '-p', multiple=True, help='ID промпта')
@click.option('--auto', is_flag=True, help='Автоматический выбор промпта')
@click.option('--workers', '-w', default=None, type=int, help='Количество workers')
@click.option('--force', '-f', is_flag=True, help='Принудительная обработка')
@click.pass_context
def process(ctx, input_file, prompt, auto, workers, force):
    """Обработка номенклатуры через LLM (legacy mode)"""
    from core.settings import get_settings
    from core.database import DatabaseManager
    from core.processor import NomenclatureProcessor, load_excel_items
    from utils.excel_loader import ExcelLoader
    settings = get_settings()
    click.echo(f"📊 Загрузка {input_file}...")
    try:
        items = load_excel_items(input_file)
    except:
        loader = ExcelLoader(input_file)
        items = loader.load()
    click.echo(f"📋 Загружено {len(items)} позиций")
    db = DatabaseManager(settings.database.path)
    processor = NomenclatureProcessor(db, max_workers=workers)
    prompt_ids = list(prompt) if prompt else []
    if auto:
        click.echo("🤖 Автоматический выбор промпта...")
        results = processor.auto_process(items, force_reprocess=force)
    else:
        if not prompt_ids:
            click.echo("❌ Укажите --auto или --prompt", err=True)
            return
        results = processor.process_batch(items, prompt_ids, force_reprocess=force)
    click.echo(f"\n✅ Обработано: {len(results)} позиций")
    stats = db.get_statistics()
    click.echo(f"📊 Всего в БД: {stats.get('total', 0)}")

@cli.command()
@click.option('--output', '-o', default='results.json', help='Путь к выходному файлу')
@click.option('--structure', type=click.Choice(['flat', 'by_code', 'by_category', 'by_prompt']),
              default='flat', help='Структура вывода')
@click.option('--prompt', '-p', help='Фильтр по ID промпта')
@click.option('--status', '-s', help='Фильтр по статусу')
@click.option('--include-raw', is_flag=True, help='Включить raw_response')
@click.option('--include-full-request', is_flag=True, help='Включить full_request')
def export(output, structure, prompt, status, include_raw, include_full_request):
    """Экспорт результатов в JSON"""
    from core.settings import get_settings
    from core.database import DatabaseManager
    settings = get_settings()
    db = DatabaseManager(settings.database.path)
    click.echo("📤 Экспорт результатов...")
    results = db.get_all_results(category=None, status=status, prompt_id=prompt, limit=None)
    if not results:
        click.echo("⚠️ Нет данных для экспорта")
        return
    export_data = db.export_filtered_to_json(
        output_path=output, results=results, structure=structure,
        include_raw=include_raw, include_full_request=include_full_request
    )
    click.echo(f"✅ Экспортировано: {len(results)} записей → {output}")

@cli.command()
def stats():
    """Статистика обработки в БД"""
    from core.settings import get_settings
    from core.database import DatabaseManager
    settings = get_settings()
    db = DatabaseManager(settings.database.path)
    stats = db.get_statistics()
    click.echo("📊 Статистика обработки:")
    click.echo(f"  Всего записей: {stats.get('total', 0)}")
    click.echo(f"  По статусам:")
    for status, count in stats.get('by_status', {}).items():
        click.echo(f"    {status}: {count}")
    click.echo(f"  По категориям:")
    for cat, count in stats.get('by_category', {}).items():
        click.echo(f"    {cat}: {count}")
    click.echo(f"  По API:")
    for api, count in stats.get('by_api', {}).items():
        click.echo(f"    {api}: {count}")

@cli.command()
@click.option('--limit', '-l', default=10, help='Количество записей')
@click.option('--prompt', '-p', help='Фильтр по ID промпта')
def errors(limit, prompt):
    """Показать ошибки обработки"""
    from core.settings import get_settings
    from core.database import DatabaseManager
    settings = get_settings()
    db = DatabaseManager(settings.database.path)
    error_results = db.get_all_results(status='error', prompt_id=prompt, limit=limit)
    if not error_results:
        click.echo("✅ Ошибок не найдено")
        return
    click.echo(f"\n❌ Найдено {len(error_results)} ошибок:\n")
    for i, result in enumerate(error_results, 1):
        click.echo(f"{i}. {result.get('article', 'N/A')}: {result.get('name', 'N/A')[:50]}...")
        click.echo(f"   Промпт: {result.get('prompt_id', 'N/A')}")
        click.echo(f"   Ошибка: {result.get('error_message', 'N/A')[:100]}...")
        click.echo()

@cli.command()
@click.argument('text')
def detect(text):
    """Определить категорию номенклатуры"""
    from core.settings import get_settings
    settings = get_settings()
    click.echo(f"🔍 Анализ: {text}")
    for pid, cfg in settings.prompts.items():
        from core.processor import NomenclatureProcessor
        class FakeItem:
            def __init__(self, name):
                self.name = name
                self.article = "test"
                self.guid = "test"
        processor = NomenclatureProcessor.__new__(NomenclatureProcessor)
        processor.settings = settings
        matches = processor._check_category_match(text, cfg)
        if matches:
            click.echo(f"✅ Подходит: {pid} ({cfg.category})")
            click.echo(f"  Сервис: {cfg.resolve_service(settings)}, Модель: {cfg.resolve_model(settings)}")
            return
    click.echo("❌ Категория не определена")

@cli.command()
@click.option('--api', 'api_name', help='Название API')
def models(api_name):
    """Вывод списка моделей API"""
    from core.settings import get_settings
    settings = get_settings()
    services = [api_name] if api_name else list(settings.api.keys())
    for service in services:
        cfg = settings.api.get(service)
        if not cfg:
            click.echo(f"❌ {service}: не найден")
            continue
        click.echo(f"\n🔧 {service.upper()}:")
        click.echo(f"  URL: {cfg.base_url}")
        try:
            if service == 'openwebui':
                if not cfg.api_key and not (cfg.username and cfg.password):
                    click.echo("  ⚠️ Нет credentials")
                    continue
                from api_clients.openwebui import OpenWebUIClient
                client = OpenWebUIClient(base_url=cfg.base_url, api_key=cfg.api_key,
                                         username=cfg.username, password=cfg.password)
            elif service == 'mws':
                if not cfg.api_key:
                    click.echo("  ⚠️ Нет api_key")
                    continue
                from api_clients.mws_gpt import MWSGPTClient
                client = MWSGPTClient(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)
            elif service == 'gigachat':
                if not cfg.api_key:
                    click.echo("  ⚠️ Нет api_key")
                    continue
                from api_clients.gigachat import GigaChatClient
                client = GigaChatClient(base_url=cfg.base_url, api_key=cfg.api_key,
                                        scope=getattr(cfg, 'scope', 'GIGACHAT_API_PERS'),
                                        timeout=cfg.timeout, verify_ssl=False)
            elif service == 'mts_ai':
                if not cfg.api_key:
                    click.echo("  ⚠️ Нет api_key")
                    continue
                from api_clients.mts_ai import MTSAIClient
                client = MTSAIClient(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)
            else:
                click.echo(f"  ⚠️ Неизвестный сервис")
                continue
            model_list = client.get_models()
            if model_list:
                click.echo(f"  Модели ({len(model_list)}):")
                for m in model_list[:10]:
                    click.echo(f"    - {m}")
                if len(model_list) > 10:
                    click.echo(f"    ... и еще {len(model_list) - 10}")
            else:
                click.echo("  ⚠️ Не удалось получить список моделей")
        except Exception as e:
            click.echo(f"  ❌ Ошибка: {e}")

# ============================
# PARAMETRIC COMMANDS (New)
# ============================

def _init_llm_clients(settings, all_services=False):
    """Инициализация LLM клиентов."""
    llm_clients = {}
    if all_services:
        services = list(settings.api.keys())
        logger.info("LLM: initializing all configured services: %s", services)
    else:
        services = [settings.mask_generation.default_service]
        logger.info(f"LLM: using default_service='{services[0]}'")
    for service_name in services:
        if service_name not in settings.api:
            logger.warning("Service '%s' not found in settings.api, skipping", service_name)
            continue
        try:
            cfg = settings.api[service_name]
            if service_name == 'openwebui':
                if not cfg.api_key and not (cfg.username and cfg.password):
                    logger.debug(f"Skipping {service_name}: no credentials")
                    continue
                from api_clients.openwebui import OpenWebUIClient
                llm_clients[service_name] = OpenWebUIClient(
                    base_url=cfg.base_url, api_key=cfg.api_key,
                    username=cfg.username, password=cfg.password)
            elif service_name == 'mws':
                if not cfg.api_key:
                    logger.debug(f"Skipping {service_name}: no api_key")
                    continue
                from api_clients.mws_gpt import MWSGPTClient
                llm_clients[service_name] = MWSGPTClient(
                    base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)
            elif service_name == 'gigachat':
                if not cfg.api_key:
                    logger.debug(f"Skipping {service_name}: no api_key")
                    continue
                from api_clients.gigachat import GigaChatClient
                llm_clients[service_name] = GigaChatClient(
                    base_url=cfg.base_url, api_key=cfg.api_key,
                    scope=getattr(cfg, 'scope', 'GIGACHAT_API_PERS'),
                    timeout=cfg.timeout, verify_ssl=False)
            elif service_name == 'mts_ai':
                if not cfg.api_key:
                    logger.debug(f"Skipping {service_name}: no api_key")
                    continue
                from api_clients.mts_ai import MTSAIClient
                llm_clients[service_name] = MTSAIClient(
                    base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)
            else:
                if not getattr(cfg, 'api_key', None):
                    logger.debug(f"Skipping {service_name}: no api_key")
                    continue
                logger.warning("Unknown service '%s', skipping", service_name)
                continue
            logger.info(f"LLM client initialized: {service_name}")
        except Exception as e:
            logger.warning(f"Failed to init {service_name}: {e}")
    return llm_clients

@cli.command()
@click.argument('text')
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', default=None, help='Путь к индексу ЕНС (default из доменного конфига)')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM для генерации масок')
@click.option('--domain', default='hardware', help='Домен ENS')
def process_parametric(text, db, ens_index, llm, domain):
    """Обработка одной номенклатуры параметрическим методом"""
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from core.settings import get_settings
    llm_clients = {}
    settings = get_settings()
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=False)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM клиенты инициализированы")
    # Автоопределение пути к индексу из доменного конфига
    if not ens_index and domain:
        cfg = DomainConfig.load(domain)
        if cfg.index_path:
            ens_index = cfg.index_path
            click.echo(f"📂 Индекс из домена '{domain}': {ens_index}")
    if not ens_index:
        click.echo("❌ Укажите --ens-index или --domain с настроенным index_path", err=True)
        return
    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db, llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index, use_llm_generation=llm,
        settings=settings, result_db_path='cache/result.db'
    )
    result = processor.process(text)
    click.echo(f"📄 Текст: {result.text}")
    click.echo(f"🏷️ Уровень: {result.level.value}")
    click.echo(f"✅ Успех: {result.success}")
    click.echo(f"🎯 Confidence: {result.confidence:.2f}")
    click.echo(f"⏱️ Время: {result.processing_time_ms:.2f} мс")
    if result.params:
        click.echo(f"📋 Параметры:")
        for key, value in result.params.items():
            if not key.startswith('_'):
                click.echo(f"  {key}: {value}")
    if result.ens_code:
        click.echo(f"🔗 ЕНС совпадение:")
        click.echo(f"  Код: {result.ens_code}")

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', default=None, help='Путь к индексу ЕНС (default из доменного конфига)')
@click.option('--output', '-o', default='results.json', help='Путь к выходному файлу')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM для генерации масок')
@click.option('--validate/--no-validate', default=True, help='Валидировать результаты')
@click.option('--success-only', is_flag=True, help='Включать только успешные результаты')
@click.option('--include-details/--no-include-details', 'include_details', default=None, help='Включать debug-информацию (default из config.output.include_ens_details)')
@click.option('--coating-map', '-c', help='Путь к Excel-файлу с картой покрытий')
@click.option('--workers', '-w', type=int, default=4, help='Количество параллельных workers')
@click.option('--result-db', '-r', default='cache/result.db', help='Путь к SQLite БД результатов')
@click.option('--domain', default='hardware', help='Домен ENS')
@click.option('--auto-domain', is_flag=True, help='Автоматический выбор домена из всех ens_*.pkl')
def batch(input_file, db, ens_index, output, llm, validate, success_only,
          include_details, coating_map, workers, result_db, domain, auto_domain):
    """Пакетная обработка номенклатуры параметрическим методом"""
    import pandas as pd
    from tqdm import tqdm
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from core.settings import get_settings
    # Автоопределение пути к индексу из доменного конфига
    if not ens_index and domain and not auto_domain:
        cfg = DomainConfig.load(domain)
        if cfg.index_path:
            ens_index = cfg.index_path
            click.echo(f"📂 Индекс из домена '{domain}': {ens_index}")
    if not ens_index and not auto_domain:
        click.echo("❌ Укажите --ens-index, --domain с настроенным index_path, или --auto-domain", err=True)
        return 1
    click.echo(f"📊 Загрузка Excel: {input_file}...")
    df = pd.read_excel(input_file)
    click.echo(f"  Прочитано {len(df)} строк, {len(df.columns)} колонок")
    name_col = _find_name_column(df)
    if name_col is None:
        click.echo("\n❌ ОШИБКА: В файле отсутствует колонка с наименованием.")
        click.echo("  Ожидается колонка, содержащая в названии одно из слов:")
        click.echo("  'Наименование', 'Номенклатура', 'Name', 'Наим.', 'Наименов'")
        click.echo(f"\n  Доступные колонки в файле:")
        for i, col in enumerate(df.columns, 1):
            click.echo(f"  {i}. {col}")
        click.echo("\n  Переименуйте колонку с наименованием изделий и повторите запуск.")
        return 1
    click.echo(f"✅ Колонка с наименованием: '{name_col}'")
    texts = df[name_col].astype(str).tolist()
    click.echo(f"📋 Загружено {len(texts)} позиций")
    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"🎨 Карта покрытий загружена: {coating_map}")
    settings = get_settings()
    if include_details is None:
        try:
            include_details = getattr(getattr(settings, 'output', None), 'include_ens_details', True)
        except Exception:
            include_details = True
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=False)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return 1
        click.echo("🤖 LLM клиенты инициализированы")
    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db, llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index, use_llm_generation=llm,
        settings=settings, result_db_path=result_db
    )
    logger.debug("[CLI] result_db_path set to: %s", result_db)
    click.echo("🔍 Обработка...")
    click.echo(f"  Workers: {workers}")
    if success_only:
        click.echo("⚡ Режим: только успешные (пропускаем ошибки)")
    results = [None] * len(texts)
    stats = {'total': 0, 'success': 0, 'failed': 0, 'filtered': 0}
    stats_lock = threading.Lock()

    def _process_one(idx_text):
        idx, text = idx_text
        try:
            result = processor.process(text)
            with stats_lock:
                nonlocal stats
                stats['total'] += 1
                if result.success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                if success_only and not result.success:
                    stats['filtered'] += 1
                    return idx, None
            if result_db:
                logger.debug("[CLI_CACHE] Saving result to %s for '%s' (code=%s)",
                             result_db, result.text[:50], result.ens_code)
                try:
                    from core.result_database import ResultDatabaseManager
                    manager = ResultDatabaseManager(db_path=result_db)
                    changed, reason = manager.upsert_result(
                        name=result.text,
                        article=str(df.iloc[idx].get('Артикул', '')).strip() or None,
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
                        processing_time_ms=result.processing_time_ms
                    )
                    logger.debug("[CLI_CACHE] Saved: changed=%s reason=%s", changed, reason)
                except Exception as e:
                    logger.warning("[CLI_CACHE] Failed to save result: %s", e)
            return idx, result
        except Exception as e:
            logger.error("Error processing item %d: %s", idx, e)
            with stats_lock:
                stats['total'] += 1
                stats['failed'] += 1
            return idx, None

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, (i, t)): i for i, t in enumerate(texts)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(texts), desc="Обработка"):
            idx, result = future.result()
            if result is not None:
                results[idx] = result

    valid_results = [r for r in results if r is not None]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in ('.xlsx', '.xls', '.xlsm'):
        excel_rows = []
        for idx, result in enumerate(results):
            if result is None:
                continue
            out_row = {}
            for col in df.columns:
                val = df.iloc[idx][col]
                if pd.isna(val):
                    out_row[str(col)] = None
                else:
                    out_row[str(col)] = val
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
            out_row['стандарт'] = str(result.standard) if result.standard else ''
            out_row['тип'] = str(result.item_type) if result.item_type else ''
            has_mask = False
            if result.standard and result.item_type:
                try:
                    m = mask_db.get_mask(result.standard, result.item_type)
                    has_mask = m is not None
                except Exception:
                    pass
            out_row['маски_в_бд'] = 'Да' if has_mask else 'Нет'
            if include_details and result.details:
                out_row['детали'] = json.dumps(result.details, ensure_ascii=False, default=str)
            excel_rows.append(out_row)
        df_out = pd.DataFrame(excel_rows)
        df_out = _truncate_dataframe_cells(df_out, max_length=1000)
        if 'Уверенность' in df_out.columns:
            df_out['Уверенность'] = pd.to_numeric(df_out['Уверенность'], errors='coerce').fillna(0.0)
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_out.to_excel(writer, sheet_name='Results', index=False)
            if 'Уверенность' in df_out.columns:
                ws = writer.sheets['Results']
                for idx_col, col_name in enumerate(df_out.columns):
                    if col_name == 'Уверенность':
                        for row_num in range(2, len(df_out) + 2):
                            cell = ws.cell(row=row_num, column=idx_col + 1)
                            cell.number_format = '0.000'
                        break
        file_size = output_path.stat().st_size / 1024
        click.echo(f"\n✅ Excel сохранен: {output}")
        click.echo(f"  Размер: {file_size:.1f} КБ")
    else:
        json_results = []
        for result in valid_results:
            d = result.to_dict()
            if not include_details:
                d.pop('details', None)
            json_results.append(d)
        clean_results = _sanitize_for_json(json_results)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2, default=str)
        click.echo(f"\n✅ JSON сохранен: {output}")

    click.echo(f"\n📊 Статистика:")
    click.echo(f"  Всего обработано: {stats['total']}")
    click.echo(f"  ✅ Успешно: {stats['success']}")
    click.echo(f"  ❌ Ошибки: {stats['failed']}")
    if success_only:
        click.echo(f"  Отфильтровано (неуспешные): {stats['filtered']}")
    if hasattr(processor, '_cache_stats'):
        cs = processor._cache_stats
        click.echo(f"\n💾 Кэш:")
        click.echo(f"  Попаданий (HIT): {cs.get('hits', 0)}")
        click.echo(f"  Промахов (MISS): {cs.get('misses', 0)}")
        total_cache = cs.get('hits', 0) + cs.get('misses', 0)
        if total_cache > 0:
            click.echo(f"  Эффективность: {cs['hits']/total_cache*100:.1f}%")
    return 0

@cli.command('analyze-quality')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', default=None, help='Путь к индексу ЕНС (default из доменного конфига)')
@click.option('--output', '-o', help='Excel-файл для отчета')
@click.option('--json', '-j', 'json_output', help='JSON-файл для отчета')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM')
@click.option('--coating-map', '-c', help='Путь к Excel-файлу с картой покрытий')
@click.option('--domain', default='hardware', help='Домен ENS')
def analyze_quality_cmd(input_file, db, ens_index, output, json_output, llm, coating_map, domain):
    """Анализ качества сопоставления"""
    from core.quality_analyzer import QualityAnalyzer
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from core import get_settings
    settings = get_settings()
    include_details = None
    try:
        include_details = getattr(getattr(settings, 'output', None), 'include_ens_details', True)
    except Exception:
        include_details = True
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=False)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM клиенты инициализированы")
    # Автоопределение пути к индексу из доменного конфига
    if not ens_index and domain:
        cfg = DomainConfig.load(domain)
        if cfg.index_path:
            ens_index = cfg.index_path
            click.echo(f"📂 Индекс из домена '{domain}': {ens_index}")
    if not ens_index:
        click.echo("❌ Укажите --ens-index или --domain с настроенным index_path", err=True)
        return
    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db, llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index, use_llm_generation=llm, settings=settings
    )
    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"🎨 Карта покрытий загружена: {coating_map}")
    analyzer = QualityAnalyzer(processor=processor)
    click.echo(f"📊 Анализ файла: {input_file}...")
    stats = analyzer.analyze_file(input_file)
    report_text = analyzer.format_report(stats)
    click.echo("\n" + report_text)
    if output:
        analyzer.save_excel(stats, output)
        click.echo(f"\n✅ Excel отчет сохранен: {output}")
    if json_output:
        analyzer.save_json(stats, json_output)
        click.echo(f"\n✅ JSON отчет сохранен: {json_output}")

@cli.command()
@click.argument('text')
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', default=None, help='Пути к индексу ЕНС (default из доменного конфига)')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM')
@click.option('--coating-map', '-c', help='Путь к Excel-файлу с картой покрытий')
@click.option('--domain', default='hardware', help='Домен ENS')
@click.option('--auto-domain', is_flag=True, help='Автоматический выбор домена')
def diagnose(text, db, ens_index, llm, coating_map, domain, auto_domain):
    """Диагностика обработки одной номенклатуры"""
    import re
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from core.parametric_client import ParametricENSClient
    from core.settings import get_settings
    settings = get_settings()
    include_details = None
    try:
        include_details = getattr(getattr(settings, 'output', None), 'include_ens_details', True)
    except Exception:
        include_details = True
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=False)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM клиенты инициализированы")
    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"🎨 Карта покрытий загружена: {coating_map}")
    # Автоопределение пути к индексу из доменного конфига
    if not ens_index and domain and not auto_domain:
        cfg = DomainConfig.load(domain)
        if cfg.index_path:
            ens_index = cfg.index_path
            click.echo(f"📂 Индекс из домена '{domain}': {ens_index}")
    if not ens_index and not auto_domain:
        click.echo("❌ Укажите --ens-index, --domain с настроенным index_path, или --auto-domain", err=True)
        return
    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db, llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index, use_llm_generation=llm, settings=settings
    )
    click.echo(f"\n{'='*60}")
    click.echo(f"🔍 ДИАГНОСТИКА: {text}")
    click.echo(f"{'='*60}")
    extracted = processor.standard_extractor.extract_all(text)
    standard_info = extracted.get('standard_info')
    item_type = extracted.get('item_type')
    click.echo(f"\n📋 Извлечено (Level 0):")
    click.echo(f"  standard_info: {standard_info.to_dict() if standard_info else None}")
    click.echo(f"  item_type: {item_type}")
    if not standard_info or not item_type:
        click.echo("\n❌ Недостаточно данных для обработки")
        return
    standard = canonicalize_standard(standard_info.normalized)
    search_item_type = item_type.upper()
    click.echo(f"\n🔍 Поиск маски (Level 1):")
    click.echo(f"  Запрос: standard='{standard}', item_type='{search_item_type}'")
    mask = mask_db.get_mask(standard, search_item_type)
    click.echo(f"  Найдено: {mask is not None}")
    if mask is None:
        mask = mask_db.get_mask(standard, item_type)
        if mask:
            click.echo(f"  Фолбэк (без upper): item_type={item_type}")
    if mask is None:
        click.echo(f"  ❌ Маска не найдена в БД")
        return
    click.echo(f"  mask.id: {getattr(mask, 'id', 'N/A')}")
    click.echo(f"  mask.standard: {getattr(mask, 'standard', 'N/A')}")
    click.echo(f"  mask.item_type: {getattr(mask, 'item_type', 'N/A')}")
    click.echo(f"  mask.is_active: {getattr(mask, 'is_active', 'N/A')}")
    click.echo(f"  mask.pattern (первые 120 симв):")
    click.echo(f"  {getattr(mask, 'pattern', 'N/A')[:120]}")
    effective_standard = getattr(mask, 'standard', None) or standard
    client = ParametricENSClient.__new__(ParametricENSClient)
    relaxed = client._relax_pattern(mask.pattern, standard=effective_standard)
    click.echo(f"\n📋 Relax pattern:")
    click.echo(f"  standard заменен: '{effective_standard}'")
    click.echo(f"  relaxed (первые 200 симв):")
    click.echo(f"  {relaxed[:200]}")
    if len(relaxed) > 200:
        click.echo(f"  ... ({len(relaxed)} символов всего)")
    try:
        compiled = re.compile(relaxed, re.IGNORECASE)
        match = compiled.search(text)
        click.echo(f"\n📋 Regex match:")
        if match:
            click.echo(f"  ✅ MATCH")
            click.echo(f"  groups: {match.groupdict()}")
        else:
            click.echo(f"  ❌ NO MATCH")
            for i in range(len(text), 0, -1):
                if compiled.search(text[:i]):
                    click.echo(f"  longest matching prefix: '{text[:i]}'")
                    break
            else:
                click.echo(f"  no prefix matches at all")
    except re.error as e:
        click.echo(f"\n📋 Regex match:")
        click.echo(f"  ❌ INVALID REGEX: {e}")
    click.echo(f"\n📋 Full processor result:")
    result = processor.process(text)
    click.echo(f"  level: {result.level.value if hasattr(result.level, 'value') else result.level}")
    click.echo(f"  success: {result.success}")
    click.echo(f"  params: {result.params}")
    click.echo(f"  ens_code: {result.ens_code}")
    click.echo(f"  ens_name: {result.ens_name}")
    click.echo(f"  confidence: {result.confidence:.3f}")
    click.echo(f"  processing_time_ms: {result.processing_time_ms:.1f}")
    if result.details:
        click.echo(f"  details: {result.details}")
    click.echo(f"\n{'='*60}")

@cli.command('generate-masks')
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', default=None, help='Путь к индексу ЕНС (default из доменного конфига)')
@click.option('--standard', '-s', help='Стандарт для генерации маски')
@click.option('--item-type', '-t', help='Тип изделия')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM')
@click.option('--validate', is_flag=True, help='Валидировать маску')
@click.option('--min-score', default=0.85, help='Минимальный score')
@click.option('--limit', '-n', default=0, help='Ограничить число стандартов')
@click.option('--force', '-f', is_flag=True, help='Принудительная перегенерация')
@click.option('--stats-output', '-so', type=click.Path(), help='Excel-файл статистики')
@click.option('--domain', default='hardware', help='Домен ENS (hardware, rolled_metal, eri)')
def generate_masks(db, ens_index, standard, item_type, llm, validate, min_score, limit, force, stats_output, domain):
    """Генерация масок для стандартов из индекса ЕНС."""
    from core.mask_database import MaskDatabase, MaskRecord
    from core.llm_mask_generator import LLMMaskGenerator
    from core.auto_validator import AutoValidator
    from core.settings import get_settings
    from pathlib import Path
    import pickle
    # Автоопределение пути к индексу из доменного конфига
    if not ens_index and domain:
        cfg = DomainConfig.load(domain)
        if cfg.index_path:
            ens_index = cfg.index_path
            click.echo(f"📂 Индекс из домена '{domain}': {ens_index}")
    if not ens_index:
        click.echo("❌ Укажите --ens-index или --domain с настроенным index_path", err=True)
        return
    if not Path(ens_index).exists():
        click.echo("❌ Индекс не найден", err=True)
        return
    settings = get_settings()
    mask_db = MaskDatabase(db_path=db)
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=False)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM клиенты инициализированы")
        generator = LLMMaskGenerator(clients=llm_clients, settings=settings, max_retries=3, domain=domain, ens_index_path=ens_index)
    else:
        generator = None
        click.echo("⚠️ Режим без LLM — только просмотр/валидация")

    # === РЕЖИМ 1: Одиночная генерация ===
    if standard and item_type:
        canon_std = canonicalize_standard(standard)
        validator = AutoValidator(ens_index_path=ens_index, domain=domain)
        examples = validator._get_ens_examples(canon_std, item_type, domain=domain)
        click.echo(f"📋 Загружено {len(examples)} примеров для {canon_std} / {item_type}")
        if not examples:
            click.echo("❌ Нет примеров")
            return
        if not generator:
            click.echo("❌ Для генерации укажите --llm")
            return
        click.echo(f"🎯 Генерация маски для {canon_std} / {item_type}...")
        mask, meta = generator.generate_mask(canon_std, item_type, examples)
        if mask:
            click.echo(f"✅ Маска сгенерирована:")
            click.echo(f"  Паттерн: {mask['pattern'][:80]}...")
            click.echo(f"  Параметры: {mask['params']}")
            click.echo(f"  Обязательные: {mask['required']}")
            if validate:
                click.echo("🔍 Валидация...")
                validation = validator.validate_mask(
                    mask['pattern'], mask['params'], mask['required'],
                    canon_std, item_type, ens_examples=examples
                )
                click.echo(f"  Score: {validation.score:.2f}, Passed: {validation.passed}")
                auto_score = validation.score
            else:
                auto_score = 0.0
            mask_record = MaskRecord(
                standard=canon_std, item_type=item_type.upper(),
                pattern=mask['pattern'], params=mask['params'],
                required=mask['required'], source='llm',
                auto_score=auto_score,
                is_active=True if auto_score >= min_score else (auto_score == 0.0)
            )
            mask_id = mask_db.save_mask(mask_record, auto_activate=True, replace_existing=True)
            if mask_id:
                click.echo(f"✅ Маска сохранена: ID={mask_id}")
                if force:
                    click.echo("  🔄 Force: перегенерирована")
            else:
                click.echo("⚠️ Не удалось сохранить маску")
        else:
            click.echo("❌ Не удалось сгенерировать маску")
        return

    # === РЕЖИМ 2: Массовая генерация ===
    with open(ens_index, 'rb') as f:
        data = pickle.load(f)
    # Поддержка нового структурированного формата индекса
    if isinstance(data, dict) and not data.get('items'):
        # Новый формат: {standard: {item_type: {examples: [...], ...}}}
        standards = {}
        for std, types in data.items():
            for itype, entry in types.items():
                examples = entry.get('examples', [])
                if len(examples) >= 10:
                    standards[(std, itype)] = examples
    else:
        # Legacy формат
        items = data.get('items', [])
        standards = {}
        for item in items:
            std_raw = item.get('стандарт') or item.get('нтд') or 'UNKNOWN'
            std = canonicalize_standard(std_raw)
            itype = item.get('тип_изделия') or item.get('наименование_типа') or item.get('тип') or 'unknown'
            key = (std, itype)
            if key not in standards:
                standards[key] = []
            standards[key].append(item)
        standards = {k: v for k, v in standards.items() if len(v) >= 10}

    # Фильтр по стандарту
    if standard:
        canon_filter = canonicalize_standard(standard)
        matched = {}
        for (std, itype), examples in standards.items():
            if canon_filter in std or std in canon_filter:
                matched[(std, itype)] = examples
        standards = matched
        click.echo(f"🔍 Фильтр по стандарту '{canon_filter}': найдено {len(standards)} пар")
        if not standards:
            click.echo("❌ Ничего не найдено")
            return

    all_items = list(standards.items())
    click.echo(f"🔍 Найдено {len(all_items)} пар с >=10 примерами")
    if limit and limit > 0:
        standards = dict(all_items[:limit])
        click.echo(f"🔧 Отладочный режим: {len(standards)} пар")
    stats = {'existing': 0, 'generated': 0, 'activated': 0}
    stats_rows = []
    with click.progressbar(standards.items(), label='Генерация') as bar:
        for (std, itype), examples in bar:
            item_type_normalized = itype.upper()
            old_mask = mask_db.get_mask(std, item_type_normalized)
            old_pattern = old_mask.pattern if old_mask else None
            old_score = old_mask.auto_score if old_mask else None
            old_is_active = old_mask.is_active if old_mask else None
            if old_mask and old_mask.is_active and not force:
                stats['existing'] += 1
                stats_rows.append({
                    'тип': itype, 'стандарт': std, 'маска': old_pattern,
                    'score': old_score, 'старая_маска': old_pattern,
                    'старый_score': old_score, 'is_active': old_is_active,
                    'служба': 'skipped', 'модель': None, 'температура': None,
                    'old_mask_id': old_mask.id if old_mask else None,
                    'warning': 'Already active, skipped (use --force)',
                    'tokens_prompt': None, 'tokens_completion': None,
                })
                continue
            if old_mask and old_mask.is_active and force:
                click.echo(f"  🔄 Force: перегенерация {std}/{itype}")
            if generator:
                limited_examples = examples[:20]
                mask, meta = generator.generate_mask(std, itype, limited_examples)
                if mask:
                    auto_score = 0.0
                    is_active = True
                    if validate:
                        from core.auto_validator import AutoValidator
                        validator = AutoValidator(ens_index_path=ens_index, domain=domain)
                        validation = validator.validate_mask(
                            mask['pattern'], mask['params'], mask['required'],
                            std, itype, ens_examples=limited_examples
                        )
                        auto_score = validation.score
                        is_active = auto_score >= min_score
                    click.echo(f"  🔍 Маска {std}/{itype}: score={auto_score:.2f}, active={is_active}")
                    temp_mask = MaskRecord(
                        standard=std, item_type=item_type_normalized,
                        pattern=mask['pattern'], params=mask['params'],
                        required=mask['required'], auto_score=auto_score,
                        is_active=is_active, source='llm'
                    )
                    mask_db.save_mask(temp_mask, auto_activate=True, replace_existing=True)
                    stats['generated'] += 1
                    if is_active:
                        stats['activated'] += 1
                    stats_rows.append({
                        'тип': item_type_normalized, 'стандарт': std,
                        'маска': mask['pattern'], 'score': auto_score,
                        'старая_маска': old_pattern, 'старый_score': old_score,
                        'is_active': is_active,
                        'служба': meta.get('provider') if meta else None,
                        'модель': meta.get('model') if meta else None,
                        'температура': meta.get('temperature') if meta else None,
                        'old_mask_id': old_mask.id if old_mask else None,
                        'warning': '; '.join(meta.get('warnings', [])) if meta and meta.get('warnings') else None,
                        'tokens_prompt': meta.get('tokens_prompt') if meta else None,
                        'tokens_completion': meta.get('tokens_completion') if meta else None,
                    })
                else:
                    stats_rows.append({
                        'тип': itype, 'стандарт': std, 'маска': None,
                        'score': None, 'старая_маска': old_pattern,
                        'старый_score': old_score, 'is_active': False,
                        'служба': meta.get('provider') if meta else None,
                        'модель': meta.get('model') if meta else None,
                        'температура': meta.get('temperature') if meta else None,
                        'old_mask_id': old_mask.id if old_mask else None,
                        'warning': '; '.join(meta.get('warnings', [])) if meta and meta.get('warnings') else 'Failed',
                        'tokens_prompt': meta.get('tokens_prompt') if meta else None,
                        'tokens_completion': meta.get('tokens_completion') if meta else None,
                    })
    click.echo(f"\n📊 Результат:")
    click.echo(f"  Уже активных: {stats['existing']}")
    click.echo(f"  Сгенерировано: {stats['generated']}")
    click.echo(f"  Активировано: {stats['activated']}")
    if stats_output and stats_rows:
        df_stats = pd.DataFrame(stats_rows)
        if 'old_mask_id' not in df_stats.columns:
            df_stats['old_mask_id'] = None
        df_stats.to_excel(stats_output, index=False)
        click.echo(f"📊 Статистика сохранена: {stats_output}")

@cli.command()
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--threshold', '-t', default=0.5, help='Минимальный score')
def cleanup(db, threshold):
    """Очистка неактивных масок с низким score"""
    from core.mask_database import MaskDatabase
    mask_db = MaskDatabase(db_path=db)
    deleted = mask_db.cleanup_low_score_masks(threshold)
    click.echo(f"🗑️ Удалено {deleted} масок с score < {threshold}")

# ============================
# ENS COMMANDS
# ============================

@cli.group()
def ens():
    """Команды для работы с индексом ЕНС"""
    pass

@ens.command('build-index')
@click.argument('excel_file', type=click.Path(exists=True))
@click.option('--output', '-o', required=True, help='Output .pkl path')
@click.option('--domain', '-d', default='hardware', help='Домен (hardware, rolled_metal, eri)')
def build_index(excel_file, output, domain):
    """Build structured ENS domain index from Excel"""
    click.echo(f"Building structured index from {excel_file} for domain={domain}...")
    config = DomainConfig.load(domain)
    builder = ENSIndexBuilder(config)
    result_path = builder.build(excel_file, output)
    click.echo(f"Index saved: {result_path}")
    click.echo(f"Domain: {domain}")
    click.echo(f"Description: {config.description}")
    # Сводная статистика
    import pickle
    with open(result_path, 'rb') as f:
        idx = pickle.load(f)
    total_std = len(idx)
    total_types = sum(len(v) for v in idx.values())
    total_examples = sum(
        len(entry.get('examples', []))
        for std_data in idx.values()
        for entry in std_data.values()
    )
    click.echo(f"\n📊 Сводная статистика:")
    click.echo(f"  Стандартов: {total_std}")
    click.echo(f"  Типов изделий: {total_types}")
    click.echo(f"  Всего примеров: {total_examples}")
    click.echo(f"  Среднее примеров на (стандарт, тип): {total_examples // max(total_types, 1)}")

@ens.command()
@click.argument('pkl_file', type=click.Path(exists=True))
def info(pkl_file):
    """Информация об индексе ЕНС"""
    import pickle
    from pathlib import Path
    from collections import Counter
    click.echo(f"📊 Анализ индекса: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and not data.get('items'):
        # Новый структурированный формат
        click.echo(f"  Формат: структурированный доменный индекс")
        click.echo(f"  Стандартов: {len(data)}")
        total_types = sum(len(v) for v in data.values())
        click.echo(f"  Типов изделий: {total_types}")
        total_examples = sum(
            len(entry.get('examples', []))
            for std_data in data.values()
            for entry in std_data.values()
        )
        click.echo(f"  Всего примеров: {total_examples}")
        click.echo(f"\n📋 Топ-10 стандартов:")
        for std, types in sorted(data.items(), key=lambda x: -len(x[1]))[:10]:
            ex_count = sum(len(entry.get('examples', [])) for entry in types.values())
            click.echo(f"  {std}: {len(types)} типов, {ex_count} примеров")
    else:
        # Legacy формат
        items = data.get('items', [])
        click.echo(f"  Формат: legacy плоский индекс")
        click.echo(f"  Всего записей: {len(items)}")
        std_counter = Counter()
        type_counter = Counter()
        for item in items:
            std = item.get('стандарт') or item.get('нтд') or 'UNKNOWN'
            itype = item.get('тип_изделия') or item.get('наименование_типа') or 'unknown'
            std_counter[std] += 1
            type_counter[itype] += 1
        click.echo(f"\n📋 Топ-10 стандартов:")
        for std, count in std_counter.most_common(10):
            click.echo(f"  {std}: {count}")
        click.echo(f"\n📋 Топ-10 типов:")
        for itype, count in type_counter.most_common(10):
            click.echo(f"  {itype}: {count}")

@ens.command()
@click.argument('pkl_file', type=click.Path(exists=True))
@click.option('--standard', '-s', help='Фильтр по стандарту')
@click.option('--item-type', '-t', help='Фильтр по типу')
@click.option('--limit', '-l', default=5, help='Лимит примеров')
def show(pkl_file, standard, item_type, limit):
    """Просмотр примеров из индекса"""
    import pickle
    from utils.standard_utils import canonicalize_standard
    click.echo(f"📋 Просмотр индекса: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and not data.get('items'):
        # Новый структурированный формат
        for std, types in data.items():
            if standard and canonicalize_standard(standard) != std:
                continue
            for itype, entry in types.items():
                if item_type and item_type.lower() != itype.lower():
                    continue
                click.echo(f"\n🔹 {std} / {itype}")
                examples = entry.get('examples', [])
                twin_groups = entry.get('twin_groups', [])
                field_meta = entry.get('field_meta', {})
                stats = entry.get('stats', {})
                click.echo(f"  Примеры: {len(examples)}")
                click.echo(f"  Близнецы: {twin_groups}")
                click.echo(f"  Поля: {stats.get('visible_fields', [])}")
                click.echo(f"  Метаданные: {stats.get('metadata_fields', [])}")
                for i, ex in enumerate(examples[:limit]):
                    meta = ex.get('_meta', {})
                    click.echo(f"  {i+1}. {meta.get('name', 'N/A')[:80]}")
    else:
        # Legacy формат
        items = data.get('items', [])
        filtered = []
        for item in items:
            std = canonicalize_standard(item.get('стандарт') or item.get('нтд') or '')
            itype = item.get('тип_изделия') or item.get('наименование_типа') or ''
            if standard and canonicalize_standard(standard) != std:
                continue
            if item_type and item_type.lower() != itype.lower():
                continue
            filtered.append(item)
        click.echo(f"  Найдено: {len(filtered)} записей")
        for i, item in enumerate(filtered[:limit]):
            name = item.get('наименование') or item.get('полное_наименование') or 'N/A'
            click.echo(f"  {i+1}. {name[:80]}")

if __name__ == '__main__':
    cli()