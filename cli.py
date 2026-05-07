#!/usr/bin/env python3
"""
Nomenclature Processor CLI
Полный интерфейс для обработки номенклатуры (LLM + Parametric modes)
LAST_FIX: 2026-05-07 13:35 UTC+3 — cfg.resolve_service/model() вместо прямого доступа; service/model опциональны в prompts.yaml
"""

import click
import logging
import yaml
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from config.settings import setup_logging

logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', default='config/config.yaml', help='Путь к конфигу')
@click.pass_context
def cli(ctx, config):
    """Nomenclature Processor - система обработки номенклатуры"""
    ctx.ensure_object(dict)
    config_path = Path(config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            ctx.obj['config'] = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found: {config}")
        ctx.obj['config'] = {}

    # Настройка логирования из config.yaml
    try:
        setup_logging(str(config_path))
    except Exception as e:
        logger.warning(f"Failed to setup logging from config: {e}")


# =============================================================================
# LEGACY COMMANDS (LLM Mode)
# =============================================================================

@cli.command()
def prompts():
    """Список доступных промптов"""
    from config.settings import get_settings
    settings = get_settings()

    click.echo("📋 Доступные промпты:")
    for pid, cfg in settings.prompts.items():
        click.echo(f"\n🔹 {pid}")
        click.echo(f"   Название: {cfg.name}")
        click.echo(f"   Категория: {cfg.category}")
        click.echo(f"   Сервис: {cfg.resolve_service(settings)}")
        click.echo(f"   Модель: {cfg.resolve_model(settings)}")
        click.echo(f"   Ключевые слова: {', '.join(cfg.keywords[:5])}...")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--prompt', '-p', multiple=True, help='ID промпта (можно несколько)')
@click.option('--auto', is_flag=True, help='Автоопределение промптов по ключевым словам')
@click.option('--workers', '-w', default=None, type=int, help='Количество workers')
@click.option('--force', '-f', is_flag=True, help='Перезапись существующих')
@click.pass_context
def process(ctx, input_file, prompt, auto, workers, force):
    """Обработка номенклатуры через LLM (legacy mode)"""
    from config.settings import get_settings
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

    click.echo(f"✅ Загружено {len(items)} записей")

    db = DatabaseManager(settings.database.path)
    processor = NomenclatureProcessor(db, max_workers=workers)

    prompt_ids = list(prompt) if prompt else []
    if auto:
        click.echo("🔍 Автоопределение промптов...")
        results = processor.auto_process(items, force_reprocess=force)
    else:
        if not prompt_ids:
            click.echo("❌ Укажите --auto или --prompt", err=True)
            return
        results = processor.process_batch(items, prompt_ids, force_reprocess=force)

    click.echo(f"\n✅ Обработка завершена: {len(results)} результатов")

    stats = db.get_statistics()
    click.echo(f"📈 Всего в БД: {stats.get('total', 0)}")


@cli.command()
@click.option('--output', '-o', default='results.json', help='Файл для экспорта')
@click.option('--structure', type=click.Choice(['flat', 'by_code', 'by_category', 'by_prompt']),
              default='flat', help='Структура вывода')
@click.option('--prompt', '-p', help='Фильтр по ID промпта')
@click.option('--status', '-s', help='Фильтр по статусу (completed, error, ignored)')
@click.option('--include-raw', is_flag=True, help='Включить raw_response')
@click.option('--include-full-request', is_flag=True, help='Включить full_request')
def export(output, structure, prompt, status, include_raw, include_full_request):
    """Экспорт результатов обработки в JSON"""
    from config.settings import get_settings
    from core.database import DatabaseManager

    settings = get_settings()
    db = DatabaseManager(settings.database.path)

    click.echo("📤 Экспорт результатов...")

    results = db.get_all_results(
        category=None,
        status=status,
        prompt_id=prompt,
        limit=None
    )

    if not results:
        click.echo("⚠️  Нет данных для экспорта")
        return

    export_data = db.export_filtered_to_json(
        output_path=output,
        results=results,
        structure=structure,
        include_raw=include_raw,
        include_full_request=include_full_request
    )

    click.echo(f"✅ Экспортировано: {len(results)} записей → {output}")


@cli.command()
def stats():
    """Статистика по результатам в БД"""
    from config.settings import get_settings
    from core.database import DatabaseManager

    settings = get_settings()
    db = DatabaseManager(settings.database.path)

    stats = db.get_statistics()

    click.echo("📊 Статистика результатов:")
    click.echo(f"   Всего записей: {stats.get('total', 0)}")
    click.echo(f"   По статусам:")
    for status, count in stats.get('by_status', {}).items():
        click.echo(f"      {status}: {count}")
    click.echo(f"   По категориям:")
    for cat, count in stats.get('by_category', {}).items():
        click.echo(f"      {cat}: {count}")
    click.echo(f"   По API:")
    for api, count in stats.get('by_api', {}).items():
        click.echo(f"      {api}: {count}")


@cli.command()
@click.option('--limit', '-l', default=10, help='Количество ошибок')
@click.option('--prompt', '-p', help='Фильтр по ID промпта')
def errors(limit, prompt):
    """Просмотр ошибок обработки"""
    from config.settings import get_settings
    from core.database import DatabaseManager

    settings = get_settings()
    db = DatabaseManager(settings.database.path)

    error_results = db.get_all_results(status='error', prompt_id=prompt, limit=limit)

    if not error_results:
        click.echo("✅ Ошибок не найдено")
        return

    click.echo(f"❌ Последние {len(error_results)} ошибок:\n")

    for i, result in enumerate(error_results, 1):
        click.echo(f"{i}. {result.get('article', 'N/A')}: {result.get('name', 'N/A')[:50]}...")
        click.echo(f"   Промпт: {result.get('prompt_id', 'N/A')}")
        click.echo(f"   Ошибка: {result.get('error_message', 'N/A')[:100]}...")
        click.echo()


@cli.command()
@click.argument('text')
def detect(text):
    """Определить категорию номенклатуры"""
    from config.settings import get_settings

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
            click.echo(f"✅ Совпадение: {pid} ({cfg.category})")
            click.echo(f"   Сервис: {cfg.resolve_service(settings)}, Модель: {cfg.resolve_model(settings)}")
            return

    click.echo("❌ Категория не определена")


@cli.command()
@click.option('--api', 'api_name', help='Проверить конкретный API (openwebui, mws, gigachat)')
def models(api_name):
    """Список доступных моделей API"""
    from config.settings import get_settings

    settings = get_settings()

    services = [api_name] if api_name else list(settings.api.keys())

    for service in services:
        cfg = settings.api.get(service)
        if not cfg:
            click.echo(f"❌ {service}: не настроен")
            continue

        click.echo(f"\n🔌 {service.upper()}:")
        click.echo(f"   URL: {cfg.base_url}")

        try:
            if service == 'openwebui':
                from api_clients.openwebui import OpenWebUIClient
                client = OpenWebUIClient(
                    base_url=cfg.base_url,
                    api_key=cfg.api_key,
                    username=cfg.username,
                    password=cfg.password
                )
            elif service == 'mws':
                from api_clients.mws_gpt import MWSGPTClient
                client = MWSGPTClient(
                    base_url=cfg.base_url,
                    api_key=cfg.api_key
                )
            elif service == 'gigachat':
                from api_clients.gigachat import GigaChatClient
                client = GigaChatClient(
                    base_url=cfg.base_url,
                    api_key=cfg.api_key
                )
            elif service == 'mts_ai':
                from api_clients.mts_ai import MTSAIClient
                client = MTSAIClient(
                    base_url=cfg.base_url,
                    api_key=cfg.api_key
                )
            else:
                continue

            model_list = client.get_models()
            if model_list:
                click.echo(f"   Модели ({len(model_list)}):")
                for m in model_list[:10]:
                    click.echo(f"      - {m}")
                if len(model_list) > 10:
                    click.echo(f"      ... и еще {len(model_list) - 10}")
            else:
                click.echo("   ⚠️  Нет доступных моделей")

        except Exception as e:
            click.echo(f"   ❌ Ошибка: {e}")


# =============================================================================
# PARAMETRIC COMMANDS (New)
# =============================================================================

def _init_llm_clients(settings, all_services=False):
    """Инициализация LLM клиентов для всех настроенных сервисов."""
    llm_clients = {}
    services = ['openwebui', 'mws', 'gigachat'] if all_services else ['openwebui', 'mws']
    for service_name in services:
        if service_name in settings.api:
            try:
                cfg = settings.api[service_name]
                if service_name == 'openwebui':
                    from api_clients.openwebui import OpenWebUIClient
                    llm_clients[service_name] = OpenWebUIClient(
                        base_url=cfg.base_url, api_key=cfg.api_key,
                        username=cfg.username, password=cfg.password
                    )
                elif service_name == 'mws':
                    from api_clients.mws_gpt import MWSGPTClient
                    llm_clients[service_name] = MWSGPTClient(
                        base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout
                    )
                elif service_name == 'gigachat':
                    from api_clients.gigachat import GigaChatClient
                    llm_clients[service_name] = GigaChatClient(
                        base_url=cfg.base_url, api_key=cfg.api_key,
                        scope=getattr(cfg, 'scope', 'GIGACHAT_API_PERS'),
                        timeout=cfg.timeout, verify_ssl=False
                    )
            except Exception as e:
                logger.warning(f"Failed to init {service_name}: {e}")
    return llm_clients


@cli.command()
@click.argument('text')
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕСН')
@click.option('--llm', '-l', is_flag=True, help='Разрешить LLM генерацию масок')
def process_parametric(text, db, ens_index, llm):
    """Обработка одной строки с параметрическим поиском"""
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from config.settings import get_settings

    llm_clients = {}
    settings = get_settings()
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM генерация включена")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    result = processor.process(text)

    click.echo(f"📋 Текст: {result.text}")
    click.echo(f"🔹 Уровень: {result.level.value}")
    click.echo(f"✅ Успех: {result.success}")
    click.echo(f"📊 Уверенность: {result.confidence:.2f}")
    click.echo(f"⏱️  Время: {result.processing_time_ms:.2f} мс")

    if result.params:
        click.echo(f"📌 Параметры:")
        for key, value in result.params.items():
            if not key.startswith('_'):
                click.echo(f"   {key}: {value}")

    if result.ens_match:
        click.echo(f"🔗 ЕСН совпадение:")
        click.echo(f"   Код: {result.ens_match.get('code')}")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕСН')
@click.option('--output', '-o', default='results.json', help='Выходной файл')
@click.option('--llm', '-l', is_flag=True, help='Разрешить LLM генерацию')
@click.option('--validate/--no-validate', default=True, help='Проверять валидность')
@click.option('--success-only', is_flag=True, help='Выгружать только успешно распознанные')
@click.option('--include-details', is_flag=True, help='Включить debug-информацию (details) в вывод')
@click.option('--coating-map', '-c', help='Путь к Excel-справочнику покрытий')
def batch(input_file, db, ens_index, output, llm, validate, success_only, include_details, coating_map):
    """Пакетная обработка с параметрическим поиском"""
    import pandas as pd
    from tqdm import tqdm
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from config.settings import get_settings

    click.echo(f"📊 Загрузка {input_file}...")
    df = pd.read_excel(input_file)

    name_col = 'Краткое наименование'
    if name_col not in df.columns:
        name_cols = [c for c in df.columns if 'наименование' in str(c).lower()]
        if name_cols:
            name_col = name_cols[0]
        else:
            click.echo("❌ Колонка с наименованием не найдена", err=True)
            return

    texts = df[name_col].astype(str).tolist()
    click.echo(f"✅ Загружено {len(texts)} записей")

    # Инициализация CoatingMapper
    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"🎨 Справочник покрытий загружен: {coating_map}")

    settings = get_settings()
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM генерация включена")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    click.echo("🚀 Начало обработки...")
    if success_only:
        click.echo("📌 Режим: только успешные результаты")
    results = []
    stats = {'total': 0, 'success': 0, 'failed': 0, 'filtered': 0}

    for text in tqdm(texts, desc="Обработка"):
        result = processor.process(text)
        stats['total'] += 1

        if result.success:
            stats['success'] += 1
        else:
            stats['failed'] += 1

        # Фильтрация: пропускаем неуспешные если --success-only
        if success_only and not result.success:
            stats['filtered'] += 1
            continue

        row = {
            'text': result.text,
            'level': result.level.value,
            'success': result.success,
            'params': result.params,
            'ens_code': result.ens_match.get('code') if result.ens_match else None,
            'ens_name': result.ens_match.get('name') if result.ens_match else None,
            'ens_params': result.ens_params,
            'ens_params_mask': result.ens_params_mask,
            'confidence': result.confidence,
            'processing_time_ms': result.processing_time_ms,
            'item_type': result.item_type,
            'standard': result.standard,
            'mask_pattern': result.details.get('mask_pattern') if result.details else None,
            'coating_substitution': result.details.get('coating_substitution') if result.details else None
        }
        if include_details and result.details:
            row['details'] = result.details

        results.append(row)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    click.echo(f"\n💾 Результаты: {output} ({len(results)} записей)")
    click.echo(f"📊 Всего обработано: {stats['total']}")
    click.echo(f"✅ Успешно: {stats['success']}")
    click.echo(f"❌ Не распознано: {stats['failed']}")
    if success_only:
        click.echo(f"🚫 Отфильтровано: {stats['filtered']}")


@cli.command('analyze-quality')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕСН')
@click.option('--output', '-o', help='Excel-файл для сохранения отчета')
@click.option('--json', '-j', 'json_output', help='JSON-файл для детального отчета')
@click.option('--llm', '-l', is_flag=True, help='Разрешить LLM генерацию масок')
@click.option('--coating-map', '-c', help='Путь к Excel-справочнику покрытий')
def analyze_quality_cmd(input_file, db, ens_index, output, json_output, llm, coating_map):
    """Анализ качества распознавания: статистика по (item_type, standard)"""
    from core.quality_analyzer import QualityAnalyzer
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from config.settings import get_settings

    settings = get_settings()
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM генерация включена")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    # Инициализация CoatingMapper
    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"🎨 Справочник покрытий загружен: {coating_map}")

    from core.quality_analyzer import QualityAnalyzer
    analyzer = QualityAnalyzer(processor=processor)

    click.echo(f"📊 Анализ файла: {input_file}...")
    stats = analyzer.analyze_file(input_file)
    report_text = analyzer.format_report(stats)

    click.echo("\n" + report_text)

    if output:
        analyzer.save_excel(stats, output)
        click.echo(f"\n💾 Excel отчет сохранен: {output}")

    if json_output:
        analyzer.save_json(stats, json_output)
        click.echo(f"\n💾 JSON отчет сохранен: {json_output}")


@cli.command('diagnose')
@click.argument('text')
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕСН')
@click.option('--llm', '-l', is_flag=True, help='Разрешить LLM генерацию масок')
@click.option('--coating-map', '-c', help='Путь к Excel-справочнику покрытий')
def diagnose(text, db, ens_index, llm, coating_map):
    """Диагностика обработки одной строки номенклатуры."""
    import re
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from core.parametric_client import ParametricENSClient
    from config.settings import get_settings

    settings = get_settings()
    llm_clients = {}
    if llm:
        # Lazy import to avoid startup overhead
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM генерация включена")

    # Инициализация CoatingMapper
    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"🎨 Справочник покрытий загружен: {coating_map}")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    click.echo(f"\n{'=' * 60}")
    click.echo(f"🔍 ДИАГНОСТИКА: {text}")
    click.echo(f"{'=' * 60}")

    # Step 0: Standard extraction
    extracted = processor.standard_extractor.extract_all(text)
    standard_info = extracted.get('standard_info')
    item_type = extracted.get('item_type')
    click.echo(f"\n📌 Извлечение (Level 0):")
    click.echo(f"   standard_info: {standard_info.to_dict() if standard_info else None}")
    click.echo(f"   item_type: {item_type}")

    if not standard_info or not item_type:
        click.echo("\n❌ Не удалось извлечь стандарт или тип — переход на LLM Direct")
        return

    standard = standard_info.normalized
    search_item_type = item_type.upper()

    # Step 1: Mask lookup
    click.echo(f"\n📌 Mask lookup (Level 1):")
    mask = mask_db.get_mask(standard, search_item_type)
    click.echo(f"   Поиск: standard='{standard}', item_type='{search_item_type}'")
    click.echo(f"   Найдена: {mask is not None}")

    if mask is None:
        mask = mask_db.get_mask(standard, item_type)
        if mask:
            click.echo(f"   Найдена (оригинальный регистр): item_type='{item_type}'")

    if mask is None:
        click.echo(f"   ❌ Маска не найдена в БД")
        return

    click.echo(f"   mask.id: {getattr(mask, 'id', 'N/A')}")
    click.echo(f"   mask.standard: {getattr(mask, 'standard', 'N/A')}")
    click.echo(f"   mask.item_type: {getattr(mask, 'item_type', 'N/A')}")
    click.echo(f"   mask.is_active: {getattr(mask, 'is_active', 'N/A')}")
    click.echo(f"   mask.pattern (первые 120 символов):")
    click.echo(f"      {getattr(mask, 'pattern', 'N/A')[:120]}")

    # Step 2: Pattern relaxation
    effective_standard = getattr(mask, 'standard', None) or standard
    client = ParametricENSClient.__new__(ParametricENSClient)
    relaxed = client._relax_pattern(mask.pattern, standard=effective_standard)
    click.echo(f"\n📌 Relax pattern:")
    click.echo(f"   standard передан: '{effective_standard}'")
    click.echo(f"   relaxed (первые 200 символов):")
    click.echo(f"      {relaxed[:200]}")
    if len(relaxed) > 200:
        click.echo(f"      ... ({len(relaxed)} символов всего)")

    # Step 3: Regex match
    try:
        compiled = re.compile(relaxed, re.IGNORECASE)
        match = compiled.search(text)
        click.echo(f"\n📌 Regex match:")
        if match:
            click.echo(f"   ✅ MATCH")
            click.echo(f"   groups: {match.groupdict()}")
        else:
            click.echo(f"   ❌ NO MATCH")
            # Find longest prefix
            for i in range(len(text), 0, -1):
                if compiled.search(text[:i]):
                    click.echo(f"   longest matching prefix: '{text[:i]}'")
                    break
            else:
                click.echo(f"   no prefix matches at all")
    except re.error as e:
        click.echo(f"\n📌 Regex match:")
        click.echo(f"   ❌ INVALID REGEX: {e}")
        click.echo(f"   pattern: {relaxed[:100]}")

    # Step 4: Full processor result
    click.echo(f"\n📌 Full processor result:")
    result = processor.process(text)
    click.echo(f"   level: {result.level.value}")
    click.echo(f"   success: {result.success}")
    click.echo(f"   params: {result.params}")
    click.echo(f"   ens_code: {result.ens_match.get('code') if result.ens_match else None}")
    click.echo(f"   ens_params: {result.ens_params}")
    click.echo(f"   confidence: {result.confidence:.3f}")
    click.echo(f"   processing_time_ms: {result.processing_time_ms:.1f}")
    if result.details:
        click.echo(f"   details: {result.details}")

    click.echo(f"\n{'=' * 60}")


@cli.group()
def ens():
    """Команды для работы с ЕСН"""
    pass


@ens.command()
@click.argument('excel_file', type=click.Path(exists=True))
@click.option('--output', '-o', required=True, help='Путь для сохранения (.pkl)')
@click.option('--category', '-c', type=click.Choice(['hardware', 'washer', 'rolledmetal']))
def build_index(excel_file, output, category):
    """Построить индекс ЕСН из Excel"""
    from core.integration import build_ens_index

    click.echo(f"📚 Построение индекса из {excel_file}...")
    result_path = build_ens_index(excel_file, output, category)
    click.echo(f"✅ Индекс сохранен: {result_path}")

    meta_path = Path(result_path).with_suffix('.meta.json')
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        click.echo(f"📊 Записей: {meta.get('item_count', 0)}")
        click.echo(f"📊 Категория: {meta.get('category', 'unknown')}")


@ens.command()
@click.argument('query')
@click.option('--index', '-i', required=True, help='Путь к индексу')
@click.option('--top-k', '-k', default=5, help='Количество результатов')
def search(query, index, top_k):
    """Поиск по индексу ЕСН"""
    from ens.indexer import ENSIndex

    if not Path(index).exists():
        click.echo(f"❌ Индекс не найден", err=True)
        return

    ens_index = ENSIndex.load(index)
    results = ens_index.search(query, k=top_k)

    for i, item in enumerate(results, 1):
        score = item.get('_similarity_score', 0)
        name = item.get('полное_наименование') or item.get('наименование', 'N/A')
        click.echo(f"{i}. [{score:.2f}] {name[:60]}...")


@ens.command()
@click.argument('excel_file', type=click.Path(exists=True))
@click.option('--index', '-i', required=True, help='Путь к индексу')
@click.option('--sample', '-s', default=100, help='Размер выборки')
def analyze(excel_file, index, sample):
    """Анализ покрытия файла индексом"""
    from core.integration import analyze_nomenclature

    stats = analyze_nomenclature(excel_file, index, sample_size=sample)

    click.echo(f"📊 Анализ (выборка {sample}):")
    click.echo(f"   Regex разбор: {stats.get('regex_parsed', 0)} ({stats.get('regex_parsed', 0) / sample * 100:.1f}%)")
    click.echo(f"   Требует LLM: {stats.get('failed', 0)} ({stats.get('failed', 0) / sample * 100:.1f}%)")

    if 'estimated_regex_parsed' in stats:
        total = stats.get('total', 0)
        click.echo(f"\n📊 Экстраполяция на {total} записей:")
        click.echo(f"   Ожидается regex: ~{stats['estimated_regex_parsed']}")


@cli.command()
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕСН')
@click.option('--min-score', '-s', default=0.85, help='Порог активации')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM')
@click.option('--limit', '-n', default=0, help='Ограничить число стандартов (0 = все, для отладки)')
@click.option('--standard', help='Генерировать маску только для указанного стандарта (пример: "ГОСТ 7798-70")')
def generate_masks(db, ens_index, min_score, llm, limit, standard):
    """Генерация масок для стандартов из индекса ЕСН"""
    from core.mask_database import MaskDatabase, MaskRecord
    from core.llm_mask_generator import LLMMaskGenerator
    from config.settings import get_settings
    from pathlib import Path
    import pickle

    if not Path(ens_index).exists():
        click.echo("❌ Индекс не найден", err=True)
        return

    with open(ens_index, 'rb') as f:
        data = pickle.load(f)
    items = data.get('items', [])

    # Группировка по стандартам (тип из ЕСН: 'тип_изделия' = 'Наименование типа')
    standards = {}
    for item in items:
        std = item.get('стандарт') or item.get('нтд') or 'UNKNOWN'
        item_type = item.get('тип_изделия') or item.get('наименование_типа') or item.get('тип') or 'unknown'
        key = (std, item_type)
        if key not in standards:
            standards[key] = []
        standards[key].append(item)

    # Фильтр: минимум 10 примеров
    standards = {k: v for k, v in standards.items() if len(v) >= 10}

    # === ФИЛЬТР ПО СТАНДАРТУ ===
    if standard:
        standard_normalized = standard.lower().replace(' ', '')
        matched = {}
        for (std, itype), examples in standards.items():
            std_str = str(std or '')
            std_normalized = std_str.lower().replace(' ', '')
            if standard_normalized in std_normalized or std_normalized in standard_normalized:
                matched[(std, itype)] = examples
        standards = matched
        click.echo(f"🔍 Фильтр по стандарту '{standard}': найдено {len(standards)} пар")
        if not standards:
            click.echo("❌ Ничего не найдено")
            return

    # Ограничение для отладки
    all_items = list(standards.items())
    click.echo(f"🔍 Найдено {len(all_items)} уникальных пар (тип + стандарт) с >=10 примерами")
    if limit and limit > 0:
        standards = dict(all_items[:limit])
        click.echo(f"🔧 Отладочный режим: обрабатываем {len(standards)} пар:")
        for (std, itype), ex_list in all_items[:limit]:
            real_type = ex_list[0].get('тип_изделия') or ex_list[0].get('тип', itype)
            click.echo(f"   - {real_type} / {std} ({len(ex_list)} примеров)")

    mask_db = MaskDatabase(db_path=db)

    settings = get_settings()

    # LLM клиенты
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM генерация включена")

    generator = LLMMaskGenerator(clients=llm_clients, settings=settings, max_retries=3) if llm else None

    stats = {'existing': 0, 'generated': 0, 'activated': 0}

    with click.progressbar(standards.items(), label='Генерация') as bar:
        for (std, item_type), examples in bar:
            existing = mask_db.get_mask(std, item_type)
            if existing and existing.is_active:
                stats['existing'] += 1
                continue

            if generator:
                mask, _ = generator.generate_mask(std, item_type, examples)
                if mask:
                    # Нормализуем item_type в uppercase (стандарты: БОЛТ, ВИНТ, ШАЙБА)
                    item_type_normalized = item_type.upper()
                    temp_mask = MaskRecord(
                        standard=std, item_type=item_type_normalized,
                        pattern=mask['pattern'], params=mask['params'],
                        required=mask['required'], auto_score=0.0,
                        is_active=True, source='llm'  # Активируем сразу
                    )
                    mask_db.save_mask(temp_mask, auto_activate=True, replace_existing=True)
                    stats['generated'] += 1
                    if temp_mask.auto_score >= min_score:
                        stats['activated'] += 1

    click.echo(f"\n📊 Результат:")
    click.echo(f"   Уже активных: {stats['existing']}")
    click.echo(f"   Сгенерировано: {stats['generated']}")
    click.echo(f"   Активировано: {stats['activated']}")


@cli.command()
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД')
@click.option('--threshold', '-t', default=0.5, help='Порог удаления')
def cleanup(db, threshold):
    """Очистка низкокачественных масок"""
    from core.mask_database import MaskDatabase

    mask_db = MaskDatabase(db_path=db)
    deleted = mask_db.cleanup_low_score_masks(threshold)
    click.echo(f"🗑️  Удалено {deleted} масок с score < {threshold}")


if __name__ == '__main__':
    cli()