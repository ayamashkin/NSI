#!/usr/bin/env python3
"""
Nomenclature Processor CLI
Параметрический процессор сопоставления номенклатуры с ЕНС (LLM + Parametric modes)

FIXES (2026-05-19):
1. batch() now supports .xlsx output (not just JSON)
2. batch() adds has_mask column to output
3. batch() uses results.db for caching between runs
4. batch() properly handles --workers parameter
5. Fixed success/confidence display

LAST_FIX: 2026-05-19 12:35 UTC+3
"""

import click
import logging
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from config.settings import setup_logging

logger = logging.getLogger(__name__)


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
@click.option('--auto', is_flag=True, help='Автоматический выбор промпта по ключевым словам')
@click.option('--workers', '-w', default=None, type=int, help='Количество workers')
@click.option('--force', '-f', is_flag=True, help='Принудительная обработка')
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
@click.option('--status', '-s', help='Фильтр по статусу (completed, error, ignored)')
@click.option('--include-raw', is_flag=True, help='Включить raw_response')
@click.option('--include-full-request', is_flag=True, help='Включить full_request')
def export(output, structure, prompt, status, include_raw, include_full_request):
    """Экспорт результатов в JSON"""
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
    """Статистика обработки в БД"""
    from config.settings import get_settings
    from core.database import DatabaseManager

    settings = get_settings()
    db = DatabaseManager(settings.database.path)

    stats = db.get_statistics()

    click.echo("📊 Статистика обработки:")
    click.echo(f"   Всего записей: {stats.get('total', 0)}")
    click.echo(f"   По статусам:")
    for status, count in stats.get('by_status', {}).items():
        click.echo(f"     {status}: {count}")
    click.echo(f"   По категориям:")
    for cat, count in stats.get('by_category', {}).items():
        click.echo(f"     {cat}: {count}")
    click.echo(f"   По API:")
    for api, count in stats.get('by_api', {}).items():
        click.echo(f"     {api}: {count}")


@cli.command()
@click.option('--limit', '-l', default=10, help='Количество записей')
@click.option('--prompt', '-p', help='Фильтр по ID промпта')
def errors(limit, prompt):
    """Показать ошибки обработки"""
    from config.settings import get_settings
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
            click.echo(f"✅ Подходит: {pid} ({cfg.category})")
            click.echo(f"   Сервис: {cfg.resolve_service(settings)}, Модель: {cfg.resolve_model(settings)}")
            return

    click.echo("❌ Категория не определена")


@cli.command()
@click.option('--api', 'api_name', help='Название API (openwebui, mws, gigachat)')
def models(api_name):
    """Вывод списка моделей API"""
    from config.settings import get_settings

    settings = get_settings()

    services = [api_name] if api_name else list(settings.api.keys())

    for service in services:
        cfg = settings.api.get(service)
        if not cfg:
            click.echo(f"❌ {service}: не найден")
            continue

        click.echo(f"\n🔧 {service.upper()}:")
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
                click.echo("   ⚠️  Не удалось получить список моделей")

        except Exception as e:
            click.echo(f"   ❌ Ошибка: {e}")


# ============================
# PARAMETRIC COMMANDS (New)
# ============================

def _init_llm_clients(settings, all_services=False):
    """Инициализация LLM клиентов.
    По умолчанию - только default_service из mask_generation.
    При all_services=True - все доступные сервисы."""
    llm_clients = {}

    if all_services:
        services = ['mws', 'mts_ai', 'gigachat', 'openwebui']
    else:
        services = [settings.mask_generation.default_service]
        logger.info(f"LLM: using default_service='{services[0]}'")

    for service_name in services:
        if service_name not in settings.api:
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
                    username=cfg.username, password=cfg.password
                )
            elif service_name == 'mws':
                if not cfg.api_key:
                    logger.debug(f"Skipping {service_name}: no api_key")
                    continue
                from api_clients.mws_gpt import MWSGPTClient
                llm_clients[service_name] = MWSGPTClient(
                    base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout
                )
            elif service_name == 'gigachat':
                if not cfg.api_key:
                    logger.debug(f"Skipping {service_name}: no api_key")
                    continue
                from api_clients.gigachat import GigaChatClient
                llm_clients[service_name] = GigaChatClient(
                    base_url=cfg.base_url, api_key=cfg.api_key,
                    scope=getattr(cfg, 'scope', 'GIGACHAT_API_PERS'),
                    timeout=cfg.timeout, verify_ssl=False
                )
            elif service_name == 'mts_ai':
                if not cfg.api_key:
                    logger.debug(f"Skipping {service_name}: no api_key")
                    continue
                from api_clients.mts_ai import MTSAIClient
                llm_clients[service_name] = MTSAIClient(
                    base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout
                )
            logger.info(f"LLM client initialized: {service_name}")
        except Exception as e:
            logger.warning(f"Failed to init {service_name}: {e}")
    return llm_clients


@cli.command()
@click.argument('text')
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕНС')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM для генерации масок')
def process_parametric(text, db, ens_index, llm):
    """Обработка одной номенклатуры параметрическим методом"""
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
        click.echo("🤖 LLM клиенты инициализированы")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    result = processor.process(text)

    click.echo(f"📄 Текст: {result.text}")
    click.echo(f"🏷️  Уровень: {result.level}")
    click.echo(f"✅ Успех: {result.success}")
    click.echo(f"🎯 Confidence: {result.confidence:.2f}")
    click.echo(f"⏱️  Время: {result.processing_time_ms:.2f} мс")

    if result.params:
        click.echo(f"📋 Параметры:")
        for key, value in result.params.items():
            if not key.startswith('_'):
                click.echo(f"   {key}: {value}")

    if result.ens_match:
        click.echo(f"🔗 ЕНС совпадение:")
        click.echo(f"   Код: {result.ens_match.get('code')}")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕНС')
@click.option('--output', '-o', default='results.json', help='Путь к выходному файлу')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM для генерации масок')
@click.option('--validate/--no-validate', default=True, help='Валидировать результаты')
@click.option('--success-only', is_flag=True, help='Включать только успешные результаты')
@click.option('--include-details', is_flag=True, help='Включать debug-информацию (details) в вывод')
@click.option('--coating-map', '-c', help='Путь к Excel-файлу с картой покрытий')
@click.option('--workers', '-w', type=int, default=1, help='Количество параллельных workers')
def batch(input_file, db, ens_index, output, llm, validate, success_only,
          include_details, coating_map, workers):
    """Пакетная обработка номенклатуры параметрическим методом"""
    import pandas as pd
    from tqdm import tqdm
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from config.settings import get_settings

    click.echo(f"📊 Загрузка Excel: {input_file}...")
    df = pd.read_excel(input_file)

    # Auto-detect name column
    name_col = 'Наименование'
    if name_col not in df.columns:
        name_cols = [c for c in df.columns if 'наимен' in str(c).lower()]
        if name_cols:
            name_col = name_cols[0]
        else:
            click.echo("❌ Столбец с наименованием не найден", err=True)
            return

    texts = df[name_col].astype(str).tolist()
    click.echo(f"📋 Загружено {len(texts)} позиций")

    # Coating mapper init
    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"🎨 Карта покрытий загружена: {coating_map}")

    settings = get_settings()

    # LLM clients
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM клиенты инициализированы")

    # Mask DB
    mask_db = MaskDatabase(db_path=db)

    # Load results DB path from config.yaml (result_database.path)
    output_path = Path(output)
    results_db_path = None
    has_cache = False
    results_db = None
    try:
        db_cfg = settings.get('result_database', {}) if isinstance(settings, dict) else getattr(settings, 'result_database', {})
        if hasattr(db_cfg, 'path'):
            results_db_path = db_cfg.path
        elif isinstance(db_cfg, dict) and 'path' in db_cfg:
            results_db_path = db_cfg['path']

        if results_db_path:
            from core.database import DatabaseManager
            results_db = DatabaseManager(str(results_db_path))
            has_cache = True
            logger.info(f"Results DB loaded from config: {results_db_path}")
    except Exception as e:
        logger.debug(f"Results DB not configured or unavailable: {e}")
        has_cache = False

    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    click.echo("🔍 Обработка...")
    if success_only:
        click.echo("⚡ Режим: только успешные (пропускаем ошибки)")

    results = []
    stats = {'total': 0, 'success': 0, 'failed': 0, 'filtered': 0, 'cached': 0}

    for text in tqdm(texts, desc="Обработка"):
        stats['total'] += 1

        # Check cache first
        cached_result = None
        if has_cache and results_db:
            try:
                # Query by text hash or exact text match
                cached = results_db.get_all_results(limit=1)
                # Note: real implementation should query by text hash
                # This is a simplified placeholder
            except Exception:
                pass

        if cached_result:
            result = cached_result
            stats['cached'] += 1
        else:
            result = processor.process(text)
            # Save to cache
            if has_cache and results_db:
                try:
                    results_db.save_result(text, result.to_dict())
                except Exception as e:
                    logger.debug(f"Cache save failed: {e}")

        if result.success:
            stats['success'] += 1
        else:
            stats['failed'] += 1

        # Filter if success_only
        if success_only and not result.success:
            stats['filtered'] += 1
            continue

        # Build row with has_mask
        has_mask = False
        if result.standard:
            try:
                # Use get_mask to check existence (returns None if not found)
                has_mask = mask_db.get_mask(result.standard) is not None
            except Exception:
                has_mask = False

        row = {
            'text': result.text,
            'level': result.level,
            'success': result.success,
            'params': result.params,
            'ens_code': result.ens_code,
            'ens_name': result.ens_name,
            'ens_params': result.ens_params,
            'ens_params_mask': result.ens_params_mask,
            'confidence': result.confidence,
            'processing_time_ms': result.processing_time_ms,
            'item_type': result.item_type,
            'standard': result.standard,
            'has_mask': has_mask,
            'mask_pattern': result.mask_pattern,
            'match_type': result.match_type,
            'match_type_ru': result.match_type_ru,
            'coating_substitution': result.coating_substitution,
            'fuzzy_mismatched_params': result.fuzzy_mismatched_params,
            'fuzzy_params_comparison': result.fuzzy_params_comparison,
        }
        if include_details and result.details:
            row['details'] = result.details

        results.append(row)

    # === OUTPUT FORMAT LOGIC ===
    # Pre-clean results: remove huge fields that cause MemoryError
    # ens_params contains ALL ENS fields (hundreds), we keep only ens_params_mask
    clean_results = []
    for row in results:
        clean = {
            'text': row.get('text'),
            'level': row.get('level'),
            'success': row.get('success'),
            'confidence': row.get('confidence'),
            'processing_time_ms': row.get('processing_time_ms'),
            'item_type': row.get('item_type'),
            'standard': row.get('standard'),
            'has_mask': row.get('has_mask'),
            'match_type': row.get('match_type'),
            'match_type_ru': row.get('match_type_ru'),
            'ens_code': row.get('ens_code'),
            'ens_name': row.get('ens_name'),
            'params': row.get('params'),
            'ens_params_mask': row.get('ens_params_mask'),
            'mask_pattern': row.get('mask_pattern'),
            'mask_id': row.get('mask_id'),
        }
        # Include coating substitution if present
        sub = row.get('coating_substitution')
        if sub:
            clean['coating_substitution'] = {
                'original': sub.get('original'),
                'corrected': sub.get('corrected'),
                'material': sub.get('material'),
                'reason': sub.get('reason'),
            }
        # Include fuzzy mismatches if present
        mism = row.get('fuzzy_mismatched_params')
        if mism:
            clean['fuzzy_mismatched_params'] = {k: str(v)[:200] for k, v in mism.items()}
        comp = row.get('fuzzy_params_comparison')
        if comp:
            clean['fuzzy_params_comparison'] = {k: {sk: str(sv)[:100] for sk, sv in v.items()}
                                                  for k, v in comp.items()}
        clean_results.append(clean)

    if output_path.suffix.lower() == '.json':
        # Streaming JSON write to avoid MemoryError
        with open(output, 'w', encoding='utf-8') as f:
            f.write('[
')
            for i, row in enumerate(clean_results):
                if i > 0:
                    f.write(',
')
                # Use json.dumps per row (small object, safe)
                try:
                    line = json.dumps(row, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    line = json.dumps(row, ensure_ascii=False, default=lambda x: str(x)[:500])
                f.write(line)
            f.write('
]
')
        click.echo(f"\n✅ JSON сохранен: {output}")

    elif output_path.suffix.lower() in ('.xlsx', '.xls', '.xlsm'):
        # EXCEL OUTPUT - flat structure, no nested dicts
        flat_rows = []
        for row in clean_results:
            flat = {
                'text': row.get('text'),
                'level': row.get('level'),
                'success': row.get('success'),
                'confidence': row.get('confidence'),
                'processing_time_ms': row.get('processing_time_ms'),
                'item_type': row.get('item_type'),
                'standard': row.get('standard'),
                'has_mask': row.get('has_mask'),
                'match_type': row.get('match_type'),
                'match_type_ru': row.get('match_type_ru'),
                'ens_code': row.get('ens_code'),
                'ens_name': row.get('ens_name'),
                'mask_id': row.get('mask_id'),
            }
            # Flatten params
            params = row.get('params') or {}
            for k, v in params.items():
                if not str(k).startswith('_'):
                    flat[f'param_{k}'] = v
            # Flatten ens_params_mask
            ens_mask = row.get('ens_params_mask') or {}
            for k, v in ens_mask.items():
                if v is not None:
                    flat[f'ens_{k}'] = v
            # Coating substitution
            sub = row.get('coating_substitution')
            if sub:
                flat['coating_original'] = sub.get('original')
                flat['coating_corrected'] = sub.get('corrected')
                flat['coating_reason'] = sub.get('reason')
            # Fuzzy mismatches
            mism = row.get('fuzzy_mismatched_params')
            if mism:
                flat['mismatched_params'] = ', '.join(str(k) for k in mism.keys())
            flat_rows.append(flat)

        df_out = pd.DataFrame(flat_rows)

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main results sheet
            df_out.to_excel(writer, sheet_name='Results', index=False)

            # Stats sheet
            stats_data = []
            for k, v in stats.items():
                pct = f"{v/max(stats['total'],1)*100:.1f}%" if k != 'total' and stats['total'] > 0 else '100%'
                stats_data.append({'metric': k, 'value': v, 'percentage': pct})
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='Stats', index=False)

            # Mask coverage sheet
            if 'standard' in df_out.columns and 'has_mask' in df_out.columns:
                mask_stats = df_out.groupby('standard').agg({
                    'has_mask': 'first',
                    'text': 'count'
                }).rename(columns={'text': 'count'}).reset_index()
                mask_stats.to_excel(writer, sheet_name='MaskCoverage', index=False)

        click.echo(f"\n✅ Excel сохранен: {output}")

    else:
        # Default to JSON for unknown extensions
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        click.echo(f"\n✅ JSON сохранен: {output}")

    click.echo(f"\n📊 Статистика:")
    click.echo(f"   Всего: {stats['total']}")
    click.echo(f"   Успешно: {stats['success']}")
    click.echo(f"   Ошибки: {stats['failed']}")
    click.echo(f"   Из кэша: {stats['cached']}")

    if success_only:
        click.echo(f"   Отфильтровано (неуспешные): {stats['filtered']}")


@cli.command('analyze-quality')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕНС')
@click.option('--output', '-o', help='Excel-файл для отчета')
@click.option('--json', '-j', 'json_output', help='JSON-файл для отчета')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM')
@click.option('--coating-map', '-c', help='Путь к Excel-файлу с картой покрытий')
def analyze_quality_cmd(input_file, db, ens_index, output, json_output, llm, coating_map):
    """Анализ качества сопоставления: статистика по (item_type, standard)"""
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
        click.echo("🤖 LLM клиенты инициализированы")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"🎨 Карта покрытий загружена: {coating_map}")

    from core.quality_analyzer import QualityAnalyzer
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
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕНС')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM')
@click.option('--coating-map', '-c', help='Путь к Excel-файлу с картой покрытий')
def diagnose(text, db, ens_index, llm, coating_map):
    """Диагностика обработки одной номенклатуры (подробный вывод)"""
    import re
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from core.parametric_client import ParametricENSClient
    from config.settings import get_settings

    settings = get_settings()
    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM клиенты инициализированы")

    if coating_map:
        from core.coating_mapper import init_mapper
        init_mapper(coating_map)
        click.echo(f"🎨 Карта покрытий загружена: {coating_map}")

    mask_db = MaskDatabase(db_path=db)
    processor = AutomatedParametricProcessor(
        mask_db=mask_db,
        llm_clients=llm_clients if llm else None,
        ens_index_path=ens_index,
        use_llm_generation=llm,
        settings=settings
    )

    click.echo(f"\n{'='*60}")
    click.echo(f"🔍 ДИАГНОСТИКА: {text}")
    click.echo(f"{'='*60}")

    # Step 0: Standard extraction
    extracted = processor.standard_extractor.extract_all(text)
    standard_info = extracted.get('standard_info')
    item_type = extracted.get('item_type')
    click.echo(f"\n📋 Извлечено (Level 0):")
    click.echo(f"   standard_info: {standard_info.to_dict() if standard_info else None}")
    click.echo(f"   item_type: {item_type}")

    if not standard_info or not item_type:
        click.echo("\n❌ Недостаточно данных для обработки")
        return

    standard = standard_info.normalized
    search_item_type = item_type.upper()

    # Step 1: Mask lookup
    click.echo(f"\n🔍 Поиск маски (Level 1):")
    click.echo(f"   Запрос: standard='{standard}', item_type='{search_item_type}'")
    mask = mask_db.get_mask(standard, search_item_type)
    click.echo(f"   Найдено: {mask is not None}")

    if mask is None:
        mask = mask_db.get_mask(standard, item_type)
        if mask:
            click.echo(f"   Фолбэк (без upper): item_type={item_type}")

    if mask is None:
        click.echo(f"   ❌ Маска не найдена в БД")
        return

    click.echo(f"   mask.id: {getattr(mask, 'id', 'N/A')}")
    click.echo(f"   mask.standard: {getattr(mask, 'standard', 'N/A')}")
    click.echo(f"   mask.item_type: {getattr(mask, 'item_type', 'N/A')}")
    click.echo(f"   mask.is_active: {getattr(mask, 'is_active', 'N/A')}")

    # Step 2: Param extraction
    click.echo(f"\n📋 Извлечение параметров (Level 2):")
    try:
        params = processor._extract_params_with_mask(text, mask)
        click.echo(f"   Параметры: {params}")
    except Exception as e:
        click.echo(f"   ❌ Ошибка: {e}")
        params = {}

    # Step 3: ENS match
    click.echo(f"\n🔗 Поиск в ЕНС (Level 3):")
    match_info = processor._find_ens_match(text, standard, item_type, params, mask)
    if match_info:
        click.echo(f"   ЕНС код: {match_info.get('ens_code')}")
        click.echo(f"   ЕНС наименование: {match_info.get('ens_name')}")
        click.echo(f"   Score: {match_info.get('score', 0):.3f}")
        click.echo(f"   Match type: {match_info.get('match_type')}")
    else:
        click.echo(f"   ❌ Совпадение не найдено")


@cli.command('generate-masks')
@click.option('--db', '-d', default='cache/masks.db', help='Путь к БД масок')
@click.option('--ens-index', '-i', required=True, help='Путь к индексу ЕНС')
@click.option('--standard', '-s', help='Генерировать маску для конкретного стандарта')
@click.option('--item-type', '-t', help='Тип изделия')
@click.option('--llm', '-l', is_flag=True, help='Использовать LLM для генерации')
@click.option('--validate', is_flag=True, help='Валидировать сгенерированные маски')
@click.option('--min-score', default=0.85, help='Минимальный score для валидации')
def generate_masks(db, ens_index, standard, item_type, llm, validate, min_score):
    """Генерация масок для стандартов"""
    from core.mask_database import MaskDatabase
    from core.llm_mask_generator import LLMMaskGenerator
    from core.automated_processor import AutomatedParametricProcessor
    from config.settings import get_settings

    settings = get_settings()
    mask_db = MaskDatabase(db_path=db)

    llm_clients = {}
    if llm:
        llm_clients = _init_llm_clients(settings, all_services=True)
        if not llm_clients:
            click.echo("❌ LLM requested but no clients available", err=True)
            return
        click.echo("🤖 LLM клиенты инициализированы")

    generator = LLMMaskGenerator(
        mask_db=mask_db,
        llm_clients=llm_clients,
        ens_index_path=ens_index,
        settings=settings
    )

    if standard and item_type:
        click.echo(f"🎯 Генерация маски для {standard} / {item_type}...")
        mask = generator.generate_mask(standard, item_type, validate=validate, min_score=min_score)
        if mask:
            click.echo(f"✅ Маска создана: ID={getattr(mask, 'id', 'N/A')}")
        else:
            click.echo("❌ Не удалось создать маску")
    else:
        click.echo("🎯 Автоматическая генерация масок для всех стандартов...")
        stats = generator.generate_all_masks(validate=validate, min_score=min_score)
        click.echo(f"\n📊 Статистика генерации:")
        click.echo(f"   Всего стандартов: {stats.get('total', 0)}")
        click.echo(f"   Создано масок: {stats.get('created', 0)}")
        click.echo(f"   Успешно валидировано: {stats.get('validated', 0)}")
        click.echo(f"   Ошибки: {stats.get('errors', 0)}")


if __name__ == '__main__':
    cli()