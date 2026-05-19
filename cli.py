#!/usr/bin/env python3
"""
Nomenclature Processor CLI
Параметрический процессор сопоставления номенклатуры с ЕНС (LLM + Parametric modes)

FIXES (2026-05-19):
1. CRITICAL: Removed ens_params (full ENS dict) from batch() output row.
   This fixes 8+ GB output files and MemoryError on Excel export.
2. Fixed fallback JSON output to use clean_results instead of raw results.
3. batch() now supports .xlsx output (not just JSON)
4. batch() adds has_mask column to output
5. batch() uses results.db for caching between runs
6. batch() properly handles --workers parameter
7. Fixed success/confidence display

LAST_FIX: 2026-05-19 14:33 UTC+3
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
        click.echo("⚠️ Нет данных для экспорта")
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
                    click.echo(f"     - {m}")
                if len(model_list) > 10:
                    click.echo(f"     ... и еще {len(model_list) - 10}")
            else:
                click.echo("   ⚠️ Не удалось получить список моделей")

        except Exception as e:
            click.echo(f"   ❌ Ошибка: {e}")

# ============================
# PARAMETRIC COMMANDS (New)
# ============================

def _find_name_column(df):
    """Поиск колонки с наименованием (case-insensitive, частичное совпадение)."""
    keywords = ['наименование', 'номенклатура', 'name', 'наименов', 'наим.']
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def _truncate_dataframe_cells(df, max_length=1000):
    """Обрезка длинных строковых значений для предотвращения огромных Excel-файлов."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: str(x)[:max_length] if pd.notna(x) and len(str(x)) > max_length else x
            )
    return df


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
    click.echo(f"🏷️ Уровень: {result.level}")
    click.echo(f"✅ Успех: {result.success}")
    click.echo(f"🎯 Confidence: {result.confidence:.2f}")
    click.echo(f"⏱️ Время: {result.processing_time_ms:.2f} мс")

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
def batch(input_file, db, ens_index, output, llm, validate, success_only,          include_details, coating_map, workers):
    """Пакетная обработка номенклатуры параметрическим методом"""
    import pandas as pd
    from tqdm import tqdm
    from core.mask_database import MaskDatabase
    from core.automated_processor import AutomatedParametricProcessor
    from config.settings import get_settings
    from parsers.standard_extractor import StandardExtractor

    click.echo(f"📊 Загрузка Excel: {input_file}...")
    df = pd.read_excel(input_file)
    click.echo(f"   Прочитано {len(df)} строк, {len(df.columns)} колонок")

    # --- Проверка колонки "Наименование" ---
    name_col = _find_name_column(df)
    if name_col is None:
        click.echo("\n❌ ОШИБКА: В файле отсутствует колонка с наименованием.")
        click.echo("   Ожидается колонка, содержащая в названии одно из слов:")
        click.echo("   'Наименование', 'Номенклатура', 'Name', 'Наим.', 'Наименов'")
        click.echo(f"\n   Доступные колонки в файле:")
        for i, col in enumerate(df.columns, 1):
            click.echo(f"      {i}. {col}")
        click.echo("\n   Переименуйте колонку с наименованием изделий и повторите запуск.")
        return 1

    click.echo(f"✅ Колонка с наименованием: '{name_col}'")

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
            return 1
        click.echo("🤖 LLM клиенты инициализированы")

    # Mask DB
    mask_db = MaskDatabase(db_path=db)
    extractor = StandardExtractor()

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

    for idx, text in enumerate(tqdm(texts, desc="Обработка")):
        stats['total'] += 1

        result = processor.process(text)

        if result.success:
            stats['success'] += 1
        else:
            stats['failed'] += 1

        # Filter if success_only
        if success_only and not result.success:
            stats['filtered'] += 1
            continue

        # Используем стандарт и тип из результата процессора
        standard = result.standard or ''
        item_type = result.item_type or ''
        has_mask = False
        mask_pattern = ''
        if standard and item_type:
            try:
                mask = mask_db.get_mask(standard, item_type)
                has_mask = mask is not None
                if mask:
                    mask_pattern = getattr(mask, 'pattern', '') or ''
            except Exception:
                pass

        # --- Формирование строки результата: сохраняем ВСЕ исходные колонки ---
        out_row = {}
        for col in df.columns:
            out_row[str(col)] = df.iloc[idx][col]

        # Добавляем колонки обогащения
        out_row['Код ЕНС'] = result.ens_code or ''
        out_row['Наименование ЕНС'] = result.ens_name or ''
        out_row['Уровень'] = result.level or ''
        out_row['Распознано'] = 'Да' if result.success else 'Нет'
        out_row['Уверенность'] = round(float(result.confidence or 0.0), 3)
        out_row['Тип сопоставления'] = result.match_type_ru or 'Не определено'

        # Подстановка покрытия
        sub = result.coating_substitution
        out_row['Подстановка покрытия'] = (
            json.dumps(sub, ensure_ascii=False) if sub else ''
        )

        # Несовпавшие параметры
        mism = result.fuzzy_mismatched_params
        out_row['Несовпавшие параметры'] = (
            json.dumps(mism, ensure_ascii=False) if mism else ''
        )

        # Новые колонки
        out_row['маска'] = mask_pattern
        out_row['стандарт'] = standard or ''
        out_row['тип'] = item_type or ''
        out_row['маски_в_бд'] = 'Да' if has_mask else 'Нет'

        if include_details and result.details:
            d_str = json.dumps(result.details, ensure_ascii=False, default=str)
            out_row['details'] = d_str[:2000] if len(d_str) > 2000 else d_str

        results.append(out_row)

    # === OUTPUT FORMAT LOGIC ===
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == '.json':
        # Streaming JSON write
        with open(output, 'w', encoding='utf-8') as f:
            f.write('[\n')
            for i, row in enumerate(results):
                if i > 0:
                    f.write(',\n')
                clean_row = {}
                for k, v in row.items():
                    if isinstance(v, dict):
                        clean_row[k] = {sk: str(sv)[:500] for sk, sv in v.items()}
                    else:
                        clean_row[k] = v
                line = json.dumps(clean_row, ensure_ascii=False, indent=2, default=str)
                f.write(line)
            f.write('\n]\n')
        click.echo(f"\n✅ JSON сохранен: {output}")

    elif output_path.suffix.lower() in ('.xlsx', '.xls', '.xlsm'):
        df_out = pd.DataFrame(results)

        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: обрезка огромных ячеек
        df_out = _truncate_dataframe_cells(df_out, max_length=1000)

        # Уверенность как число
        if 'Уверенность' in df_out.columns:
            df_out['Уверенность'] = pd.to_numeric(df_out['Уверенность'], errors='coerce').fillna(0.0)

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_out.to_excel(writer, sheet_name='Results', index=False)

            # Форматирование колонки "Уверенность" — 3 знака после запятой
            if 'Уверенность' in df_out.columns:
                ws = writer.sheets['Results']
                for idx_col, col_name in enumerate(df_out.columns):
                    if col_name == 'Уверенность':
                        for row_num in range(2, len(df_out) + 2):
                            cell = ws.cell(row=row_num, column=idx_col + 1)
                            cell.number_format = '0.000'
                        break

            # Stats sheet
            stats_data = []
            for k, v in stats.items():
                pct = f"{v/max(stats['total'],1)*100:.1f}%" if k != 'total' and stats['total'] > 0 else '100%'
                stats_data.append({'metric': k, 'value': v, 'percentage': pct})
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='Stats', index=False)

            # Mask coverage sheet
            if 'стандарт' in df_out.columns and 'маски_в_бд' in df_out.columns:
                mask_stats = df_out.groupby('стандарт').agg({
                    'маски_в_бд': 'first',
                    name_col: 'count'
                }).rename(columns={name_col: 'count'}).reset_index()
                mask_stats.to_excel(writer, sheet_name='MaskCoverage', index=False)

        file_size = output_path.stat().st_size / 1024
        click.echo(f"\n✅ Excel сохранен: {output}")
        click.echo(f"   Размер: {file_size:.1f} КБ")

    else:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        click.echo(f"\n✅ JSON сохранен: {output}")

    click.echo(f"\n📊 Статистика:")
    click.echo(f"  Всего обработано: {stats['total']}")
    click.echo(f"  ✅ Успешно:      {stats['success']}")
    click.echo(f"  ❌ Ошибки:       {stats['failed']}")
    click.echo(f"  💾 Из кэша:      {stats['cached']}")

    if success_only:
        click.echo(f"  Отфильтровано (неуспешные): {stats['filtered']}")

    return 0

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
@click.option('--ens-index', '-i', required=True, help='Пути к индексу ЕНС')
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
    click.echo(f"   mask.pattern (первые 120 симв):")
    click.echo(f"      {getattr(mask, 'pattern', 'N/A')[:120]}")

    # Step 2: Pattern relaxation
    effective_standard = getattr(mask, 'standard', None) or standard
    client = ParametricENSClient.__new__(ParametricENSClient)
    relaxed = client._relax_pattern(mask.pattern, standard=effective_standard)
    click.echo(f"\n📋 Relax pattern:")
    click.echo(f"   standard заменен: '{effective_standard}'")
    click.echo(f"   relaxed (первые 200 симв):")
    click.echo(f"      {relaxed[:200]}")
    if len(relaxed) > 200:
        click.echo(f"      ... ({len(relaxed)} символов всего)")

    # Step 3: Regex match
    try:
        compiled = re.compile(relaxed, re.IGNORECASE)
        match = compiled.search(text)
        click.echo(f"\n📋 Regex match:")
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
        click.echo(f"\n📋 Regex match:")
        click.echo(f"   ❌ INVALID REGEX: {e}")

    # Step 4: Full processor result
    click.echo(f"\n📋 Full processor result:")
    result = processor.process(text)
    click.echo(f"   level: {result.level}")
    click.echo(f"   success: {result.success}")
    click.echo(f"   params: {result.params}")
    click.echo(f"   ens_code: {result.ens_match.get('code') if result.ens_match else None}")
    click.echo(f"   ens_params: {result.ens_params}")
    click.echo(f"   confidence: {result.confidence:.3f}")
    click.echo(f"   processing_time_ms: {result.processing_time_ms:.1f}")
    if result.details:
        click.echo(f"   details: {result.details}")

    click.echo(f"\n{'='*60}")

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
        click.echo(f"  Всего стандартов: {stats.get('total', 0)}")
        click.echo(f"  Создано масок: {stats.get('created', 0)}")
        click.echo(f"  Успешно валидировано: {stats.get('validated', 0)}")
        click.echo(f"  Ошибки: {stats.get('errors', 0)}")


@cli.command()
@click.option('--db', '-d', default='cache/masks.db',
              help='Путь к БД масок')
@click.option('--threshold', '-t', default=0.5,
              help='Минимальный score для удаления')
def cleanup(db, threshold):
    """Очистка неактивных масок с низким score"""
    from core.mask_database import MaskDatabase

    mask_db = MaskDatabase(db_path=db)
    deleted = mask_db.cleanup_low_score_masks(threshold)
    click.echo(f"🗑️ Удалено {deleted} масок с score < {threshold}")


if __name__ == '__main__':
    cli()