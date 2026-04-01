#!/usr/bin/env python3
"""
Nomenclature Processor CLI
Управление обработкой номенклатуры через командную строку.
"""

import re
import click
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Только базовые импорты без pandas
from config.settings import get_settings, reload_settings
from core.database import DatabaseManager
from core.processor import NomenclatureProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Nomenclature Processor - система обработки номенклатуры с помощью LLM"""
    pass


@cli.command()
def prompts():
    """Список доступных промптов (не требует pandas/numpy)"""
    settings = get_settings()

    click.echo("📚 Доступные промпты:")
    for pid, cfg in settings.prompts.items():
        click.echo(f"\n  {pid}:")
        click.echo(f"    Название: {cfg.name}")
        click.echo(f"    Категория: {cfg.category}")
        click.echo(f"    Сервис: {cfg.service}")
        click.echo(f"    Модель: {cfg.model}")
        click.echo(f"    Ключевые слова: {', '.join(cfg.keywords)}")


def _check_category_match(name: str, prompt_cfg) -> bool:
    """
    Проверка соответствия категории по ключевым словам.
    Поддерживает обычные строки, регулярные выражения и glob-шаблоны.
    (Копия метода из processor.py для использования в CLI)
    """
    name_lower = name.lower()

    for keyword in prompt_cfg.keywords:
        keyword = keyword.strip()

        # Проверяем, является ли keyword регулярным выражением
        if keyword.startswith('regex:') or keyword.startswith('re:'):
            # Извлекаем паттерн
            pattern = keyword.split(':', 1)[1].strip()
            try:
                if re.search(pattern, name, re.IGNORECASE):
                    return True
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue

        # Проверяем как glob-шаблон (с поддержкой wildcard * и ?)
        elif '*' in keyword or '?' in keyword:
            # Конвертируем glob в regex
            pattern = keyword.replace('.', r'\.').replace('*', '.*').replace('?', '.')
            try:
                if re.search(pattern, name_lower):
                    return True
            except re.error:
                continue

        # Простое вхождение подстроки
        else:
            if keyword.lower() in name_lower:
                return True

    return False


@cli.command()
@click.argument('excel_path', type=click.Path(exists=True))
@click.option('--prompt', '-p', multiple=True, help='ID промпта (можно несколько)')
@click.option('--auto', '-a', is_flag=True, help='Автоопределение промптов')
@click.option('--api', type=click.Choice(['openwebui', 'mws']), default=None)
@click.option('--workers', '-w', default=None, type=int)
@click.option('--force', '-f', is_flag=True)
def process(excel_path, prompt, auto, api, workers, force):
    """Обработка Excel файла"""
    # ЛЕНИВЫЙ ИМПОРТ pandas и ExcelLoader
    try:
        import pandas as pd
        from utils.excel_loader import ExcelLoader, NomenclatureItem
    except ImportError as e:
        click.echo(f"❌ Ошибка импорта pandas: {e}", err=True)
        click.echo("💡 Установите: pip install pandas openpyxl", err=True)
        return

    settings = get_settings()

    # Проверка API клиентов
    if api:
        if api not in settings.api:
            click.echo(f"❌ API '{api}' не настроен в config.yaml", err=True)
            return

        # Фильтруем промпты по указанному API
        available_prompts = {
            pid: cfg for pid, cfg in settings.prompts.items()
            if cfg.service == api
        }

        # Проверяем, есть ли запрошенные промпты с другим сервисом
        if prompt:
            mismatched = []
            for pid in prompt:
                if pid in settings.prompts and settings.prompts[pid].service != api:
                    mismatched.append((pid, settings.prompts[pid].service))

            if mismatched:
                click.echo("❌ Несоответствие сервиса API:", err=True)
                for pid, svc in mismatched:
                    click.echo(f"   Промпт '{pid}' использует сервис '{svc}', но выбран '{api}'", err=True)
                click.echo(f"\n💡 Варианты:", err=True)
                click.echo(f"   1. Используйте --api {mismatched[0][1]}", err=True)
                click.echo(f"   2. Или не указывайте --api для использования всех промптов", err=True)
                return

        if not available_prompts:
            all_services = set(cfg.service for cfg in settings.prompts.values())
            click.echo(f"❌ Нет промптов для сервиса '{api}'", err=True)
            click.echo(f"💡 Доступные сервисы в промптах: {', '.join(all_services)}", err=True)
            return
    else:
        available_prompts = settings.prompts

    if not available_prompts:
        click.echo("❌ Нет доступных промптов", err=True)
        return

    click.echo(f"📊 Загрузка данных из {excel_path}...")

    try:
        df = pd.read_excel(excel_path)
        items = [
            NomenclatureItem(
                article=str(row['артикул']),
                name=str(row['Краткое наименование']),
                guid=str(row['GUID'])
            )
            for _, row in df.iterrows()
        ]
    except KeyError as e:
        click.echo(f"❌ Отсутствует колонка в Excel: {e}", err=True)
        click.echo("💡 Ожидаемые колонки: артикул, Краткое наименование, GUID", err=True)
        return
    except Exception as e:
        click.echo(f"❌ Ошибка загрузки Excel: {e}", err=True)
        return

    click.echo(f"✅ Загружено {len(items)} позиций")

    # Инициализация и обработка
    db = DatabaseManager()
    processor = NomenclatureProcessor(db, max_workers=workers)

    if auto:
        results = processor.auto_process(items, force_reprocess=force)
    else:
        prompt_ids = list(prompt) if prompt else list(available_prompts.keys())
        # Проверяем существование промптов
        invalid = set(prompt_ids) - set(available_prompts.keys())
        if invalid:
            click.echo(f"❌ Неизвестные промпты: {', '.join(invalid)}", err=True)
            return

        results = processor.process_batch(items, prompt_ids, force_reprocess=force)

    # Статистика
    stats = {'completed': 0, 'ignored': 0, 'error': 0}
    for r in results:
        status = r.get('status', 'unknown')
        stats[status] = stats.get(status, 0) + 1

    click.echo(f"\n✅ Завершено: {stats.get('completed', 0)}")
    click.echo(f"⏭️  Пропущено: {stats.get('ignored', 0)}")
    click.echo(f"❌ Ошибок: {stats.get('error', 0)}")

    if stats.get('error', 0) > 0:
        click.echo(f"\n💡 Просмотр ошибок: python cli.py errors")


@cli.command()
@click.option('--output', '-o', default='results.json', help='Путь для сохранения')
@click.option('--structure', '-s', type=click.Choice(['flat', 'by_code', 'by_category', 'by_prompt']),
              default='flat', help='Структура JSON')
def export(output, structure):
    """Экспорт результатов в JSON"""
    db = DatabaseManager()
    path = db.export_to_json(output, structure)
    click.echo(f"💾 Экспортировано в: {path}")


@cli.command()
def stats():
    """Статистика обработки"""
    db = DatabaseManager()
    statistics = db.get_statistics()

    click.echo("📈 Статистика обработки:")
    click.echo(f"  Всего записей: {statistics.get('total', 0)}")

    click.echo("\n  По статусам:")
    for status, count in statistics.get('by_status', {}).items():
        click.echo(f"    {status}: {count}")

    click.echo("\n  По категориям:")
    for cat, count in statistics.get('by_category', {}).items():
        click.echo(f"    {cat}: {count}")

    click.echo("\n  По сервисам API:")
    for svc, count in statistics.get('by_api', {}).items():
        click.echo(f"    {svc}: {count}")


@cli.command()
@click.option('--limit', '-l', default=10, help='Количество ошибок для показа')
@click.option('--prompt', '-p', default=None, help='Фильтр по ID промпта')
def errors(limit, prompt):
    """Показать последние ошибки обработки"""
    db = DatabaseManager()

    # Получаем ошибки из БД
    error_results = db.get_all_results(status='error', prompt_id=prompt, limit=limit)

    if not error_results:
        click.echo("✅ Ошибок не найдено")
        return

    click.echo(f"❌ Последние {len(error_results)} ошибок:\n")

    for r in error_results:
        click.echo(f"Артикул: {r['article']}")
        click.echo(f"Наименование: {r['name']}")
        click.echo(f"Промпт: {r['prompt_id']}")
        click.echo(f"Сервис: {r.get('api_source', 'N/A')}")
        click.echo(f"Модель: {r.get('model_used', 'N/A')}")

        error_msg = r.get('error_message', 'Неизвестная ошибка')
        click.echo(f"Ошибка: {error_msg}")

        if r.get('raw_response'):
            raw = r['raw_response']
            click.echo(f"Ответ API (первые 300 симв.): {raw[:300]}...")
        click.echo("-" * 60)


@cli.command()
@click.argument('name')
def detect(name):
    """Определить категорию для наименования"""
    settings = get_settings()

    click.echo(f"📋 Наименование: {name}")

    # Проверяем все промпты на совпадение (с полной поддержкой regex и glob)
    matched = []
    for pid, cfg in settings.prompts.items():
        if _check_category_match(name, cfg):
            matched.append((pid, cfg))

    if matched:
        click.echo(f"🏷️  Найдено {len(matched)} подходящих промптов:")
        for pid, cfg in matched:
            click.echo(f"    - {pid} ({cfg.category}, сервис: {cfg.service})")
            # Показываем какое ключевое слово сработало
            for kw in cfg.keywords:
                kw_stripped = kw.strip()
                if kw_stripped.startswith('regex:') or kw_stripped.startswith('re:'):
                    pattern = kw_stripped.split(':', 1)[1].strip()
                    try:
                        if re.search(pattern, name, re.IGNORECASE):
                            click.echo(f"      ✓ Совпадение по regex: {kw[:50]}...")
                            break
                    except re.error:
                        continue
                elif kw_stripped.lower() in name.lower():
                    click.echo(f"      ✓ Совпадение по ключевому слову: {kw}")
                    break
    else:
        click.echo("🏷️  Категория: не определена")
        click.echo("🔧 Нет подходящих промптов")


@cli.command()
@click.option('--api', type=click.Choice(['openwebui', 'mws', 'all']), default='all',
              help='Сервис для запроса моделей')
def models(api):
    """Список доступных моделей у сервисов API"""
    settings = get_settings()

    services_to_check = []
    if api == 'all':
        services_to_check = list(settings.api.keys())
    else:
        if api not in settings.api:
            click.echo(f"❌ API '{api}' не настроен в config.yaml", err=True)
            return
        services_to_check = [api]

    for service_name in services_to_check:
        api_config = settings.api[service_name]
        click.echo(f"\n🔌 Сервис: {service_name}")
        click.echo(f"   URL: {api_config.base_url}")
        click.echo(f"   Модель по умолчанию: {api_config.default_model or 'N/A'}")

        try:
            if service_name == "openwebui":
                from api_clients.openwebui import OpenWebUIClient
                client = OpenWebUIClient(
                    base_url=api_config.base_url,
                    api_key=api_config.api_key
                )
            elif service_name == "mws":
                from api_clients.mws_gpt import MWSGPTClient
                client = MWSGPTClient(
                    base_url=api_config.base_url,
                    api_key=api_config.api_key
                )
            else:
                continue

            available_models = client.get_models()

            if available_models:
                click.echo(f"   Доступные модели ({len(available_models)}):")
                for model in available_models[:20]:
                    click.echo(f"      • {model}")
                if len(available_models) > 20:
                    click.echo(f"      ... и ещё {len(available_models) - 20} моделей")
            else:
                click.echo("   ⚠️ Не удалось получить список моделей")

        except Exception as e:
            click.echo(f"   ❌ Ошибка подключения: {e}", err=True)


if __name__ == '__main__':
    cli()