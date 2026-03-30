#!/usr/bin/env python3
import click
import pandas as pd
from pathlib import Path
import logging
import json
from typing import List

from core.database import DatabaseManager
from core.models import NomenclatureItem
from core.processor import NomenclatureProcessor
from prompts.registry import PromptRegistry
from api_clients.openwebui import OpenWebUIClient
from api_clients.mws_gpt import MWSGPTClient
from config.settings import Settings

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
@click.argument('excel_path', type=click.Path(exists=True))
@click.option('--prompt', '-p', multiple=True, help='ID промпта для использования (можно несколько)')
@click.option('--auto', '-a', is_flag=True, help='Автоопределение промптов по категориям')
@click.option('--api', type=click.Choice(['openwebui', 'mws']), default='openwebui')
@click.option('--workers', '-w', default=4, help='Количество параллельных workers')
@click.option('--force', '-f', is_flag=True, help='Перезаписать существующие результаты')
def process(excel_path, prompt, auto, api, workers, force):
    """Обработка Excel файла"""

    # Загрузка настроек
    settings = Settings()

    # Инициализация клиента API
    if api == 'openwebui':
        client = OpenWebUIClient(
            base_url=settings.OPENWEBUI_URL,
            api_key=settings.OPENWEBUI_API_KEY
        )
    else:
        client = MWSGPTClient(
            base_url=settings.MWS_URL,
            api_key=settings.MWS_API_KEY
        )

    if not client.health_check():
        click.echo(f"❌ API {api} недоступен", err=True)
        return

    # Загрузка данных
    df = pd.read_excel(excel_path)
    items = [
        NomenclatureItem(
            article=str(row['артикул']),
            name=str(row['Краткое наименование']),
            guid=str(row['GUID'])
        )
        for _, row in df.iterrows()
    ]

    click.echo(f"📊 Загружено {len(items)} позиций")

    # Инициализация компонентов
    db = DatabaseManager()
    registry = PromptRegistry()
    processor = NomenclatureProcessor(db, registry, client, max_workers=workers)

    # Определение промптов
    if auto:
        click.echo("🤖 Автоопределение категорий...")
        results = processor.auto_process(items, force_reprocess=force)
    else:
        prompt_ids = list(prompt) if prompt else [p.id for p in registry.list_all()]
        click.echo(f"🔧 Используем промпты: {prompt_ids}")
        results = processor.process_batch(items, prompt_ids, force_reprocess=force)

    # Статистика
    stats = {
        'completed': sum(1 for r in results if r.status == 'completed'),
        'ignored': sum(1 for r in results if r.status == 'ignored'),
        'error': sum(1 for r in results if r.status == 'error')
    }

    click.echo(f"\n✅ Завершено: {stats['completed']}")
    click.echo(f"⏭️  Пропущено: {stats['ignored']}")
    click.echo(f"❌ Ошибок: {stats['error']}")


@cli.command()
@click.option('--output', '-o', default='results.json', help='Путь для сохранения')
@click.option('--structure', '-s', type=click.Choice(['flat', 'by_code']), default='flat')
def export(output, structure):
    """Экспорт результатов в JSON"""
    db = DatabaseManager()
    path = db.export_to_json(output, structure)
    click.echo(f"💾 Экспортировано в: {path}")


@cli.command()
def stats():
    """Показать статистику"""
    db = DatabaseManager()
    stats = db.get_statistics()

    click.echo("📈 Статистика обработки:")
    click.echo(f"  Всего записей: {stats['total']}")
    click.echo("\n  По статусам:")
    for status, count in stats['by_status'].items():
        click.echo(f"    {status}: {count}")
    click.echo("\n  По категориям:")
    for cat, count in stats['by_category'].items():
        click.echo(f"    {cat}: {count}")


@cli.command()
@click.argument('name')
def detect(name):
    """Определить категорию для наименования"""
    registry = PromptRegistry()
    category = registry.detect_category(name)
    prompts = registry.get_suitable_prompts(name)

    click.echo(f"📋 Наименование: {name}")
    click.echo(f"🏷️  Категория: {category or 'не определена'}")
    click.echo(f"🔧 Подходящие промпты: {prompts}")


@cli.command()
def prompts():
    """Список доступных промптов"""
    registry = PromptRegistry()

    click.echo("📚 Доступные промпты:")
    for p in registry.list_all():
        click.echo(f"\n  {p.id}:")
        click.echo(f"    Название: {p.name}")
        click.echo(f"    Категория: {p.category}")
        click.echo(f"    Ключевые слова: {', '.join(p.keywords)}")


if __name__ == '__main__':
    cli()