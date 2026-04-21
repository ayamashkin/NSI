"""
Integration Module
Интеграция каскадного парсера с существующей системой nomenclature-processor.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Импортируем из существующей системы
from config.settings import get_settings, PromptConfig
from utils.excel_loader import NomenclatureItem
from core.database import DatabaseManager

# Импортируем новые компоненты
from parsers.cascade import CascadeParser, ParseResult, ParseLevel
from ens.loader import ENSLoader, ENSCategory
from ens.indexer import HybridENSIndex, ENSIndex

logger = logging.getLogger(__name__)


class ENSNomenclatureProcessor:
    """
    Интегрированный процессор номенклатуры с использованием ЕСН.

    Объединяет:
    - Каскадный парсер (regex + LLM)
    - Индекс ЕСН для few-shot
    - Существующую базу данных
    - Существующие API клиенты
    """

    def __init__(
        self,
        db: DatabaseManager,
        ens_index_path: Optional[str] = None,
        use_llm: bool = True,
        llm_service: str = "openwebui",
        regex_threshold: float = 0.85
    ):
        """
        Инициализация процессора.

        Args:
            db: Менеджер базы данных
            ens_index_path: Путь к индексу ЕСН
            use_llm: Использовать LLM
            llm_service: Сервис LLM (openwebui, mws, gigachat)
            regex_threshold: Порог уверенности для regex
        """
        self.db = db
        self.settings = get_settings()
        self.use_llm = use_llm
        self.llm_service = llm_service

        # Загружаем индекс ЕСН
        self.ens_index = None
        if ens_index_path and Path(ens_index_path).exists():
            logger.info(f"Loading ENS index from {ens_index_path}")
            self.ens_index = ENSIndex.load(ens_index_path)

        # Инициализируем LLM клиент
        self.llm_client = None
        if use_llm:
            self.llm_client = self._init_llm_client(llm_service)

        # Инициализируем каскадный парсер
        self.parser = CascadeParser(
            llm_client=self.llm_client,
            ens_index=self.ens_index,
            regex_confidence_threshold=regex_threshold,
            use_llm=use_llm
        )

        logger.info(f"ENSNomenclatureProcessor initialized (LLM: {use_llm}, service: {llm_service})")

    def _init_llm_client(self, service_name: str):
        """Инициализация LLM клиента."""
        api_config = self.settings.api.get(service_name)
        if not api_config:
            logger.warning(f"API config not found for {service_name}")
            return None

        try:
            if service_name == "openwebui":
                from api_clients.openwebui import OpenWebUIClient
                if api_config.api_key:
                    return OpenWebUIClient(
                        base_url=api_config.base_url,
                        api_key=api_config.api_key
                    )
                elif api_config.username and api_config.password:
                    return OpenWebUIClient(
                        base_url=api_config.base_url,
                        username=api_config.username,
                        password=api_config.password
                    )
            elif service_name == "mws":
                from api_clients.mws_gpt import MWSGPTClient
                return MWSGPTClient(
                    base_url=api_config.base_url,
                    api_key=api_config.api_key
                )
            elif service_name == "gigachat":
                from api_clients.gigachat import GigaChatClient
                return GigaChatClient(
                    base_url=api_config.base_url,
                    api_key=api_config.api_key,
                    scope=api_config.scope
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return None

        return None

    def process_item(
        self,
        item: NomenclatureItem,
        prompt_id: str = "ens_hardware",
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Обработка одного элемента номенклатуры.

        Args:
            item: Элемент номенклатуры
            prompt_id: ID промпта (для совместимости с существующей системой)
            force_reprocess: Принудительная перезапись

        Returns:
            Результат обработки в формате существующей системы
        """
        # Проверяем кэш
        if not force_reprocess:
            cached = self.db.get_result(item.article, prompt_id)
            if cached and cached.get('status') == 'completed':
                logger.debug(f"Cache hit for {item.article}")
                return cached

        # Парсим через каскад
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            parse_result = loop.run_until_complete(self.parser.parse(item.name))
        except RuntimeError:
            # Если нет event loop (например в новом потоке)
            parse_result = self.parser._parse_sync(item.name)

        # Конвертируем ParseResult в формат существующей системы
        result = self._convert_to_legacy_format(parse_result, item, prompt_id)

        # Сохраняем в БД
        self.db.upsert_result(result)

        return result

    def _convert_to_legacy_format(
        self,
        parse_result: ParseResult,
        item: NomenclatureItem,
        prompt_id: str
    ) -> Dict[str, Any]:
        """Конвертация ParseResult в формат существующей системы."""

        # Определяем статус
        if parse_result.level == ParseLevel.FAILED:
            status = "error"
        elif parse_result.confidence < 0.5:
            status = "error"
        else:
            status = "completed"

        # Формируем params в формате существующей системы
        params_list = []
        for key, value in parse_result.params.items():
            if not key.startswith('_'):
                params_list.append({
                    "name": key,
                    "value": str(value) if value is not None else "",
                    "default": "",
                    "um": self._guess_unit(key)
                })

        return {
            "article": item.article,
            "name": item.name,
            "guid": item.guid,
            "prompt_id": prompt_id,
            "category": parse_result.params.get('тип', 'unknown').lower(),
            "status": status,
            "display_name": item.name,
            "params": params_list,
            "raw_response": parse_result.params.get('_llm_raw'),
            "error_message": parse_result.params.get('_error'),
            "processed_at": datetime.utcnow().isoformat(),
            "model_used": self.llm_service if parse_result.level == ParseLevel.LLM else "regex",
            "api_source": self.llm_service if parse_result.level == ParseLevel.LLM else "cascade",
            "_cascade_level": parse_result.level.value,
            "_cascade_confidence": parse_result.confidence,
            "_ens_matches": parse_result.ens_matches[:2] if parse_result.ens_matches else []
        }

    def _guess_unit(self, param_name: str) -> str:
        """Угадывание единицы измерения по названию параметра."""
        units = {
            'диаметр': 'мм',
            'длина': 'мм',
            'толщина': 'мм',
            'шаг': 'мм',
            'толщина_покрытия': 'мкм',
        }
        return units.get(param_name.lower(), '')

    def process_batch(
        self,
        items: List[NomenclatureItem],
        prompt_id: str = "ens_hardware",
        force_reprocess: bool = False,
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """Пакетная обработка с прогресс-баром."""
        from tqdm import tqdm

        results = []

        for item in tqdm(items, desc="Processing"):
            result = self.process_item(item, prompt_id, force_reprocess)
            results.append(result)

            if progress_callback:
                progress_callback(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики парсера."""
        return self.parser.get_stats()



def _analyze_field_statistics(loader, items: List[Dict]) -> Dict[str, Any]:
    """
    Подробная статистика по полям ЕСН.
    Собирает ВСЕ поля из данных — без фиксированных списков.
    """
    if not items or loader.df is None:
        return {"error": "No data"}

    total = len(items)
    raw_columns = [str(c) for c in loader.df.columns if c is not None]

    # 1. Статистика по ВСЕМ полям (сколько заполнено)
    field_stats = {}
    for item in items:
        for key, val in item.items():
            if key.startswith('_'):
                continue
            if val is not None and str(val).strip():
                field_stats[key] = field_stats.get(key, 0) + 1

    # Сортируем по заполненности
    sorted_stats = sorted(
        [(k, v, round(v / total * 100, 1)) for k, v in field_stats.items()],
        key=lambda x: -x[1]
    )

    # 2. Определяем категории динамически (по заполненности)
    required_status = {}
    recommended_status = {}
    optional_status = {}

    for field, filled, pct in sorted_stats:
        info = {'filled': filled, 'total': total, 'percent': pct, 'ok': filled > 0}
        if pct >= 95:
            required_status[field] = info
        elif pct >= 50:
            recommended_status[field] = info
        else:
            optional_status[field] = info

    # 3. Неиспользуемые колонки Excel
    mapped_sources = set()
    for src, dst in loader.schema.column_mapping.items():
        mapped_sources.add(src.lower())

    unused_columns = []
    for col in raw_columns:
        col_lower = col.lower()
        if col_lower not in mapped_sources:
            auto_mapped = loader.column_mapping.auto_map_column(col)
            if not auto_mapped:
                unused_columns.append(col)

    # 4. Поля с низкой заполненностью (< 10%)
    low_fill_fields = [
        {'field': k, 'filled': v, 'percent': p}
        for k, v, p in sorted_stats
        if p < 10 and p > 0
    ]

    # 5. Обратный маппинг: какое поле ЕСН -> откуда в Excel
    reverse_mapping = {}
    for src, dst in loader.schema.column_mapping.items():
        reverse_mapping[dst] = src

    return {
        'total_items': total,
        'total_fields': len(sorted_stats),
        'field_stats': sorted_stats,
        'required_fields': required_status,
        'recommended_fields': recommended_status,
        'optional_fields': optional_status,
        'unused_columns': unused_columns,
        'low_fill_fields': low_fill_fields,
        'reverse_mapping': reverse_mapping,
    }


def _print_field_report(stats: Dict[str, Any]):
    """Вывод отчёта в консоль."""
    import click

    click.echo("\n" + "=" * 70)
    click.echo("📊 СТАТИСТИКА ПОЛЕЙ ЕСН (динамический анализ)")
    click.echo("=" * 70)
    click.echo(f"Всего записей: {stats['total_items']}")
    click.echo(f"Уникальных полей: {stats['total_fields']}")

    # Обязательные поля (≥95%)
    click.echo("\n🔴 ОБЯЗАТЕЛЬНЫЕ ПОЛЯ (≥95% заполненности):")
    for field, info in stats['required_fields'].items():
        src = stats['reverse_mapping'].get(field, '?')
        click.echo(f"  ✅ {field:<30} {info['filled']:>8}/{info['total']} ({info['percent']}%)")
        if src != field:
            click.echo(f"     └─ из Excel колонки: \"{src}\"")

    # Рекомендуемые поля (50-95%)
    click.echo("\n🟡 РЕКОМЕНДУЕМЫЕ ПОЛЯ (50-95% заполненности):")
    for field, info in stats['recommended_fields'].items():
        src = stats['reverse_mapping'].get(field, '?')
        click.echo(f"  {field:<30} {info['filled']:>8}/{info['total']} ({info['percent']}%)")
        if src != field:
            click.echo(f"     └─ из Excel колонки: \"{src}\"")

    # Опциональные поля (<50%)
    click.echo("\n⚪ ОПЦИОНАЛЬНЫЕ ПОЛЯ (<50% заполненности):")
    shown = 0
    for field, info in stats['optional_fields'].items():
        if shown >= 10:
            remaining = len(stats['optional_fields']) - shown
            click.echo(f"  ... и ещё {remaining} полей")
            break
        src = stats['reverse_mapping'].get(field, '?')
        click.echo(f"  {field:<30} {info['filled']:>8}/{info['total']} ({info['percent']}%)")
        shown += 1

    # Неиспользуемые колонки
    if stats['unused_columns']:
        click.echo(f"\n⚪ НЕИСПОЛЬЗУЕМЫЕ КОЛОНКИ EXCEL ({len(stats['unused_columns'])}):")
        for col in stats['unused_columns'][:15]:
            click.echo(f"    - \"{col}\"")
        if len(stats['unused_columns']) > 15:
            click.echo(f"    ... и ещё {len(stats['unused_columns']) - 15}")
        click.echo("  💡 Добавьте в ens_column_mapping.yaml если содержат полезные данные")

    # Поля с низкой заполненностью
    if stats['low_fill_fields']:
        click.echo(f"\n⚠️  ПОЛЯ С НИЗКОЙ ЗАПОЛНЕННОСТЬЮ (< 10%):")
        for f in stats['low_fill_fields'][:10]:
            click.echo(f"    {f['field']:<30} {f['filled']:>8} ({f['percent']}%)")

    # Топ-20 по заполненности
    click.echo("\n📈 ТОП-20 ПО ЗАПОЛНЕННОСТИ:")
    for field, filled, pct in stats['field_stats'][:20]:
        bar = "█" * int(pct / 5)
        click.echo(f"  {bar:<20} {field:<30} {filled:>8}/{stats['total_items']} ({pct}%)")

    click.echo("=" * 70)


# Утилиты для работы с ЕСН
def build_ens_index(
    ens_file: str,
    output_path: str,
    category: Optional[str] = None
) -> str:
    """
    Построение индекса ЕСН из файла.

    Args:
        ens_file: Путь к Excel файлу ЕСН
        output_path: Путь для сохранения индекса
        category: Категория (hardware, washer, rolled_metal)

    Returns:
        Путь к сохраненному индексу
    """
    from datetime import datetime

    # Загружаем
    cat_enum = ENSCategory(category) if category else None
    loader = ENSLoader(ens_file, category=cat_enum)
    items = loader.load()

    logger.info(f"Loaded {len(items)} items from {ens_file}")

    # === СТАТИСТИКА ПОЛЕЙ ===
    field_stats = _analyze_field_statistics(loader, items)
    _print_field_report(field_stats)

    # Строим индекс
    index = HybridENSIndex(items)

    # Сохраняем
    index.fuzzy_index.save(output_path)

    # Сохраняем метаданные (с полной статистикой)
    meta = {
        "source_file": ens_file,
        "category": loader.category.value,
        "item_count": len(items),
        "created_at": datetime.utcnow().isoformat(),
        "columns": list(loader.df.columns) if loader.df is not None else [],
        "column_mapping": dict(loader.schema.column_mapping) if loader.schema else {},
        "field_statistics": {
            "total_fields": field_stats['total_fields'],
            "field_stats": [
                {"field": k, "filled": v, "percent": p}
                for k, v, p in field_stats['field_stats']
            ],
            "required_fields": field_stats['required_fields'],
            "recommended_fields": field_stats['recommended_fields'],
            "optional_fields": field_stats['optional_fields'],
            "optional_fields": field_stats['optional_fields'],
            "unused_columns": field_stats['unused_columns'],
            "low_fill_fields": field_stats['low_fill_fields'],
        }
    }

    meta_path = Path(output_path).with_suffix('.meta.json')
    import json
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Index saved to {output_path}")
    return output_path


def analyze_nomenclature(
    input_file: str,
    ens_index_path: str,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Анализ файла номенклатуры без полной обработки.

    Args:
        input_file: Путь к Excel файлу
        ens_index_path: Путь к индексу ЕСН
        sample_size: Размер выборки для анализа

    Returns:
        Статистика по файлу
    """
    import pandas as pd
    from parsers.cascade import RegexFastenerParser

    # Загружаем
    df = pd.read_excel(input_file)
    texts = df['Краткое наименование'].astype(str).tolist()

    # Берём выборку
    if len(texts) > sample_size:
        import random
        sample = random.sample(texts, sample_size)
    else:
        sample = texts

    # Анализируем
    parser = FastenerRegexParser()
    stats = {
        'total': len(texts),
        'regex_parsed': 0,
        'failed': 0,
        'by_type': {}
    }

    for text in sample:
        result = parser.parse(text)
        if result:
            stats['regex_parsed'] += 1
            item_type = result.params.get('тип', 'unknown')
            stats['by_type'][item_type] = stats['by_type'].get(item_type, 0) + 1
        else:
            stats['failed'] += 1

    # Экстраполируем на полный набор
    if sample_size < len(texts):
        ratio = len(texts) / sample_size
        stats['estimated_regex_parsed'] = int(stats['regex_parsed'] * ratio)
        stats['estimated_failed'] = int(stats['failed'] * ratio)

    # Загружаем индекс для проверки
    if Path(ens_index_path).exists():
        ens_index = ENSIndex.load(ens_index_path)
        coverage = len(ens_index.items)
        stats['ens_coverage'] = coverage

    return stats