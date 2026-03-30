import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Callable
from tqdm import tqdm

from core.models import NomenclatureItem, ProcessingResult, ProcessingStatus, Parameter
from core.database import DatabaseManager
from prompts.registry import PromptRegistry
from api_clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class NomenclatureProcessor:
    def __init__(
            self,
            db: DatabaseManager,
            registry: PromptRegistry,
            client: BaseLLMClient,
            max_workers: int = 4
    ):
        self.db = db
        self.registry = registry
        self.client = client
        self.max_workers = max_workers

    def process_item(
            self,
            item: NomenclatureItem,
            prompt_id: str,
            force_reprocess: bool = False
    ) -> ProcessingResult:
        """Обработка одного элемента"""

        # Проверяем кэш
        if not force_reprocess:
            cached = self.db.get_result(item.article, prompt_id)
            if cached:
                logger.debug(f"Cache hit for {item.article}")
                return cached

        config = self.registry.get(prompt_id)
        if not config:
            return ProcessingResult(
                article=item.article,
                name=item.name,
                guid=item.guid,
                prompt_id=prompt_id,
                category="unknown",
                status=ProcessingStatus.ERROR,
                error_message=f"Prompt {prompt_id} not found"
            )

        # Проверяем соответствие категории
        detected_category = self.registry.detect_category(item.name)
        if detected_category != config.category:
            # Категория не совпадает - возвращаем ignored
            result = ProcessingResult(
                article=item.article,
                name=item.name,
                guid=item.guid,
                prompt_id=prompt_id,
                category=config.category,
                status=ProcessingStatus.IGNORED,
                display_name=item.name,
                params=[],
                model_used=config.model,
                api_source=self.client.__class__.__name__
            )
            self.db.upsert_result(result)
            return result

        # Формируем и отправляем промпт
        try:
            prompt = self.registry.build_prompt(prompt_id, item.name)
            response = self.client.complete(
                prompt=prompt,
                model=config.model,
                temperature=config.temperature
            )

            if response['success']:
                content = response['content']

                # Парсим ответ
                if isinstance(content, list) and len(content) > 0:
                    item_data = content[0]
                    params = [
                        Parameter(
                            name=p.get('name', ''),
                            value=p.get('value', ''),
                            default=p.get('default', ''),
                            um=p.get('um', '')
                        )
                        for p in item_data.get('params', [])
                    ]

                    result = ProcessingResult(
                        article=item.article,
                        name=item.name,
                        guid=item.guid,
                        prompt_id=prompt_id,
                        category=config.category,
                        status=ProcessingStatus.COMPLETED,
                        display_name=item_data.get('display_name', item.name),
                        params=params,
                        raw_response=response['raw'],
                        model_used=response['model'],
                        api_source=self.client.__class__.__name__
                    )
                else:
                    result = ProcessingResult(
                        article=item.article,
                        name=item.name,
                        guid=item.guid,
                        prompt_id=prompt_id,
                        category=config.category,
                        status=ProcessingStatus.ERROR,
                        error_message="Invalid response structure",
                        raw_response=response['raw']
                    )
            else:
                result = ProcessingResult(
                    article=item.article,
                    name=item.name,
                    guid=item.guid,
                    prompt_id=prompt_id,
                    category=config.category,
                    status=ProcessingStatus.ERROR,
                    error_message=response.get('error', 'Unknown error'),
                    raw_response=response.get('raw')
                )

            self.db.upsert_result(result)
            return result

        except Exception as e:
            logger.error(f"Processing error for {item.article}: {e}")
            result = ProcessingResult(
                article=item.article,
                name=item.name,
                guid=item.guid,
                prompt_id=prompt_id,
                category=config.category,
                status=ProcessingStatus.ERROR,
                error_message=str(e)
            )
            self.db.upsert_result(result)
            return result

    def process_batch(
            self,
            items: List[NomenclatureItem],
            prompt_ids: List[str],
            force_reprocess: bool = False,
            progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """Пакетная обработка с параллелизмом"""

        tasks = []
        for item in items:
            for pid in prompt_ids:
                tasks.append((item, pid))

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.process_item, item, pid, force_reprocess
                ): (item, pid)
                for item, pid in tasks
            }

            with tqdm(total=len(tasks), desc="Processing") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)

                        if progress_callback:
                            progress_callback(result)

                    except Exception as e:
                        item, pid = future_to_task[future]
                        logger.error(f"Task failed for {item.article}: {e}")

                    pbar.update(1)

        return results

    def auto_process(
            self,
            items: List[NomenclatureItem],
            force_reprocess: bool = False
    ) -> List[ProcessingResult]:
        """Автоматическая обработка с определением подходящих промптов"""

        # Группируем по категориям
        by_category: Dict[str, List[NomenclatureItem]] = {}

        for item in items:
            category = self.registry.detect_category(item.name)
            if category:
                by_category.setdefault(category.value, []).append(item)
            else:
                logger.warning(f"Cannot detect category for: {item.name}")

        results = []

        for category, cat_items in by_category.items():
            prompts = self.registry.get_by_category(category)
            prompt_ids = [p.id for p in prompts]

            if prompt_ids:
                logger.info(f"Processing {len(cat_items)} items with category '{category}' using prompts: {prompt_ids}")
                batch_results = self.process_batch(cat_items, prompt_ids, force_reprocess)
                results.extend(batch_results)

        return results