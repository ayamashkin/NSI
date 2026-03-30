"""
Core Processor Module
Основной движок обработки номенклатуры с параллельной обработкой.
"""

import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Callable
from tqdm import tqdm

from config.settings import get_settings, PromptConfig
from utils.excel_loader import NomenclatureItem
from core.database import DatabaseManager

logger = logging.getLogger(__name__)


class NomenclatureProcessor:
    """
    Процессор обработки номенклатуры.

    Features:
    - Параллельная обработка через ThreadPoolExecutor
    - Автоматический выбор API клиента по сервису из промпта
    - Интеграция с DatabaseManager для UPSERT
    - Поддержка прогресс-бара и обработки ошибок
    """

    def __init__(
            self,
            db: DatabaseManager,
            max_workers: Optional[int] = None
    ):
        """
        Инициализация процессора.

        Args:
            db: Менеджер базы данных
            max_workers: Количество параллельных workers (None = из конфига)
        """
        self.db = db
        self.settings = get_settings()
        self.max_workers = max_workers or self.settings.processing.default_workers

        # Инициализация клиентов API
        self._init_api_clients()

    def _init_api_clients(self):
        """Инициализация клиентов API на основе конфигурации."""
        self.clients = {}

        for service_name, api_config in self.settings.api.items():
            logger.info(f"Initializing API client: {service_name} -> {api_config.base_url}")  # ← Добавить лог

            if service_name == "openwebui":
                from api_clients.openwebui import OpenWebUIClient
                self.clients[service_name] = OpenWebUIClient(
                    base_url=api_config.base_url,
                    api_key=api_config.api_key
                )
            elif service_name == "mws":
                from api_clients.mws_gpt import MWSGPTClient
                self.clients[service_name] = MWSGPTClient(
                    base_url=api_config.base_url,
                    api_key=api_config.api_key
                )
            else:
                logger.warning(f"Unknown API service: {service_name}")

    def _get_client(self, service_name: str):
        """Получение клиента API по имени сервиса."""
        if service_name not in self.clients:
            raise ValueError(f"API client not initialized: {service_name}")
        return self.clients[service_name]

    def process_item(
            self,
            item: NomenclatureItem,
            prompt_id: str,
            force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Обработка одного элемента номенклатуры.

        Args:
            item: Элемент номенклатуры
            prompt_id: ID промпта для обработки
            force_reprocess: Принудительная перезапись результата

        Returns:
            Словарь с результатом обработки
        """
        # Проверяем кэш
        if not force_reprocess:
            cached = self.db.get_result(item.article, prompt_id)
            if cached:
                logger.debug(f"Cache hit for {item.article}/{prompt_id}")
                return cached

        # Получаем конфигурацию промпта
        prompt_cfg = self.settings.get_prompt(prompt_id)
        if not prompt_cfg:
            return self._create_error_result(
                item, prompt_id, f"Prompt {prompt_id} not found"
            )

        # Проверяем соответствие категории (простая проверка по ключевым словам)
        if not self._check_category_match(item.name, prompt_cfg):
            return {
                "article": item.article,
                "name": item.name,
                "guid": item.guid,
                "prompt_id": prompt_id,
                "category": prompt_cfg.category,
                "status": "ignored",
                "display_name": item.name,
                "params": [],
                "processed_at": datetime.utcnow().isoformat()
            }

        # Загружаем текст промпта
        try:
            prompt_text = self._load_prompt_text(prompt_cfg.file_path, item.name)
        except Exception as e:
            return self._create_error_result(item, prompt_id, f"Failed to load prompt: {e}")

        logger.info(f"Prompt {prompt_id} uses service: {prompt_cfg.service}")  # ← Добавить лог

        # Получаем клиент API для сервиса из промпта
        try:
            client = self._get_client(prompt_cfg.service)
        except ValueError as e:
            return self._create_error_result(item, prompt_id, str(e))

        # Отправляем запрос к API
        try:
            logger.info(f"Calling API for {item.article} with prompt {prompt_id}")
            response = client.complete(
                prompt=prompt_text,
                model=prompt_cfg.model,
                temperature=prompt_cfg.temperature
            )
            logger.info(f"API response: success={response.get('success')}, error={response.get('error')}")
        except Exception as e:
            return self._create_error_result(item, prompt_id, f"API error: {e}")

        # Парсим ответ
        result = self._parse_response(item, prompt_id, prompt_cfg, response)

        # Сохраняем в БД
        self.db.upsert_result(result)

        return result

    def _load_prompt_text(self, file_path: str, nomenclature: str) -> str:
        """Загрузка и подготовка текста промпта."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Заменяем плейсхолдер на наименование
        return text.replace("{{NOMENCLATURE}}", nomenclature)

    def _check_category_match(self, name: str, prompt_cfg: PromptConfig) -> bool:
        """Проверка соответствия категории по ключевым словам."""
        name_lower = name.lower()
        for keyword in prompt_cfg.keywords:
            if keyword.lower() in name_lower:
                return True
        return False

    def _parse_response(
            self,
            item: NomenclatureItem,
            prompt_id: str,
            prompt_cfg: PromptConfig,
            response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Парсинг ответа от API."""

        if not response.get('success'):
            return self._create_error_result(
                item, prompt_id,
                response.get('error', 'Unknown API error'),
                response.get('raw')
            )

        content = response.get('content')

        # Пытаемся извлечь JSON
        try:
            if isinstance(content, list) and len(content) > 0:
                result_data = content[0]
            elif isinstance(content, dict):
                result_data = content
            else:
                raise ValueError("Invalid response structure")

            return {
                "article": item.article,
                "name": item.name,
                "guid": item.guid,
                "prompt_id": prompt_id,
                "category": prompt_cfg.category,
                "status": result_data.get('status', 'completed'),
                "display_name": result_data.get('display_name', item.name),
                "params": result_data.get('params', []),
                "raw_response": response.get('raw'),
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": response.get('model', prompt_cfg.model),
                "api_source": prompt_cfg.service
            }

        except Exception as e:
            return self._create_error_result(
                item, prompt_id, f"Parse error: {e}", response.get('raw')
            )

    def _create_error_result(
            self,
            item: NomenclatureItem,
            prompt_id: str,
            error_message: str,
            raw_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """Создание результата с ошибкой."""
        return {
            "article": item.article,
            "name": item.name,
            "guid": item.guid,
            "prompt_id": prompt_id,
            "category": "unknown",
            "status": "error",
            "display_name": item.name,
            "params": [],
            "error_message": error_message,
            "raw_response": raw_response,
            "processed_at": datetime.utcnow().isoformat()
        }

    def process_batch(
            self,
            items: List[NomenclatureItem],
            prompt_ids: List[str],
            force_reprocess: bool = False,
            progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Пакетная обработка с параллелизмом.

        Args:
            items: Список элементов номенклатуры
            prompt_ids: Список ID промптов для применения
            force_reprocess: Принудительная перезапись
            progress_callback: Callback для прогресса

        Returns:
            Список результатов обработки
        """
        # Формируем все задачи (item + prompt_id)
        tasks = [
            (item, pid)
            for item in items
            for pid in prompt_ids
        ]

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Создаем Future для всех задач
            future_to_task = {
                executor.submit(
                    self.process_item, item, pid, force_reprocess
                ): (item, pid)
                for item, pid in tasks
            }

            # Обрабатываем по мере готовности с прогресс-баром
            with tqdm(total=len(tasks), desc="Processing") as pbar:
                for future in as_completed(future_to_task):
                    item, pid = future_to_task[future]

                    try:
                        result = future.result()
                        results.append(result)

                        if progress_callback:
                            progress_callback(result)

                    except Exception as e:
                        logger.error(f"Task failed for {item.article}/{pid}: {e}")
                        # Создаем результат с ошибкой
                        error_result = self._create_error_result(
                            item, pid, f"Task execution error: {e}"
                        )
                        results.append(error_result)

                    pbar.update(1)

        return results

    def auto_process(
            self,
            items: List[NomenclatureItem],
            force_reprocess: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Автоматическая обработка с определением подходящих промптов.

        Для каждого элемента выбирает промпты по совпадению ключевых слов.

        Args:
            items: Список элементов номенклатуры
            force_reprocess: Принудительная перезапись

        Returns:
            Список результатов обработки
        """
        results = []

        for item in items:
            # Находим подходящие промпты по ключевым словам
            matching_prompts = []
            for pid, cfg in self.settings.prompts.items():
                if self._check_category_match(item.name, cfg):
                    matching_prompts.append(pid)

            if not matching_prompts:
                logger.warning(f"No matching prompts for: {item.name}")
                continue

            logger.info(
                f"Processing {item.article} with prompts: {matching_prompts}"
            )

            # Обрабатываем только подходящие промпты
            batch_results = self.process_batch(
                [item],
                matching_prompts,
                force_reprocess
            )
            results.extend(batch_results)

        return results
def load_excel_items(self, excel_path: str) -> List[NomenclatureItem]:
    """Загрузка элементов из Excel (ленивый импорт)."""
    from utils.excel_loader import ExcelLoader  # ← Импорт внутри метода
    loader = ExcelLoader(excel_path)
    return loader.load()

from datetime import datetime
from pathlib import Path