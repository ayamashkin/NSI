"""
LLM Mask Generator
Генерация regex масок через LLM (local/cloud) с каскадным fallback.

LAST_FIX: 2026-05-15 17:11 UTC+3 — provider_priority динамический (default_service первым); _call_llm: client.complete + per-provider model; generate_mask: item_type из ЕСН
"""

import logging
import re
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Результат генерации маски."""
    pattern: str
    params: List[str]
    required: List[str]
    standard: str
    item_type: str
    score: float = 0.0
    test_examples: List[Dict] = None
    raw_response: str = ""
    attempts: int = 0
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern': self.pattern,
            'params': self.params,
            'required': self.required,
            'standard': self.standard,
            'item_type': self.item_type,
            'score': self.score,
            'test_examples': self.test_examples or []
        }


class LLMMaskGenerator:
    """
    Генератор масок через LLM с каскадным fallback:
    1. default_service (из config.yaml) -> 2. Остальные доступные клиенты

    Features:
    - Динамический provider_priority из конфигурации
    - Per-provider model selection (из APIConfig.default_model)
    - Retry с увеличением temperature
    - Auto-validation после генерации
    """

    def __init__(
        self,
        clients: Dict[str, Any],
        settings: Optional[Any] = None,
        max_retries: int = 3,
        min_examples: int = 10
    ):
        self.clients = clients
        self.settings = settings
        self.max_retries = max_retries
        self.min_examples = min_examples

        # Динамический порядок fallback:
        # 1. default_service из settings.mask_generation (если указан и клиент доступен)
        # 2. Все остальные доступные клиенты (ключи из self.clients)
        self.provider_priority = self._build_provider_priority()

        logger.info("LLMMaskGenerator initialized with %d clients, priority: %s",
                    len(clients), self.provider_priority)

    def _build_provider_priority(self) -> List[str]:
        """Строит приоритет провайдеров: default_service первым, затем остальные."""
        priority = []
        default_service = None

        if self.settings is not None:
            # Пробуем получить из settings напрямую (не должно сработать, т.к. поле в mask_generation)
            default_service = getattr(self.settings, 'default_service', None)
            # Основной источник — mask_generation
            if not default_service and hasattr(self.settings, 'mask_generation'):
                default_service = getattr(self.settings.mask_generation, 'default_service', None)
                logger.info("[LLM] default_service from mask_generation: '%s'", default_service)
            elif default_service:
                logger.info("[LLM] default_service from settings: '%s'", default_service)
        else:
            logger.warning("[LLM] self.settings is None — default_service ignored!")

        if default_service:
            if default_service in self.clients:
                priority.append(default_service)
                logger.info("[LLM] default_service='%s' set as primary", default_service)
            else:
                logger.error(
                    "[LLM] default_service='%s' NOT FOUND in clients %s. "
                    "Check api_key / config / import errors.",
                    default_service, list(self.clients.keys())
                )

        for provider in self.clients.keys():
            if provider not in priority:
                priority.append(provider)

        logger.info("[LLM] Final provider_priority: %s", priority)
        return priority
    
    def generate_mask(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict],
        name: str = "",
        standard_info: Optional[Any] = None
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        """
        Генерация маски через LLM.

        Args:
            standard: Стандарт (например, 'ОСТ 1 31133-80')
            item_type: Тип изделия (например, 'шайба', 'болт', 'винт') — уже нормализован из ЕСН
            examples: Примеры из ЕСН
            name: Исходное наименование (для контекста)
            standard_info: Информация о стандарте

        Returns:
            Tuple[mask_dict, attempts_count]
        """
        if len(examples) < self.min_examples:
            logger.warning("Not enough examples: %d < %d", len(examples), self.min_examples)
            return None, 0

        # Подготовка промпта
        prompt = self._build_prompt(standard, item_type, examples, standard_info)

        # Попытки генерации через разных провайдеров
        for attempt in range(self.max_retries):
            for provider in self.provider_priority:
                if provider not in self.clients:
                    continue

                try:
                    result = self._call_llm(provider, prompt, attempt)
                    if result:
                        mask = self._parse_response(result, standard, item_type)
                        if mask:
                            logger.info("Generated mask via %s (attempt %d)", provider, attempt + 1)
                            return mask, attempt + 1
                except Exception as e:
                    logger.warning("LLM %s failed (attempt %d): %s", provider, attempt + 1, e)

        logger.error("Failed to generate mask after %d attempts", self.max_retries)
        return None, self.max_retries

    def _build_prompt(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict],
        standard_info: Optional[Any] = None
    ) -> str:
        """Построение промпта для LLM."""
        # Берем примеры наименований
        sample_names = []
        for ex in examples[:20]:
            name = ex.get('полное_наименование') or ex.get('наименование', '')
            if name:
                sample_names.append(name)

        # Уникальные примеры
        unique_names = list(dict.fromkeys(sample_names))[:15]

        prompt = f"""Сгенерируй Python regex паттерн для извлечения параметров из наименований крепежа.

Стандарт: {standard}
Тип изделия: {item_type}

Примеры наименований:
"""
        for name in unique_names:
            prompt += f"- {name}\n"

        prompt += """
Требования:
1. Паттерн должен быть в формате Python regex с именованными группами (?P<<name>...)
2. Группы должны иметь короткие имена на русском языке (например: диаметр, длина, покрытие, исполнение)
3. Обязательные параметры: номинальный диаметр, длина, покрытие
4. Опциональные: исполнение, шаг резьбы, класс прочности
5. Паттерн должен быть case-insensitive
6. Разделители: пробел, дефис, точка

Верни результат в формате JSON:
{
    "pattern": "regex pattern here",
    "params": ["param1", "param2", ...],
    "required": ["param1", ...]
}
"""
        return prompt

    def _call_llm(self, provider: str, prompt: str, attempt: int) -> Optional[str]:
        """Вызов LLM через клиент."""
        client = self.clients.get(provider)
        if not client:
            return None

        # Увеличиваем temperature с каждой попыткой
        temperature = min(0.1 + attempt * 0.1, 0.5)

        try:
            # Получаем модель для конкретного провайдера из конфига
            model = None
            if self.settings and hasattr(self.settings, 'api') and provider in self.settings.api:
                api_cfg = self.settings.api[provider]
                model = getattr(api_cfg, 'default_model', None)
                logger.debug("[LLM] Using model '%s' from api.%s.default_model", model, provider)

            # Fallback: общая default_model из mask_generation
            if not model and self.settings and hasattr(self.settings, 'mask_generation'):
                model = getattr(self.settings.mask_generation, 'default_model', None)
                logger.debug("[LLM] Fallback to mask_generation.default_model: '%s'", model)

            # Ultimate fallback
            if not model:
                model = "qwen2.5-72b-instruct"
                logger.debug("[LLM] Ultimate fallback model: '%s'", model)

            response = client.complete(
                prompt=prompt,
                model=model,
                temperature=temperature
            )
            # response — это dict с ключами: success, content, raw, error, model
            if response and response.get('success'):
                return response.get('raw', '')
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                logger.warning("LLM call failed: %s", error_msg)
                return None
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None

    def _parse_response(
        self,
        response: str,
        standard: str,
        item_type: str
    ) -> Optional[Dict[str, Any]]:
        """Парсинг ответа LLM."""
        try:
            # Извлекаем JSON из ответа
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return None

            data = json.loads(json_match.group())

            pattern = data.get('pattern', '')
            params = data.get('params', [])
            required = data.get('required', [])

            # Валидация паттерна
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning("Invalid regex pattern: %s", e)
                return None

            return {
                'standard': standard,
                'item_type': item_type,
                'pattern': pattern,
                'params': params,
                'required': required
            }

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response: %s", e)
            return None
        except Exception as e:
            logger.warning("Unexpected error parsing response: %s", e)
            return None