"""
LLM Mask Generator Module
Level 2: Автоматическая генерация regex масок с помощью LLM.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Провайдеры LLM."""
    OPENWEBUI = "openwebui"
    MWS = "mws"
    GIGACHAT = "gigachat"


@dataclass
class GenerationAttempt:
    """Попытка генерации маски."""
    attempt_number: int
    provider: str
    model: str
    temperature: float
    success: bool
    pattern: Optional[str] = None
    params: Optional[List[str]] = None
    required: Optional[List[str]] = None
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    validation_score: Optional[float] = None


class LLMMaskGenerator:
    """
    Генератор масок через LLM с fallback стратегией.

    Features:
    - Локальные модели по умолчанию (Qwen3:7b через OpenWebUI)
    - Fallback на облачные (GPT-4 через MWS)
    - Retry с разными температурами
    - Few-shot с примерами из ЕСН
    """

    # Стратегия retry
    RETRY_CONFIG = [
        {"provider": LLMProvider.OPENWEBUI, "model": "qwen3:7b", "temperature": 0.1},
        {"provider": LLMProvider.OPENWEBUI, "model": "qwen3:7b", "temperature": 0.3},
        {"provider": LLMProvider.OPENWEBUI, "model": "qwen3:30b", "temperature": 0.1},
        {"provider": LLMProvider.MWS, "model": "gpt-4", "temperature": 0.1},
    ]

    def __init__(
        self,
        clients: Dict[LLMProvider, Any],
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Инициализация генератора.

        Args:
            clients: Словарь {provider: client_instance}
            timeout: Таймаут запроса в секундах
            max_retries: Максимальное количество попыток
        """
        self.clients = clients
        self.timeout = timeout
        self.max_retries = min(max_retries, len(self.RETRY_CONFIG))
        self.attempts: List[GenerationAttempt] = []

    def generate_mask(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict[str, Any]],
        context: Optional[Dict] = None
    ) -> Tuple[Optional[Dict[str, Any]], List[GenerationAttempt]]:
        """
        Генерация маски с retry стратегией.

        Args:
            standard: Стандарт (ГОСТ, ОСТ, etc.)
            item_type: Тип изделия
            examples: Примеры из ЕСН
            context: Дополнительный контекст

        Returns:
            (mask_dict или None, список попыток)
        """
        self.attempts = []

        for i in range(self.max_retries):
            config = self.RETRY_CONFIG[i]
            provider = config["provider"]

            # Проверяем наличие клиента
            if provider not in self.clients or self.clients[provider] is None:
                logger.warning(f"Client for {provider} not available, skipping")
                continue

            attempt = self._try_generate(
                attempt_number=i + 1,
                provider=provider,
                model=config["model"],
                temperature=config["temperature"],
                standard=standard,
                item_type=item_type,
                examples=examples,
                context=context
            )

            self.attempts.append(attempt)

            if attempt.success:
                mask = {
                    "pattern": attempt.pattern,
                    "params": attempt.params,
                    "required": attempt.required,
                    "standard": standard,
                    "item_type": item_type,
                    "source": "llm",
                    "auto_score": 0.0  # Будет установлено после валидации
                }

                logger.info(
                    f"Successfully generated mask for {standard}/{item_type} "
                    f"on attempt {i + 1}"
                )
                return mask, self.attempts

        logger.error(
            f"Failed to generate mask for {standard}/{item_type} "
            f"after {self.max_retries} attempts"
        )
        return None, self.attempts

    def _try_generate(
        self,
        attempt_number: int,
        provider: LLMProvider,
        model: str,
        temperature: float,
        standard: str,
        item_type: str,
        examples: List[Dict[str, Any]],
        context: Optional[Dict]
    ) -> GenerationAttempt:
        """Одна попытка генерации."""
        client = self.clients[provider]

        # Формируем промпт
        prompt = self._build_prompt(standard, item_type, examples, context)

        try:
            # Вызываем LLM
            response = client.complete(
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_prompt=self._get_system_prompt()
            )

            raw_response = response.get("raw", "")

            # Парсим JSON из ответа
            content = response.get("content")
            if isinstance(content, dict):
                result = content
            else:
                result = self._extract_json(raw_response)

            if result and "pattern" in result:
                return GenerationAttempt(
                    attempt_number=attempt_number,
                    provider=provider.value,
                    model=model,
                    temperature=temperature,
                    success=True,
                    pattern=result["pattern"],
                    params=result.get("params", []),
                    required=result.get("required", []),
                    raw_response=raw_response[:500]  # Только первые 500 символов
                )
            else:
                return GenerationAttempt(
                    attempt_number=attempt_number,
                    provider=provider.value,
                    model=model,
                    temperature=temperature,
                    success=False,
                    raw_response=raw_response[:500],
                    error_message="No valid pattern in response"
                )

        except Exception as e:
            logger.error(f"Generation attempt {attempt_number} failed: {e}")
            return GenerationAttempt(
                attempt_number=attempt_number,
                provider=provider.value,
                model=model,
                temperature=temperature,
                success=False,
                error_message=str(e)
            )

    def _build_prompt(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict[str, Any]],
        context: Optional[Dict]
    ) -> str:
        """Построение промпта для LLM."""
        # Берем до 5 примеров
        sample_examples = examples[:5]

        examples_text = "\n".join([
            f'{i+1}. {ex.get("полное_наименование") or ex.get("наименование", "")}'
            for i, ex in enumerate(sample_examples)
        ])

        context_text = ""
        if context:
            context_text = f'\nДополнительный контекст: {json.dumps(context, ensure_ascii=False)}\n'

        prompt = f"""Ты - эксперт по техническим стандартам и регулярным выражениям.

Создай regex-паттерн для извлечения параметров из номенклатуры {item_type} по стандарту {standard}.

Примеры из справочника:
{examples_text}
{context_text}
Требования к паттерну:
1. Используй named groups (?P<name>...) для каждого параметра
2. Поддерживай различные варианты написания (с пробелами, без)
3. Учитывай специфику стандарта {standard}
4. Паттерн должен быть достаточно гибким, но точным

Ожидаемые параметры для {item_type}:
- тип: тип изделия
- исполнение: номер исполнения (если есть)
- диаметр: номинальный диаметр
- длина: длина изделия
- покрытие: тип покрытия
- стандарт: полное название стандарта

Ответь ТОЛЬКО в формате JSON:
{{
  "pattern": "regex pattern with named groups",
  "params": ["список", "всех", "параметров"],
  "required": ["список", "обязательных", "параметров"]
}}
"""
        return prompt

    def _get_system_prompt(self) -> str:
        """Системный промпт для LLM."""
        return """Вы - эксперт по техническим стандартам ГОСТ/ОСТ и регулярным выражениям Python.
Ваша задача - создавать точные regex-паттерны для парсинга номенклатуры.
Всегда отвечайте в формате JSON."""

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Извлечение JSON из текста."""
        # Пробуем найти JSON в markdown code blocks
        import re

        # ```json ... ```
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # ``` ... ```
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Пробуем найти JSON объект напрямую
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def get_generation_stats(self) -> Dict[str, Any]:
        """Статистика по генерациям."""
        if not self.attempts:
            return {}

        total = len(self.attempts)
        successful = sum(1 for a in self.attempts if a.success)

        by_provider = {}
        for attempt in self.attempts:
            provider = attempt.provider
            if provider not in by_provider:
                by_provider[provider] = {"total": 0, "success": 0}
            by_provider[provider]["total"] += 1
            if attempt.success:
                by_provider[provider]["success"] += 1

        return {
            "total_attempts": total,
            "successful_attempts": successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_provider": by_provider
        }


class MaskQualityGate:
    """
    Ворота качества для масок.

    - Score < 0.50 -> отклонить
    - 0.50 <= Score < 0.85 -> сохранить как draft
    - Score >= 0.85 -> активировать
    """

    def __init__(
        self,
        activation_threshold: float = 0.85,
        retry_threshold: float = 0.50
    ):
        self.activation_threshold = activation_threshold
        self.retry_threshold = retry_threshold

    def evaluate(self, score: float) -> Tuple[str, str]:
        """
        Оценка качества маски.

        Returns:
            (action, reason)
            action: "activate", "draft", "reject"
        """
        if score >= self.activation_threshold:
            return "activate", f"Score {score:.2f} >= {self.activation_threshold}"
        elif score >= self.retry_threshold:
            return "draft", f"Score {score:.2f} between {self.retry_threshold} and {self.activation_threshold}"
        else:
            return "reject", f"Score {score:.2f} < {self.retry_threshold}"
