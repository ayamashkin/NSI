"""
LLM Mask Generator Module
Level 2: Автоматическая генерация regex масок с помощью LLM.
Модель/температура/system_prompt берутся из prompts.yaml
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
    Генератор масок через LLM.
    Модели берутся из prompts.yaml (model, temperature, system_prompt, service)
    """

    def __init__(
        self,
        clients: Dict[LLMProvider, Any],
        settings: Optional[Any] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.clients = clients
        self.settings = settings
        self.timeout = timeout
        self.max_retries = max_retries
        self.attempts: List[GenerationAttempt] = []

    def _get_prompt_config(self, prompt_id: Optional[str] = None) -> Optional[Any]:
        """Получение конфигурации промпта из prompts.yaml или напрямую из файла."""
        logger.debug(f"[LLMMaskGenerator] _get_prompt_config: prompt_id={prompt_id}, settings={self.settings is not None}")

        # Приоритет 1: settings
        if self.settings and hasattr(self.settings, 'prompts'):
            prompts = self.settings.prompts
            if not prompt_id:
                prompt_id = 'hardware'
            result = prompts.get(prompt_id) if hasattr(prompts, 'get') else None
            if result:
                logger.debug(f"[LLMMaskGenerator] Найден в settings.prompts: {prompt_id}")
                return result

        # Приоритет 2: читаем prompts.yaml напрямую
        logger.info("[LLMMaskGenerator] settings недоступен, читаем prompts.yaml напрямую")
        try:
            import yaml
            from pathlib import Path

            for path in ['config/prompts.yaml', 'prompts.yaml', '../config/prompts.yaml']:
                if Path(path).exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    prompts_data = data.get('prompts', {})
                    if not prompt_id:
                        prompt_id = 'hardware'
                    if prompt_id in prompts_data:
                        cfg = prompts_data[prompt_id]
                        # Создаем простой объект с нужными атрибутами
                        class SimplePromptConfig:
                            pass
                        result = SimplePromptConfig()
                        result.service = cfg.get('service', 'openwebui')
                        result.model = cfg.get('model', 'qwen2.5:7b')
                        result.temperature = cfg.get('temperature', 0.1)
                        result.system_prompt = cfg.get('system_prompt')
                        logger.debug(f"[LLMMaskGenerator] Загружен из {path}: {prompt_id} -> {result.model}")
                        return result
                    break
        except Exception as e:
            logger.error(f"[LLMMaskGenerator] Ошибка чтения prompts.yaml: {e}")

        logger.error("[LLMMaskGenerator] Не удалось получить конфигурацию промпта")
        return None

    def _build_retry_config(self, prompt_id: Optional[str] = None) -> List[Dict]:
        """Формирование retry конфигурации из prompts.yaml."""
        configs = []

        # Приоритет 1: prompts.yaml
        prompt_cfg = self._get_prompt_config(prompt_id)
        if prompt_cfg:
            provider_map = {
                'openwebui': LLMProvider.OPENWEBUI,
                'mws': LLMProvider.MWS,
                'gigachat': LLMProvider.GIGACHAT,
            }
            provider = provider_map.get(prompt_cfg.service)

            if provider and self._has_client(provider):
                model = prompt_cfg.model
                temp = prompt_cfg.temperature
                system_prompt = getattr(prompt_cfg, 'system_prompt', None)

                logger.debug(f"[LLMMaskGenerator] Используем prompts.yaml/{prompt_id}: model={model}, temp={temp}, service={prompt_cfg.service}")

                configs.append({
                    "provider": provider,
                    "model": model,
                    "temperature": temp,
                    "system_prompt": system_prompt,
                    "source": f"prompts.yaml:{prompt_id}"
                })
                configs.append({
                    "provider": provider,
                    "model": model,
                    "temperature": min(temp + 0.2, 0.5),
                    "system_prompt": system_prompt,
                    "source": f"prompts.yaml:{prompt_id}(retry)"
                })

        # Fallback: config.yaml api.*.default_model
        if not configs and self.settings and hasattr(self.settings, 'api'):
            for provider in [LLMProvider.OPENWEBUI, LLMProvider.MWS, LLMProvider.GIGACHAT]:
                service_name = provider.value
                if service_name in self.settings.api and self._has_client(provider):
                    api_cfg = self.settings.api[service_name]
                    model = getattr(api_cfg, 'default_model', None)
                    if model and self._has_client(provider):
                        configs.append({
                            "provider": provider,
                            "model": model,
                            "temperature": 0.1,
                            "system_prompt": None,
                            "source": f"config.yaml:{service_name}"
                        })

        # Last resort fallback
        if not configs:
            if self._has_client(LLMProvider.OPENWEBUI):
                configs.append({
                    "provider": LLMProvider.OPENWEBUI,
                    "model": "qwen2.5:7b",
                    "temperature": 0.1,
                    "system_prompt": None,
                    "source": "fallback"
                })

        logger.debug(f"[LLMMaskGenerator] Retry конфигурация: {len(configs)} попыток")
        for i, cfg in enumerate(configs, 1):
            logger.debug(f"  {i}. {cfg['provider'].value}: {cfg['model']} (temp={cfg['temperature']}, source={cfg['source']})")

        return configs

    def generate_mask(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict[str, Any]],
        context: Optional[Dict] = None,
        prompt_id: Optional[str] = None,
        name: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], List[GenerationAttempt]]:
        """Генерация маски с retry стратегией."""
        self.attempts = []

        logger.debug(f"[LLMMaskGenerator] Начало: {standard}/{item_type}, примеров={len(examples)}, prompt_id={prompt_id}")

        retry_configs = self._build_retry_config(prompt_id)
        if not retry_configs:
            logger.error("[LLMMaskGenerator] Нет доступных конфигураций LLM")
            return None, []

        max_attempts = min(self.max_retries, len(retry_configs))

        for i in range(max_attempts):
            config = retry_configs[i]

            attempt = self._try_generate(
                attempt_number=i + 1,
                config=config,
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
                    "source": config.get("source", "unknown"),
                    "auto_score": 0.0
                }
                logger.debug(f"[LLMMaskGenerator] Успех на попытке {i+1}")
                return mask, self.attempts

        logger.error(f"[LLMMaskGenerator] Все {max_attempts} попыток неудачны")
        return None, self.attempts

    def _try_generate(
        self,
        attempt_number: int,
        config: Dict,
        standard: str,
        item_type: str,
        examples: List[Dict[str, Any]],
        context: Optional[Dict]
    ) -> GenerationAttempt:
        """Одна попытка генерации."""
        provider = config["provider"]
        model = config["model"]
        temperature = config["temperature"]
        system_prompt = config.get("system_prompt")
        source = config.get("source", "unknown")

        logger.debug(f"[LLMMaskGenerator] Попытка {attempt_number}: {provider.value}/{model}, temp={temperature} ({source})")

        client = self._get_client(provider)
        prompt = self._build_prompt(standard, item_type, examples, context)

        try:
            response = client.complete(
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_prompt=system_prompt
                # timeout передается через конструктор клиента
            )

            if response is None:
                return GenerationAttempt(
                    attempt_number=attempt_number,
                    provider=provider.value,
                    model=model,
                    temperature=temperature,
                    success=False,
                    error_message="API returned None"
                )

            if not response.get("success"):
                error = response.get("error", "Unknown API error")
                return GenerationAttempt(
                    attempt_number=attempt_number,
                    provider=provider.value,
                    model=model,
                    temperature=temperature,
                    success=False,
                    raw_response=str(response.get("raw", ""))[:500],
                    error_message=f"API error: {error}"
                )

            raw_response = response.get("raw", "")
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
                    raw_response=raw_response[:500]
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
            logger.error(f"[LLMMaskGenerator] Попытка {attempt_number} failed: {e}")
            return GenerationAttempt(
                attempt_number=attempt_number,
                provider=provider.value,
                model=model,
                temperature=temperature,
                success=False,
                error_message=str(e)
            )

    def _build_prompt(self, standard, item_type, examples, context):
        """Построение промпта."""
        sample_examples = examples[:5]
        examples_text = "\n".join([
            f"{i+1}. {ex.get('полное_наименование') or ex.get('наименование', '')}"
            for i, ex in enumerate(sample_examples)
        ])

        context_text = ""
        if context:
            context_text = f"\nДоп. контекст: {json.dumps(context, ensure_ascii=False)}\n"

        return f"""Ты - эксперт по техническим стандартам и регулярным выражениям.

Создай regex-паттерн для извлечения параметров из номенклатуры {item_type} по стандарту {standard}.

Примеры:
{examples_text}
{context_text}
Требования:
1. Используй named groups (?P<name>...)
2. Поддерживай разные варианты написания
3. Учитывай специфику стандарта {standard}

Ответь ТОЛЬКО в формате JSON:
{{
  "pattern": "regex with named groups",
  "params": ["param1", "param2"],
  "required": ["required1"]
}}
"""

    def _extract_json(self, text):
        """Извлечение JSON из текста."""
        import re

        for pattern in [r"```json\s*(.*?)```", r"```\s*(.*?)```", r"\{[\s\S]*\}"]:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    return json.loads(match.group(1).strip() if match.lastindex else match.group(0))
                except json.JSONDecodeError:
                    pass
        return None


class MaskQualityGate:
    """Ворота качества для масок."""

    def __init__(self, activation_threshold=0.85, retry_threshold=0.50):
        self.activation_threshold = activation_threshold
        self.retry_threshold = retry_threshold

    def evaluate(self, score: float) -> Tuple[str, str]:
        """Оценка качества маски."""
        if score >= self.activation_threshold:
            return "activate", f"Score {score:.2f} >= {self.activation_threshold}"
        elif score >= self.retry_threshold:
            return "draft", f"Score {score:.2f} between {self.retry_threshold} and {self.activation_threshold}"
        else:
            return "reject", f"Score {score:.2f} < {self.retry_threshold}"