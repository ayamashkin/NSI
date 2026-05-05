"""
LLM Mask Generator Module
Level 2: Автоматическая генерация regex масок с помощью LLM.
Модель/температура/system_prompt определяются автоматически по keywords из prompts.yaml.
"""

import re
import json
import logging
from pathlib import Path
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
    Автоматически определяет конфигурацию (service, model, temperature, system_prompt)
    по keywords из prompts.yaml — та же логика, что и в NomenclatureProcessor.
    """

    def __init__(
        self,
        clients: Dict[Any, Any],
        settings: Optional[Any] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.clients = clients
        self.settings = settings
        self.timeout = timeout
        self.max_retries = max_retries
        self.attempts: List[GenerationAttempt] = []

    # ==========================================================================
    # CLIENT HELPERS (поддержка строковых ключей и enum)
    # ==========================================================================

    def _has_client(self, provider: LLMProvider) -> bool:
        """
        Проверка наличия клиента для провайдера.
        Поддерживает строковые ключи (как в cli.py: 'mws', 'openwebui') и LLMProvider enum.
        """
        if provider in self.clients:
            return True
        if provider.value in self.clients:
            return True
        return False

    def _get_client(self, provider: LLMProvider):
        """
        Получение клиента для провайдера.
        Поддерживает строковые ключи и LLMProvider enum.
        """
        if provider in self.clients:
            return self.clients[provider]
        if provider.value in self.clients:
            return self.clients[provider.value]
        raise KeyError(f"No client for provider: {provider}")

    # ==========================================================================
    # KEYWORD-BASED PROMPT RESOLUTION (как в processor.py)
    # ==========================================================================

    def _match_keywords(self, name_lower: str, keywords: List[str]) -> bool:
        """
        Проверка совпадения keywords — точная копия логики из processor.py.
        Поддерживает: regex:, glob-шаблоны (*, ?), обычные подстроки.
        """
        for keyword in keywords:
            keyword = keyword.strip()

            # regex: префикс
            if keyword.startswith('regex:') or keyword.startswith('re:'):
                pattern = keyword.split(':', 1)[1].strip()
                try:
                    if re.search(pattern, name_lower, re.IGNORECASE):
                        return True
                except re.error as e:
                    logger.warning(f"[LLMMaskGenerator] Невалидный regex '{pattern}': {e}")
                    continue

            # glob-шаблоны
            elif '*' in keyword or '?' in keyword:
                pattern = keyword.replace('.', r'\.').replace('*', '.*').replace('?', '.')
                try:
                    if re.search(pattern, name_lower):
                        return True
                except re.error:
                    continue

            # простое вхождение подстроки
            else:
                if keyword.lower() in name_lower:
                    return True

        return False

    def _load_prompts_raw(self) -> Dict[str, Dict]:
        """Загрузка сырых данных prompts из prompts.yaml."""
        # Приоритет 1: settings.prompts
        if self.settings and hasattr(self.settings, 'prompts'):
            try:
                prompts = {}
                for pid, cfg in self.settings.prompts.items():
                    prompts[pid] = {
                        'keywords': getattr(cfg, 'keywords', []),
                        'service': getattr(cfg, 'service', 'openwebui'),
                        'model': getattr(cfg, 'model', 'qwen2.5:7b'),
                        'temperature': getattr(cfg, 'temperature', 0.1),
                        'system_prompt': getattr(cfg, 'system_prompt', None),
                    }
                return prompts
            except Exception as e:
                logger.warning(f"[LLMMaskGenerator] Ошибка чтения settings.prompts: {e}")

        # Приоритет 2: читаем prompts.yaml напрямую
        try:
            import yaml
            from pathlib import Path

            for path in ['config/prompts.yaml', 'prompts.yaml', '../config/prompts.yaml']:
                if Path(path).exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    return data.get('prompts', {})
        except Exception as e:
            logger.error(f"[LLMMaskGenerator] Ошибка чтения prompts.yaml: {e}")

        return {}

    def _resolve_prompt_by_keywords(
        self,
        item_type_or_name: str,
        name: Optional[str] = None,
        standard_type: Optional[str] = None,
        standard_normalized: Optional[str] = None,
        example_samples: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Каскадное определение prompt_id по keywords из 5 источников данных.

        Порядок поиска (от точного к общему):
        1. item_type     — "болт", "гайка", "труба"...
        2. standard_type — "ГОСТ", "ОСТ", "ТУ", "ISO", "DIN", "РАМ"
        3. standard      — "ГОСТ 7798-70" (нормализованный)
        4. examples      — первые 3 примера из ЕСН
        5. name          — полное наименование номенклатуры

        Args:
            item_type_or_name: Тип изделия
            name: Полное наименование номенклатуры
            standard_type: Тип стандарта (ГОСТ/ОСТ/ТУ/ISO/DIN/РАМ)
            standard_normalized: Нормализованное название стандарта
            example_samples: Примеры из ЕСН для keyword matching

        Returns:
            prompt_id (например, 'hardware', 'rolledMetal')
        """
        prompts_data = self._load_prompts_raw()

        if not prompts_data:
            logger.warning("[LLMMaskGenerator] Не удалось загрузить prompts, fallback на 'hardware'")
            return 'hardware'

        # КАСКАД 1-5: собираем источники по приоритету
        sources: List[Tuple[str, str]] = []  # (source_type, source_text)

        # 1. item_type (самый точный)
        if item_type_or_name and item_type_or_name.lower() not in ('unknown', 'none', ''):
            sources.append(('item_type', item_type_or_name))

        # 2. standard_type (ГОСТ → крепеж/металл, ТУ → custom)
        if standard_type and standard_type not in ('UNKNOWN', ''):
            sources.append(('standard_type', standard_type))

        # 3. standard_normalized (конкретный стандарт)
        if standard_normalized and standard_normalized.strip():
            sources.append(('standard', standard_normalized.strip()))

        # 4. examples (содержимое примеров из ЕСН)
        if example_samples:
            for i, sample in enumerate(example_samples[:3]):
                if sample and sample.strip():
                    sources.append((f'example#{i+1}', sample.strip()))

        # 5. name (полное наименование — самый общий)
        if name and name.strip():
            sources.append(('name', name.strip()))

        if not sources:
            logger.warning("[LLMMaskGenerator] Нет данных для keyword matching, fallback на 'hardware'")
            return 'hardware'

        # Поиск по всем источникам
        for source_type, source_text in sources:
            name_lower = source_text.lower()
            for prompt_id, cfg in prompts_data.items():
                keywords = cfg.get('keywords', [])
                if self._match_keywords(name_lower, keywords):
                    logger.info(
                        f"[LLMMaskGenerator] Keywords match by {source_type}: "
                        f"'{source_text[:50]}' → prompt_id='{prompt_id}'"
                    )
                    return prompt_id

        logger.warning(
            f"[LLMMaskGenerator] Нет совпадений по keywords ни по одному из "
            f"{len(sources)} источников, fallback на 'hardware'"
        )
        return 'hardware'

    # ==========================================================================
    # PROMPT CONFIGURATION
    # ==========================================================================

    def _load_skip_fields(self) -> set:
        """Загрузка skip_fields из ens_column_mapping.yaml."""
        default = {'код', 'mdm_key', 'единицы_измерения', 'наименование_типа.1',
                   'полное_наименование', 'наименование', 'нтд',
                   'наименование_типа',  # дублирует тип_изделия, используем только тип_изделия
                   }
        try:
            import yaml
            from pathlib import Path
            for path in ['config/ens_column_mapping.yaml', 'ens_column_mapping.yaml']:
                if Path(path).exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    fields = data.get('skip_fields', [])
                    if fields:
                        return set(fields)
                    break
        except Exception as e:
            logger.warning(f"[LLMMaskGenerator] Не удалось загрузить skip_fields: {e}")
        return default

    def _get_prompt_config(self, prompt_id: Optional[str] = None) -> Optional[Any]:
        """
        Получение конфигурации промпта из prompts.yaml.
        Если prompt_id не передан — используем fallback 'hardware'.
        """
        logger.debug(f"[LLMMaskGenerator] _get_prompt_config: prompt_id={prompt_id}")

        if not prompt_id:
            prompt_id = 'hardware'

        # Приоритет 1: settings
        if self.settings and hasattr(self.settings, 'prompts'):
            prompts = self.settings.prompts
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
                    if prompt_id in prompts_data:
                        cfg = prompts_data[prompt_id]
                        class SimplePromptConfig:
                            pass
                        result = SimplePromptConfig()
                        result.service = cfg.get('service', 'openwebui')
                        result.model = cfg.get('model', 'qwen2.5:7b')
                        result.temperature = cfg.get('temperature', 0.1)
                        result.system_prompt = cfg.get('system_prompt')
                        logger.debug(f"[LLMMaskGenerator] Загружен из {path}: {prompt_id} → {result.model}")
                        return result
                    break
        except Exception as e:
            logger.error(f"[LLMMaskGenerator] Ошибка чтения prompts.yaml: {e}")

        logger.error("[LLMMaskGenerator] Не удалось получить конфигурацию промпта")
        return None

    def _build_retry_config(
        self,
        prompt_id: Optional[str] = None,
        item_type: Optional[str] = None,
        name: Optional[str] = None,
        standard_type: Optional[str] = None,
        standard_normalized: Optional[str] = None,
        example_samples: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Формирование retry конфигурации.

        Логика:
        1. Если передан prompt_id — используем его
        2. Иначе — определяем по keywords из item_type или name
        3. Из prompts.yaml берём service, model, temperature, system_prompt
        """
        configs = []

        # Автоопределение prompt_id по keywords (каскадный поиск)
        if not prompt_id:
            prompt_id = self._resolve_prompt_by_keywords(
                item_type_or_name=item_type or "",
                name=name,
                standard_type=standard_type,
                standard_normalized=standard_normalized,
                example_samples=example_samples
            )
            source = item_type or (name[:40] if name else "unknown")
            logger.info(f"[LLMMaskGenerator] Auto-resolved prompt_id='{prompt_id}' from '{source}'")

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

                logger.info(
                    f"[LLMMaskGenerator] Конфиг из prompts.yaml/{prompt_id}: "
                    f"provider={prompt_cfg.service}, model={model}, temp={temp}"
                )

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
                    "temperature": round(min(temp + 0.2, 0.5), 2),
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

        # Last resort fallback — из раздела mask_generation в config.yaml
        if not configs:
            fallback_cfg = getattr(self.settings, 'mask_generation', None) if self.settings else None

            if fallback_cfg:
                # Явно указанный сервис и модель из config.yaml
                svc = getattr(fallback_cfg, 'default_service', 'mws')
                fallback_model = getattr(fallback_cfg, 'default_model', 'qwen2.5-72b-instruct')
                fallback_temp = getattr(fallback_cfg, 'default_temperature', 0.1)

                provider_map = {
                    'openwebui': LLMProvider.OPENWEBUI,
                    'mws': LLMProvider.MWS,
                    'gigachat': LLMProvider.GIGACHAT,
                }
                fallback_provider = provider_map.get(svc, LLMProvider.MWS)
                logger.info(f"[LLMMaskGenerator] Fallback из mask_generation: {svc}/{fallback_model}")
            else:
                # Хардкод-только если нет settings вообще
                fallback_provider = LLMProvider.MWS
                fallback_model = "qwen2.5-72b-instruct"
                fallback_temp = 0.1

            if self._has_client(fallback_provider):
                configs.append({
                    "provider": fallback_provider,
                    "model": fallback_model,
                    "temperature": fallback_temp,
                    "system_prompt": None,
                    "source": f"fallback:mask_generation:{fallback_provider.value}"
                })
            else:
                logger.error(f"[LLMMaskGenerator] Fallback клиент {fallback_provider.value} не инициализирован!")

        logger.info(f"[LLMMaskGenerator] Retry конфигурация: {len(configs)} попыток")
        for i, cfg in enumerate(configs, 1):
            logger.info(
                f"  {i}. {cfg['provider'].value}: {cfg['model']} "
                f"(temp={cfg['temperature']}, source={cfg['source']})"
            )

        return configs

    # ==========================================================================
    # MASK GENERATION
    # ==========================================================================

    def generate_mask(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict[str, Any]],
        context: Optional[Dict] = None,
        prompt_id: Optional[str] = None,
        name: Optional[str] = None,
        standard_info: Optional[Any] = None
    ) -> Tuple[Optional[Dict[str, Any]], List[GenerationAttempt]]:
        """
        Генерация маски с auto-resolution prompt_id по keywords (каскадный поиск).

        Args:
            standard: Стандарт (ГОСТ 7798-70, etc.)
            item_type: Тип изделия (болт, гайка, etc.)
            examples: Примеры номенклатуры для обучения
            context: Дополнительный контекст
            prompt_id: Явный prompt_id (автоопределение если None)
            name: Полное наименование (source #5 в каскаде)
            standard_info: Объект StandardInfo (source #2, #3 в каскаде)

        Returns:
            (mask_dict, attempts)
        """
        self.attempts = []

        logger.info(
            f"[LLMMaskGenerator] Начало генерации: standard={standard}, "
            f"item_type={item_type}, примеров={len(examples)}, prompt_id={prompt_id}"
        )

        # Извлекаем sample texts из examples для keyword matching (source #4)
        example_samples = [
            ex.get('полное_наименование') or ex.get('наименование', '')
            for ex in examples[:3]
        ]

        # Извлекаем данные из standard_info (source #2, #3)
        standard_type = None
        standard_normalized = None
        if standard_info:
            standard_type = getattr(standard_info, 'standard_type', None)
            if standard_type:
                standard_type = standard_type.value if hasattr(standard_type, 'value') else str(standard_type)
            standard_normalized = getattr(standard_info, 'normalized', None) or getattr(standard_info, 'full_name', None)

        # Строим retry-конфиг с каскадным keyword resolution
        retry_configs = self._build_retry_config(
            prompt_id=prompt_id,
            item_type=item_type,
            name=name,
            standard_type=standard_type,
            standard_normalized=standard_normalized,
            example_samples=example_samples
        )
        if not retry_configs:
            logger.error("[LLMMaskGenerator] Нет доступных конфигураций LLM")
            return None, []

        # === СОЗДАЕМ ПРОМПТ (один раз на все попытки) ===
        prompt = self._build_prompt(standard, item_type, examples, context)

        # Сохраняем промпт для отладки
        self._save_debug_prompt(standard, item_type, prompt)

        max_attempts = min(self.max_retries, len(retry_configs))

        for i in range(max_attempts):
            config = retry_configs[i]

            attempt = self._try_generate(
                attempt_number=i + 1,
                config=config,
                standard=standard,
                item_type=item_type,
                examples=examples,
                context=context,
                prompt=prompt
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
                logger.info(f"[LLMMaskGenerator] Успех на попытке {i+1} через {config['source']}")
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
        context: Optional[Dict],
        prompt: Optional[str] = None
    ) -> GenerationAttempt:
        """Одна попытка генерации маски с полным логированием."""
        provider = config["provider"]
        model = config["model"]
        temperature = config["temperature"]
        system_prompt = config.get("system_prompt")
        source = config.get("source", "unknown")

        logger.info(
            f"[LLMMaskGenerator] Попытка {attempt_number}: "
            f"{provider.value}/{model}, temp={temperature} (source: {source})"
        )

        # Промпт
        if prompt is None:
            prompt = self._build_prompt(standard, item_type, examples, context)

        # Проверка промпта перед вызовом
        if not prompt or not prompt.strip():
            logger.error(f"[LLMMaskGenerator] Попытка {attempt_number}: ПРОМПТ ПУСТОЙ!")
            return GenerationAttempt(attempt_number=attempt_number, provider=provider.value,
                model=model, temperature=temperature, success=False,
                error_message="Prompt is empty", raw_response="PROMPT_WAS_EMPTY")

        logger.info(f"[LLMMaskGenerator] Попытка {attempt_number}: prompt_len={len(prompt)}, prompt_preview={prompt[:100]}")

        # Вызов API с полным логированием
        raw_response = ""
        response = None
        try:
            client = self._get_client(provider)
            logger.info(f"[LLMMaskGenerator] Попытка {attempt_number}: ВЫЗОВ API {provider.value}")
            response = client.complete(
                prompt=prompt, model=model, temperature=temperature, system_prompt=system_prompt
            )
            logger.info(f"[LLMMaskGenerator] Попытка {attempt_number}: API ответ type={type(response)}, value={'None' if response is None else 'not None'}")

            if response is None:
                raw_response = "API_RESPONSE_WAS_NONE"
            elif isinstance(response, dict):
                raw_response = response.get("raw") or str(response)
                raw_len = len(raw_response) if raw_response is not None else 0
                logger.info(f"[LLMMaskGenerator] Попытка {attempt_number}: success={response.get('success')}, raw_len={raw_len}")
            else:
                raw_response = str(response)
                logger.info(f"[LLMMaskGenerator] Попытка {attempt_number}: response type={type(response)}, str_len={len(raw_response)}")

        except Exception as e:
            import traceback
            raw_response = f"EXCEPTION: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"[LLMMaskGenerator] Попытка {attempt_number}: ИСКЛЮЧЕНИЕ: {e}")
        finally:
            # ВСЕГДА сохраняем ответ (защита от None)
            safe_response = raw_response if raw_response is not None else "RAW_RESPONSE_WAS_NONE"
            self._save_debug_response(standard, item_type, attempt_number, safe_response)
            logger.info(f"[LLMMaskGenerator] Попытка {attempt_number}: ответ СОХРАНЕН, length={len(safe_response)}")

        # Обработка результата
        if raw_response.startswith("API_RESPONSE_WAS_NONE"):
            return GenerationAttempt(attempt_number=attempt_number, provider=provider.value,
                model=model, temperature=temperature, success=False,
                error_message="API returned None", raw_response=raw_response[:500])

        if raw_response.startswith("EXCEPTION:"):
            return GenerationAttempt(attempt_number=attempt_number, provider=provider.value,
                model=model, temperature=temperature, success=False,
                error_message=raw_response[:200], raw_response=raw_response[:500])

        if isinstance(response, dict) and not response.get("success"):
            # API сообщил об ошибке, но raw может содержать JSON
            logger.warning(f"[LLMMaskGenerator] Попытка {attempt_number}: API success=False, но пробуем парсить raw")
            # Продолжаем к парсингу ниже — не возвращаем ошибку

        # Парсинг JSON
        content = response.get("content") if isinstance(response, dict) else None
        result = content if isinstance(content, dict) else self._extract_json(raw_response)

        if result and "pattern" in result:
            logger.info(f"[LLMMaskGenerator] Попытка {attempt_number}: УСПЕХ")
            return GenerationAttempt(attempt_number=attempt_number, provider=provider.value,
                model=model, temperature=temperature, success=True, pattern=result["pattern"],
                params=result.get("params", []), required=result.get("required", []),
                raw_response=raw_response[:500])
        else:
            fail_reason = "JSON не найден" if result is None else "Нет поля 'pattern'"
            preview = raw_response[:300].replace('\n', ' ')
            logger.error(f"[LLMMaskGenerator] Попытка {attempt_number}: {fail_reason}. Preview: {preview}")
            return GenerationAttempt(attempt_number=attempt_number, provider=provider.value,
                model=model, temperature=temperature, success=False,
                error_message=fail_reason, raw_response=raw_response[:500])

    # ==========================================================================
    # PROMPT BUILDER (промпт для генерации маски)
    # ==========================================================================

    def _load_prompt_template(self) -> str:
        """Загрузка шаблона промпта из файла (путь из settings)."""
        # Путь из settings -> mask_generation.prompt_template
        if self.settings and hasattr(self.settings, 'mask_generation'):
            mg = self.settings.mask_generation
            template_path = getattr(mg, 'prompt_template', None)
            if template_path and Path(template_path).exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()

        # Fallback: ищем по стандартным путям
        for path in [
            'prompts/templates/mask_generation.txt',
            'config/prompts/templates/mask_generation.txt',
            '../prompts/templates/mask_generation.txt',
        ]:
            if Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()

        # Hardcoded fallback (если файл не найден)
        logger.warning("[LLMMaskGenerator] Шаблон промпта не найден, используем fallback")
        return self._default_prompt_template()

    def _sanitize_filename(self, text: str) -> str:
        """Очистка строки для использования в имени файла."""
        import re
        # Заменяем недопустимые символы на _
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', str(text))
        # Убираем множественные _
        sanitized = re.sub(r'_+', '_', sanitized)
        # Обрезаем до 80 символов
        return sanitized.strip('_')[:80]

    def _save_debug_prompt(self, standard: str, item_type: str, prompt: str):
        """Сохранение промпта в файл для отладки."""
        self._save_debug_file(standard, item_type, "prompt", prompt)

    def _save_debug_response(self, standard: str, item_type: str, attempt: int, response: str):
        """Сохранение raw ответа LLM для отладки."""
        self._save_debug_file(standard, item_type, f"response_a{attempt}", response)

    def _save_debug_file(self, standard: str, item_type: str, suffix: str, content: str):
        """Сохранение произвольного файла отладки."""
        save_enabled = False
        debug_dir = "prompts/debug"

        if self.settings and hasattr(self.settings, 'mask_generation'):
            mg = self.settings.mask_generation
            save_enabled = getattr(mg, 'save_debug_prompts', False)
            debug_dir = getattr(mg, 'debug_prompts_dir', 'prompts/debug')

        if not save_enabled:
            return

        try:
            from pathlib import Path
            from datetime import datetime

            Path(debug_dir).mkdir(parents=True, exist_ok=True)

            safe_type = self._sanitize_filename(item_type or "unknown")
            safe_std = self._sanitize_filename(standard or "unknown")
            # Имя по маске: [тип]_[стандарт].txt
            # Для response добавляем _a{N}
            if suffix == "prompt":
                filename = f"{safe_type}_{safe_std}.txt"
            else:
                # suffix = "response_a1" -> "_a1"
                attempt_suffix = suffix.replace("response", "")
                filename = f"{safe_type}_{safe_std}{attempt_suffix}.txt"

            filepath = Path(debug_dir) / filename

            header = (
                f"# Тип: {item_type}\n"
                f"# Стандарт: {standard}\n"
                f"# Время: {datetime.now().isoformat()}\n"
                f"# {'=' * 50}\n\n"
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write(content)

            logger.info(f"[LLMMaskGenerator] Файл сохранен: {filepath}")
        except Exception as e:
            logger.warning(f"[LLMMaskGenerator] Не удалось сохранить файл: {e}")

    def _default_prompt_template(self) -> str:
        """Fallback шаблон если файл не найден."""
        return (
            "Ты — эксперт по техническим стандартам ГОСТ и регулярным выражениям Python.\n\n"
            "ЗАДАЧА: Создай regex-паттерн с named groups (?P<name>...) "
            "для извлечения параметров из номенклатуры типа \"{item_type}\" по стандарту {standard}.\n\n"
            "Примеры:\n{examples_text}\n\n"
            "Статистика:\n{stats_text}\n\n"
            "### Формат ответа\n"
            "```json\n"
            "{{\n"
            "  \"pattern\": \"...\",\n"
            "  \"params\": {params_list},\n"
            "  \"required\": {required_list}\n"
            "}}\n"
            "```\n\n"
            "### Строгое соответствие вывода\n"
            "Выведите результат в виде одного JSON-объекта. "
            "Не добавляйте в ответ никаких других объектов или комментариев, кроме итогового JSON."
        )

    def _build_prompt(self, standard, item_type, examples, context):
        """
        Построение промпта для LLM-генерации regex-маски.
        Шаблон загружается из файла (prompts/templates/mask_generation.txt).
        """
        template = self._load_prompt_template()

        # --- Алиасы полей: если 'тип_изделия' отсутствует, берём из 'наименование_типа' ---
        field_aliases = {'тип_изделия': 'наименование_типа'}

        # Применяем алиасы к examples (создаём 'тип_изделия' из 'наименование_типа' если нужно)
        aliased_examples = []
        for ex in examples:
            new_ex = dict(ex)
            for target, source in field_aliases.items():
                if target not in new_ex or not new_ex.get(target):
                    if source in new_ex and new_ex.get(source):
                        new_ex[target] = new_ex[source]
            aliased_examples.append(new_ex)
        examples = aliased_examples

        # --- Анализ полей ЕСН ---
        field_stats = {}
        for ex in examples:
            for key, val in ex.items():
                if key.startswith('_'):
                    continue
                if val is None or (isinstance(val, str) and not val.strip()):
                    continue
                field_stats[key] = field_stats.get(key, 0) + 1

        total = len(examples)
        threshold = max(1, int(total * 0.1))
        sorted_fields = sorted(field_stats.items(), key=lambda x: -x[1])
        relevant_fields = [k for k, v in sorted_fields if v >= threshold]

        # Убеждаемся что тип_изделия всегда первым в списке (заменяет наименование_типа)
        if 'тип_изделия' not in relevant_fields:
            relevant_fields.insert(0, 'тип_изделия')
        else:
            # Перемещаем тип_изделия на первое место
            relevant_fields.remove('тип_изделия')
            relevant_fields.insert(0, 'тип_изделия')

        # Загружаем skip_fields из конфига (ens_column_mapping.yaml)
        skip_fields = self._load_skip_fields()

        def clean_name(n: str, max_len: int = 30) -> str:
            """Очистка имени поля для regex group name + ограничение длины."""
            result = n.replace('.', '_').replace('-', '_').replace('(', '_').replace(')', '_').replace(',', '_')
            while '__' in result:
                result = result.replace('__', '_')
            result = result.strip('_')
            if len(result) > max_len:
                result = result[:max_len].rstrip('_')
            return result

        # Очищаем имена: оригинальное поле -> имя для regex
        field_name_map = {}
        seen_names = set()
        for f in relevant_fields:
            if f not in skip_fields:
                cleaned = clean_name(f)
                # Обработка дубликатов после обрезки
                original_cleaned = cleaned
                suffix = 2
                while cleaned in seen_names:
                    suffix_str = f"_{suffix}"
                    cleaned = original_cleaned[:max_len - len(suffix_str)] + suffix_str
                    suffix += 1
                seen_names.add(cleaned)
                field_name_map[f] = cleaned

        # Убеждаемся что тип_изделия есть в маппинге (ключевое поле)
        if 'тип_изделия' not in field_name_map:
            field_name_map['тип_изделия'] = 'тип_изделия'

        regex_fields = list(field_name_map.values())
        display_fields = [f for f in relevant_fields if f not in skip_fields][:15]

        # --- Форматирование примеров ---
        # Берём разнообразные примеры: с пропущенными параметрами, разные покрытия и т.д.
        sample_examples = self._select_diverse_examples(examples, n=10)
        examples_lines = []
        for i, ex in enumerate(sample_examples, 1):
            name = ex.get('полное_наименование') or ex.get('наименование', '')
            if not name:
                continue
            # Всегда показываем полное_наименование (даже если в skip_fields)
            filled_fields = [f"    полное_наименование: {name}"]
            for field in display_fields:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    filled_fields.append(f"    {field}: {val}")

            # Отмечаем пропущенные параметры (важно для опциональности в regex)
            missing_fields = [f for f in display_fields if not ex.get(f) or not str(ex.get(f)).strip()]
            if missing_fields:
                filled_fields.append(f"    [ПРОПУЩЕНЫ: {', '.join(missing_fields)}]")

            # Структура: разбиваем строку на части по найденным параметрам
            structure_parts = []
            for field in display_fields:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    group_name = field_name_map.get(field, field)
                    structure_parts.append(f"(?P<{group_name}>{val})")

            structure_str = ' '.join(structure_parts)

            examples_lines.append(
                f"{i}. ИСХОДНАЯ СТРОКА: \"{name}\"\n"
                f"   СТРУКТУРА: [тип_изделия] [разделители] " + structure_str + "\n"
                f"   ПОЛЯ ЕСН:\n" + "\n".join(filled_fields)
            )

        examples_text = "\n\n".join(examples_lines) if examples_lines else "Нет примеров"

        # --- Статистика ---
        stats_lines = [f"    {field_name_map.get(k, k)}: {field_stats.get(k, total)} из {total}" for k in field_name_map]
        stats_text = "\n".join(stats_lines) if stats_lines else "Нет статистики"

        # --- Подстановка в шаблон ---
        # params_list: JSON-список очищенных имён
        import json
        params_list = json.dumps(regex_fields, ensure_ascii=False)

        # required_list: JSON-список (все кроме опциональных)
        optional_params = {'исполнение', 'покрытие', 'марка_материала'}
        required_fields = [f for f in regex_fields if f not in optional_params]
        required_list = json.dumps(required_fields, ensure_ascii=False)

        # params_hint: строка для подсказки
        params_hint = ", ".join(regex_fields[:10])

        # context_text
        context_text = context.get('context', '') if context else ''

        # Подстановка через str.replace (НЕ format! из-за {} в данных)
        result = template
        replacements = {
            "{item_type}": item_type,
            "{standard}": standard,
            "{example_count}": str(len(sample_examples)),
            "{examples_text}": examples_text,
            "{stats_text}": stats_text,
            "{params_hint}": params_hint,
            "{params_list}": params_list,
            "{required_list}": required_list,
            "{context_text}": context_text,
        }
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        # Проверяем, остались ли неподставленные placeholder'ы
        remaining = [m for m in re.finditer(r'\{[a-z_]+\}', result) if m.group() not in ('{{', '}}')]
        if remaining:
            logger.warning(f"[LLMMaskGenerator] Неподставленные placeholder'ы: {[m.group() for m in remaining[:5]]}")

        return result

    @staticmethod
    def _select_diverse_examples(examples: List[Dict], n: int = 10) -> List[Dict]:
        """
        Выбор разнообразных примеров для промпта.

        Цель: показать LLM записи с РАЗНЫМ набором параметров:
        - С полным набором (все поля заполнены)
        - С пропущенным исполнением
        - С пропущенным покрытием
        - С пропущенным диаметром (если есть)

        Это позволяет LLM сделать параметры опциональными через ? в regex.
        """
        if len(examples) <= n:
            return examples

        selected = []
        seen_patterns = set()

        # 1. Пример с максимальным набором полей (первый, обычно полный)
        selected.append(examples[0])
        seen_patterns.add('full')

        # 2. Ищем примеры с пропущенными полями
        key_fields = ['исполнение', 'покрытие', 'тип_резьбы', 'марка_материала', 'шаг_резьбы', 'номинальный_диаметр_резьбы']

        for field in key_fields:
            for ex in examples[1:]:
                pattern_key = f"missing_{field}"
                if pattern_key in seen_patterns:
                    continue
                # Проверяем что это поле пропущено, но другие важные есть
                if not ex.get(field) or not str(ex.get(field)).strip():
                    # Проверяем что не все поля пустые (иначе бесполезный пример)
                    has_some = any(ex.get(f) and str(ex.get(f)).strip() for f in key_fields if f != field)
                    if has_some:
                        selected.append(ex)
                        seen_patterns.add(pattern_key)
                        break
            if len(selected) >= n // 2:
                break

        # 3. Дополняем случайными разнообразными до n
        import random
        random.seed(42)  # воспроизводимость
        remaining = [ex for ex in examples if ex not in selected]
        random.shuffle(remaining)

        # Добавляем до n штук
        needed = n - len(selected)
        if remaining and needed > 0:
            selected.extend(remaining[:needed])

        return selected[:n]

    def _preprocess_json_text(self, text: str) -> str:
        r"""
        Предобработка JSON текста от LLM.
        LLM часто генерирует regex с одиночными backslash (\s, \d, \w) внутри JSON-строк,
        что делает JSON невалидным (JSON допускает только \\, \", \/, \b, \f, \n, \r, \t, \uXXXX).
        Экранируем regex-escapes, сохраняя уже двойные (\\) нетронутыми.
        """
        import re
        # Placeholder для уже двойных backslash
        placeholder = '\x00DBL\x00'
        result = text.replace('\\\\', placeholder)
        # Экранируем одиночные regex backslash: \s -> \\s, \d -> \\d и т.д.
        result = re.sub(r'\\([sdwSDWbB])', r'\\\\\1', result)
        # Восстанавливаем двойные
        result = result.replace(placeholder, '\\\\')
        return result

    def _extract_json(self, text):
        """Извлечение JSON из ответа LLM. Приоритет: ```json блоки, затем все {...}."""
        if not text or not text.strip():
            return None

        # Исправляем невалидные JSON escape от LLM
        text = self._preprocess_json_text(text)

        candidates = []

        # === Стратегия 1: Code blocks ```json ... ``` (приоритетная) ===
        parts = text.split('```')
        for i in range(1, len(parts), 2):  # берем содержимое между ```
            block = parts[i].strip()
            if block.startswith('json'):
                block = block[4:].strip()
            brace_start = block.find('{')
            if brace_start >= 0:
                depth = 0
                for j, ch in enumerate(block[brace_start:], brace_start):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(block[brace_start:j+1])
                                if isinstance(obj, dict):
                                    score = 0
                                    if "pattern" in obj: score += 100
                                    if "params" in obj: score += 10
                                    candidates.append((score + 50, obj))
                            except json.JSONDecodeError:
                                pass
                            break

        # === Стратегия 2: Все {...} на балансировке скобок ===
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = text[start:i+1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            score = 0
                            if "pattern" in obj: score += 100
                            if "params" in obj: score += 10
                            if "required" in obj: score += 10
                            candidates.append((score, obj))
                    except json.JSONDecodeError:
                        pass

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            best = candidates[0][1]
            logger.info(f"[LLMMaskGenerator] JSON найден: score={candidates[0][0]}, keys={list(best.keys())}")
            return best

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