"""
LLM Mask Generator Module
Level 2: Автоматическая генерация regex масок с помощью LLM.
Модель/температура/system_prompt определяются автоматически по keywords из prompts.yaml.

LAST_FIXES:
  - 2026-05-12 14:00 UTC+3 — LLMProvider: добавлен MTS_AI; resolve_service/resolve_model из settings
  - 2026-05-12 13:50 UTC+3 — _get_prompt_config: убраны жесткие fallback 'openwebui'/'qwen2.5:7b'
  - 2026-05-12 13:45 UTC+3 — _build_retry_config: resolve service/model через mask_generation defaults
  - 2026-05-07 08:28 UTC+3 — _preprocess_json_text: escape ALL regex backslashes
  - 2026-05-07 08:25 UTC+3 — Базовая структура llm_mask_generator
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
    MTS_AI = "mts_ai"


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
    Правила выбора сервиса/модели:
      1. Если в prompts.yaml указаны service/model → используем их
      2. Если не указаны → берем из mask_generation.default_service/default_model
      3. Если нет подходящего правила → берем из mask_generation defaults
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
    # CLIENT HELPERS
    # ==========================================================================

    def _has_client(self, provider: LLMProvider) -> bool:
        """Проверка наличия клиента для провайдера."""
        if provider in self.clients:
            return True
        if provider.value in self.clients:
            return True
        return False

    def _get_client(self, provider: LLMProvider):
        """Получение клиента для провайдера."""
        if provider in self.clients:
            return self.clients[provider]
        if provider.value in self.clients:
            return self.clients[provider.value]
        raise KeyError(f"No client for provider: {provider}")

    # ==========================================================================
    # KEYWORD-BASED PROMPT RESOLUTION
    # ==========================================================================

    def _match_keywords(self, name_lower: str, keywords: List[str]) -> bool:
        """Проверка совпадения keywords."""
        for keyword in keywords:
            keyword = str(keyword).strip() if keyword else ""
            if not keyword:
                continue

            if keyword.startswith('regex:') or keyword.startswith('re:'):
                pattern = keyword.split(':', 1)[1].strip()
                try:
                    if re.search(pattern, name_lower, re.IGNORECASE):
                        return True
                except re.error as e:
                    logger.warning(f"[LLMMaskGenerator] Невалидный regex '{pattern}': {e}")
                    continue

            elif '*' in keyword or '?' in keyword:
                pattern = keyword.replace('.', r'\.').replace('*', '.*').replace('?', '.')
                try:
                    if re.search(pattern, name_lower):
                        return True
                except re.error:
                    continue
            else:
                if keyword.lower() in name_lower:
                    return True

        return False

    def _load_prompts_raw(self) -> Dict[str, Dict]:
        """Загрузка сырых данных prompts из prompts.yaml."""
        if self.settings and hasattr(self.settings, 'prompts'):
            try:
                prompts = {}
                for pid, cfg in self.settings.prompts.items():
                    prompts[pid] = {
                        'keywords': getattr(cfg, 'keywords', []),
                        'service': getattr(cfg, 'service', None),
                        'model': getattr(cfg, 'model', None),
                        'temperature': getattr(cfg, 'temperature', 0.1),
                        'system_prompt': getattr(cfg, 'system_prompt', None),
                    }
                return prompts
            except Exception as e:
                logger.warning(f"[LLMMaskGenerator] Ошибка чтения settings.prompts: {e}")

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
        """Каскадное определение prompt_id по keywords из 5 источников."""
        prompts_data = self._load_prompts_raw()

        if not prompts_data:
            logger.warning("[LLMMaskGenerator] Не удалось загрузить prompts, fallback на 'hardware'")
            return 'hardware'

        sources: List[Tuple[str, str]] = []

        if item_type_or_name and item_type_or_name.lower() not in ('unknown', 'none', ''):
            sources.append(('item_type', item_type_or_name))
        if standard_type and standard_type not in ('UNKNOWN', ''):
            sources.append(('standard_type', standard_type))
        if standard_normalized and standard_normalized.strip():
            sources.append(('standard', standard_normalized.strip()))
        if example_samples:
            for i, sample in enumerate(example_samples[:3]):
                if sample and sample.strip():
                    sources.append((f'example#{i+1}', sample.strip()))
        if name and name.strip():
            sources.append(('name', name.strip()))

        if not sources:
            logger.warning("[LLMMaskGenerator] Нет данных для keyword matching, fallback на 'hardware'")
            return 'hardware'

        for source_type, source_text in sources:
            name_lower = source_text.lower()
            for prompt_id, cfg in prompts_data.items():
                keywords = cfg.get('keywords', [])
                if self._match_keywords(name_lower, keywords):
                    logger.info(
                        f"[LLMMaskGenerator] Keywords match by {source_type}: "
                        f"'{source_text[:50]}' -> prompt_id='{prompt_id}'"
                    )
                    return prompt_id

        logger.warning(
            f"[LLMMaskGenerator] Нет совпадений по keywords, fallback на 'hardware'"
        )
        return 'hardware'

    # ==========================================================================
    # PROMPT CONFIGURATION (с resolve_service / resolve_model)
    # ==========================================================================

    def _load_skip_fields(self) -> set:
        """Загрузка skip_fields из ens_column_mapping.yaml."""
        default = {'код', 'mdm_key', 'единицы_измерения', 'наименование_типа.1',
                   'полное_наименование', 'наименование', 'нтд',
                   'наименование_типа'}
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

    def _resolve_service_model(self, prompt_cfg: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Определяет финальный service и model по правилам:
          1. Если в prompt_cfg указаны service/model (не None) → используем
          2. Иначе → берем из mask_generation.default_service/default_model
        """
        # Получаем raw значения (могут быть None)
        service = getattr(prompt_cfg, 'service', None)
        model = getattr(prompt_cfg, 'model', None)

        logger.debug(f"[LLMMaskGenerator] Raw from prompt: service={service}, model={model}")

        # Fallback на mask_generation defaults
        if (not service or not model) and self.settings and hasattr(self.settings, 'mask_generation'):
            mg = self.settings.mask_generation
            if not service:
                service = getattr(mg, 'default_service', None)
                logger.debug(f"[LLMMaskGenerator] Service resolved from mask_generation: {service}")
            if not model:
                model = getattr(mg, 'default_model', None)
                logger.debug(f"[LLMMaskGenerator] Model resolved from mask_generation: {model}")

        return service, model

    def _get_prompt_config(self, prompt_id: Optional[str] = None) -> Optional[Any]:
        """Получение конфигурации промпта из prompts.yaml."""
        logger.debug(f"[LLMMaskGenerator] _get_prompt_config: prompt_id={prompt_id}")

        if not prompt_id:
            prompt_id = 'hardware'

        # Приоритет 1: settings.prompts (PromptConfig объекты с resolve_service/model)
        if self.settings and hasattr(self.settings, 'prompts'):
            try:
                prompts = self.settings.prompts
                if hasattr(prompts, 'get'):
                    result = prompts.get(prompt_id)
                    if result:
                        logger.debug(f"[LLMMaskGenerator] Найден в settings.prompts: {prompt_id}")
                        return result
            except Exception as e:
                logger.warning(f"[LLMMaskGenerator] Ошибка чтения settings.prompts: {e}")

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

                        # Создаем объект без жестких fallback'ов
                        class ResolvedPromptConfig:
                            pass

                        result = ResolvedPromptConfig()
                        result.keywords = cfg.get('keywords', [])
                        result.service = cfg.get('service')  # None если не указан!
                        result.model = cfg.get('model')      # None если не указан!
                        result.temperature = cfg.get('temperature', 0.1)
                        result.system_prompt = cfg.get('system_prompt')

                        # Resolve через mask_generation defaults
                        resolved_svc, resolved_mdl = self._resolve_service_model(result)
                        if resolved_svc:
                            result.service = resolved_svc
                        if resolved_mdl:
                            result.model = resolved_mdl

                        logger.debug(
                            f"[LLMMaskGenerator] Загружен из {path}: "
                            f"{prompt_id} -> service={result.service}, model={result.model}"
                        )
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
        Формирование retry конфигурации с единым resolved сервисом.

        Правила:
          1. prompt_id найден в prompts.yaml + service/model указаны → используем
          2. prompt_id найден, service/model НЕ указаны → mask_generation defaults
          3. prompt_id НЕ найден → mask_generation defaults
        """
        configs = []

        # Автоопределение prompt_id
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

        # --- Правило 1 и 2: prompt_id найден ---
        prompt_cfg = self._get_prompt_config(prompt_id)

        if prompt_cfg:
            # Resolve service/model (с fallback на mask_generation)
            resolved_service, resolved_model = self._resolve_service_model(prompt_cfg)
            temp = getattr(prompt_cfg, 'temperature', 0.1)
            system_prompt = getattr(prompt_cfg, 'system_prompt', None)

            # Определяем provider
            provider_map = {
                'openwebui': LLMProvider.OPENWEBUI,
                'mws': LLMProvider.MWS,
                'gigachat': LLMProvider.GIGACHAT,
                'mts_ai': LLMProvider.MTS_AI,
            }
            provider = provider_map.get(resolved_service) if resolved_service else None

            if provider and self._has_client(provider):
                logger.info(
                    f"[LLMMaskGenerator] Resolved: service={resolved_service}, "
                    f"model={resolved_model}, temp={temp} (prompt='{prompt_id}')"
                )
                configs.append({
                    "provider": provider,
                    "model": resolved_model,
                    "temperature": temp,
                    "system_prompt": system_prompt,
                    "source": f"prompts.yaml:{prompt_id}"
                })
                # Retry с повышенной температурой (тот же сервис!)
                configs.append({
                    "provider": provider,
                    "model": resolved_model,
                    "temperature": round(min(temp + 0.2, 0.5), 2),
                    "system_prompt": system_prompt,
                    "source": f"prompts.yaml:{prompt_id}(retry)"
                })
                logger.info(f"[LLMMaskGenerator] Retry конфигурация: {len(configs)} попыток с {resolved_service}")
                return configs
            else:
                logger.warning(
                    f"[LLMMaskGenerator] Resolved service '{resolved_service}' не имеет клиента, "
                    f"fallback на mask_generation"
                )

        # --- Правило 3: fallback на mask_generation ---
        fallback_cfg = getattr(self.settings, 'mask_generation', None) if self.settings else None

        if fallback_cfg:
            svc = getattr(fallback_cfg, 'default_service', 'mws')
            fallback_model = getattr(fallback_cfg, 'default_model', 'qwen2.5-72b-instruct')
            fallback_temp = getattr(fallback_cfg, 'default_temperature', 0.1)

            provider_map = {
                'openwebui': LLMProvider.OPENWEBUI,
                'mws': LLMProvider.MWS,
                'gigachat': LLMProvider.GIGACHAT,
                'mts_ai': LLMProvider.MTS_AI,
            }
            fallback_provider = provider_map.get(svc, LLMProvider.MWS)
            logger.info(f"[LLMMaskGenerator] Fallback из mask_generation: {svc}/{fallback_model}")
        else:
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
        """Генерация маски с auto-resolution prompt_id."""
        self.attempts = []

        logger.info(
            f"[LLMMaskGenerator] Начало генерации: standard={standard}, "
            f"item_type={item_type}, примеров={len(examples)}, prompt_id={prompt_id}"
        )

        example_samples = [
            ex.get('полное_наименование') or ex.get('наименование', '')
            for ex in examples[:3]
        ]

        standard_type = None
        standard_normalized = None
        if standard_info:
            standard_type = getattr(standard_info, 'standard_type', None)
            if standard_type:
                standard_type = standard_type.value if hasattr(standard_type, 'value') else str(standard_type)
            standard_normalized = getattr(standard_info, 'normalized', None) or getattr(standard_info, 'full_name', None)

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

        prompt = self._build_prompt(standard, item_type, examples, context)
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
        """Одна попытка генерации маски."""
        provider = config["provider"]
        model = config["model"]
        temperature = config["temperature"]
        system_prompt = config.get("system_prompt")
        source = config.get("source", "unknown")

        logger.info(
            f"[LLMMaskGenerator] Попытка {attempt_number}: "
            f"{provider.value}/{model}, temp={temperature} (source: {source})"
        )

        if prompt is None:
            prompt = self._build_prompt(standard, item_type, examples, context)

        if not prompt or not prompt.strip():
            logger.error(f"[LLMMaskGenerator] Попытка {attempt_number}: ПРОМПТ ПУСТОЙ!")
            return GenerationAttempt(attempt_number=attempt_number, provider=provider.value,
                model=model, temperature=temperature, success=False,
                error_message="Prompt is empty", raw_response="PROMPT_WAS_EMPTY")

        raw_response = ""
        response = None
        try:
            client = self._get_client(provider)
            logger.info(f"[LLMMaskGenerator] Попытка {attempt_number}: ВЫЗОВ API {provider.value}")
            response = client.complete(
                prompt=prompt, model=model, temperature=temperature, system_prompt=system_prompt
            )

            if response is None:
                raw_response = "API_RESPONSE_WAS_NONE"
            elif isinstance(response, dict):
                raw_response = response.get("raw") or str(response)
            else:
                raw_response = str(response)

        except Exception as e:
            import traceback
            raw_response = f"EXCEPTION: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"[LLMMaskGenerator] Попытка {attempt_number}: ИСКЛЮЧЕНИЕ: {e}")
        finally:
            safe_response = raw_response if raw_response is not None else "RAW_RESPONSE_WAS_NONE"
            self._save_debug_response(standard, item_type, attempt_number, safe_response)

        if raw_response.startswith("API_RESPONSE_WAS_NONE"):
            return GenerationAttempt(attempt_number=attempt_number, provider=provider.value,
                model=model, temperature=temperature, success=False,
                error_message="API returned None", raw_response=raw_response[:500])

        if raw_response.startswith("EXCEPTION:"):
            return GenerationAttempt(attempt_number=attempt_number, provider=provider.value,
                model=model, temperature=temperature, success=False,
                error_message=raw_response[:200], raw_response=raw_response[:500])

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
    # PROMPT BUILDER
    # ==========================================================================

    def _load_prompt_template(self) -> str:
        """Загрузка шаблона промпта из файла."""
        if self.settings and hasattr(self.settings, 'mask_generation'):
            mg = self.settings.mask_generation
            template_path = getattr(mg, 'prompt_template', None)
            if template_path and Path(template_path).exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()

        for path in [
            'prompts/templates/mask_generation.txt',
            'config/prompts/templates/mask_generation.txt',
            '../prompts/templates/mask_generation.txt',
        ]:
            if Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()

        logger.warning("[LLMMaskGenerator] Шаблон промпта не найден, используем fallback")
        return self._default_prompt_template()

    def _sanitize_filename(self, text: str) -> str:
        """Очистка строки для использования в имени файла."""
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', str(text))
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')[:80]

    def _save_debug_prompt(self, standard: str, item_type: str, prompt: str):
        self._save_debug_file(standard, item_type, "prompt", prompt)

    def _save_debug_response(self, standard: str, item_type: str, attempt: int, response: str):
        self._save_debug_file(standard, item_type, f"response_a{attempt}", response)

    def _save_debug_file(self, standard: str, item_type: str, suffix: str, content: str):
        """Сохранение файла отладки."""
        save_enabled = False
        debug_dir = "prompts/debug"

        if self.settings and hasattr(self.settings, 'mask_generation'):
            mg = self.settings.mask_generation
            save_enabled = getattr(mg, 'save_debug_prompts', False)
            debug_dir = getattr(mg, 'debug_prompts_dir', 'prompts/debug')

        if not save_enabled:
            return

        try:
            from datetime import datetime
            Path(debug_dir).mkdir(parents=True, exist_ok=True)

            safe_type = self._sanitize_filename(item_type or "unknown")
            safe_std = self._sanitize_filename(standard or "unknown")
            if suffix == "prompt":
                filename = f"{safe_type}_{safe_std}.txt"
            else:
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
        return (
            "Ты — эксперт по техническим стандартам ГОСТ и регулярным выражениям Python.\n\n"
            "ЗАДАЧА: Создай regex-паттерн с named groups (?P<name>...) "
            "для извлечения параметров из номенклатуры типа \"{item_type}\" по стандарту {standard}.\n\n"
            "### КРИТИЧЕСКОЕ ПРАВИЛО\n"
            "Создавай named groups ТОЛЬКО для параметров, которые реально видны в исходной строке.\n"
            "НЕ добавляй группы для метаданных ЕСН (тип_резьбы, марка_материала и т.д.), "
            "если их значения не присутствуют в номенклатурной строке.\n"
            "Примеры показывают 'ВИДИМЫЕ В СТРОКЕ' и 'МЕТАДАННЫЕ БД' — используй только ВИДИМЫЕ для regex.\n\n"
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
        """Построение промпта для LLM-генерации regex-маски."""
        template = self._load_prompt_template()

        field_aliases = {'тип_изделия': 'наименование_типа'}

        aliased_examples = []
        for ex in examples:
            new_ex = dict(ex)
            for target, source in field_aliases.items():
                if target not in new_ex or not new_ex.get(target):
                    if source in new_ex and new_ex.get(source):
                        new_ex[target] = new_ex[source]
            aliased_examples.append(new_ex)
        examples = aliased_examples

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

        if 'тип_изделия' not in relevant_fields:
            relevant_fields.insert(0, 'тип_изделия')
        else:
            relevant_fields.remove('тип_изделия')
            relevant_fields.insert(0, 'тип_изделия')

        skip_fields = self._load_skip_fields()

        def clean_name(n: str, max_len: int = 30) -> str:
            result = n.replace('.', '_').replace('-', '_').replace('(', '_').replace(')', '_').replace(',', '_')
            while '__' in result:
                result = result.replace('__', '_')
            result = result.strip('_')
            if len(result) > max_len:
                result = result[:max_len].rstrip('_')
            return result

        field_name_map = {}
        seen_names = set()
        for f in relevant_fields:
            if f not in skip_fields:
                cleaned = clean_name(f)
                original_cleaned = cleaned
                suffix = 2
                while cleaned in seen_names:
                    suffix_str = f"_{suffix}"
                    cleaned = original_cleaned[:max_len - len(suffix_str)] + suffix_str
                    suffix += 1
                seen_names.add(cleaned)
                field_name_map[f] = cleaned

        if 'тип_изделия' not in field_name_map:
            field_name_map['тип_изделия'] = 'тип_изделия'

        visible_fields = set()
        for ex in examples:
            name = ex.get('полное_наименование') or ex.get('наименование', '')
            if not name:
                continue
            name_lower = name.lower()
            for field in list(field_name_map.keys()):
                val = ex.get(field)
                if val is None:
                    continue
                val_str = str(val).strip()
                if not val_str:
                    continue
                val_norm = val_str.lower().replace('.', '').replace(' ', '').replace(',', '')
                name_norm = name_lower.replace('.', '').replace(' ', '').replace(',', '')
                if val_str.lower() in name_lower or val_norm in name_norm:
                    visible_fields.add(field)

        visible_field_names = [f for f in relevant_fields if f in visible_fields and f not in skip_fields]
        invisible_field_names = [f for f in relevant_fields if f not in visible_fields and f not in skip_fields]

        regex_fields = [field_name_map[f] for f in visible_field_names if f in field_name_map]

        if 'тип_изделия' not in visible_field_names and 'тип_изделия' in field_name_map:
            visible_field_names.insert(0, 'тип_изделия')
            if field_name_map['тип_изделия'] not in regex_fields:
                regex_fields.insert(0, field_name_map['тип_изделия'])

        display_fields = [f for f in relevant_fields if f not in skip_fields][:15]

        sample_examples = self._select_diverse_examples(examples, n=10)
        examples_lines = []
        for i, ex in enumerate(sample_examples, 1):
            name = ex.get('полное_наименование') or ex.get('наименование', '')
            if not name:
                continue
            filled_fields = [f"    полное_наименование: {name}"]
            for field in display_fields:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    filled_fields.append(f"    {field}: {val}")

            missing_fields = [f for f in display_fields if not ex.get(f) or not str(ex.get(f)).strip()]
            if missing_fields:
                filled_fields.append(f"    [ПРОПУЩЕНЫ: {', '.join(missing_fields)}]")

            visible_parts = []
            for field in visible_field_names:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    group_name = field_name_map.get(field, field)
                    visible_parts.append(f"(?P<{group_name}>{val})")

            invisible_parts = []
            for field in invisible_field_names:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    invisible_parts.append(f"{field}={val}")

            structure_lines = []
            if visible_parts:
                structure_lines.append("   ВИДИМЫЕ В СТРОКЕ: " + ' '.join(visible_parts))
            if invisible_parts:
                structure_lines.append("   МЕТАДАННЫЕ БД: " + ', '.join(invisible_parts))

            examples_lines.append(
                f"{i}. ИСХОДНАЯ СТРОКА: \"{name}\"\n"
                + "\n".join(structure_lines) + "\n"
                + "   ПОЛЯ ЕСН:\n" + "\n".join(filled_fields)
            )

        examples_text = "\n\n".join(examples_lines) if examples_lines else "Нет примеров"

        stats_lines = []
        for k in visible_field_names:
            if k in field_name_map:
                stats_lines.append(f"    {field_name_map[k]}: {field_stats.get(k, total)} из {total}")
        if invisible_field_names:
            stats_lines.append("    --- МЕТАДАННЫЕ (не для regex): ---")
            for k in invisible_field_names:
                if k in field_name_map:
                    stats_lines.append(f"    {field_name_map[k]}: {field_stats.get(k, total)} из {total} [в БД, не в строке]")
        stats_text = "\n".join(stats_lines) if stats_lines else "Нет статистики"

        import json
        params_list = json.dumps(regex_fields, ensure_ascii=False)

        optional_params = {'исполнение', 'покрытие', 'марка_материала'}
        required_fields = [f for f in regex_fields if f not in optional_params]
        required_list = json.dumps(required_fields, ensure_ascii=False)

        params_hint = ", ".join(regex_fields[:10])
        context_text = context.get('context', '') if context else ''

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

        remaining = [m for m in re.finditer(r'\{[a-z_]+\}', result) if m.group() not in ('{{', '}}')]
        if remaining:
            logger.warning(f"[LLMMaskGenerator] Неподставленные placeholder'ы: {[m.group() for m in remaining[:5]]}")

        return result

    @staticmethod
    def _select_diverse_examples(examples: List[Dict], n: int = 10) -> List[Dict]:
        if len(examples) <= n:
            return examples

        selected = []
        seen_patterns = set()
        selected.append(examples[0])
        seen_patterns.add('full')

        key_fields = ['исполнение', 'покрытие', 'тип_резьбы', 'марка_материала', 'шаг_резьбы', 'номинальный_диаметр_резьбы']

        for field in key_fields:
            for ex in examples[1:]:
                pattern_key = f"missing_{field}"
                if pattern_key in seen_patterns:
                    continue
                if not ex.get(field) or not str(ex.get(field)).strip():
                    has_some = any(ex.get(f) and str(ex.get(f)).strip() for f in key_fields if f != field)
                    if has_some:
                        selected.append(ex)
                        seen_patterns.add(pattern_key)
                        break
            if len(selected) >= n // 2:
                break

        import random
        random.seed(42)
        remaining = [ex for ex in examples if ex not in selected]
        random.shuffle(remaining)

        needed = n - len(selected)
        if remaining and needed > 0:
            selected.extend(remaining[:needed])

        return selected[:n]

    def _preprocess_json_text(self, text: str) -> str:
        r"""Предобработка JSON текста от LLM.
        LLM часто генерирует regex с одиночными backslash (\s, \d, \w) внутри JSON-строк,
        что делает JSON невалидным (JSON допускает только \\, \", \/, \b, \f, \n, \r, \t, \uXXXX).
        Экранируем regex-escapes через negative lookbehind, сохраняя уже двойные (\\) нетронутыми.
        """
        return re.sub(r'(?<!\\)\\([^"\\/bfnrtu])', r'\\\\\1', text)

    def _extract_json(self, text):
        """Извлечение JSON из ответа LLM."""
        if not text or not text.strip():
            return None

        text = self._preprocess_json_text(text)
        candidates = []

        parts = text.split('```')
        for i in range(1, len(parts), 2):
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
        if score >= self.activation_threshold:
            return "activate", f"Score {score:.2f} >= {self.activation_threshold}"
        elif score >= self.retry_threshold:
            return "draft", f"Score {score:.2f} between {self.retry_threshold} and {self.activation_threshold}"
        else:
            return "reject", f"Score {score:.2f} < {self.retry_threshold}"