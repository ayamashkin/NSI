"""
LLM Mask Generator Module
Level 2: Автоматическая генерация regex масок с помощью LLM.
Модель/температура/system_prompt определяются автоматически по keywords из prompts.yaml.
"""

import re
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
        context: Optional[Dict]
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

        client = self._get_client(provider)

        # === ПРОМПТ ДЛЯ ГЕНЕРАЦИИ МАСКИ ===
        prompt = self._build_prompt(standard, item_type, examples, context)

        # Логируем полный промпт на DEBUG уровне
        logger.debug(f"[LLMMaskGenerator] === PROMPT (attempt {attempt_number}) ===\n{prompt}\n=== END PROMPT ===")

        try:
            response = client.complete(
                prompt=prompt,
                model=model,
                temperature=temperature,
                system_prompt=system_prompt
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

    # ==========================================================================
    # PROMPT BUILDER (промпт для генерации маски)
    # ==========================================================================

    def _build_prompt(self, standard, item_type, examples, context):
        """
        Построение промпта для LLM-генерации regex-маски.

        Передает ВСЕ поля из ЕСН для каждого примера, включая разбор
        где какой параметр находится в строке номенклатуры.
        """
        # Берем до 10 примеров для показа в промпте
        sample_examples = examples[:10]

        # Собираем статистику по ВСЕМ полям из examples (кроме служебных _*)
        field_stats = {}
        for ex in examples:
            for key, val in ex.items():
                # Пропускаем служебные поля (_original_, _ens_, _detected_, etc.)
                if key.startswith('_'):
                    continue
                # Пропускаем пустые значения
                if val is None or (isinstance(val, str) and not val.strip()):
                    continue
                field_stats[key] = field_stats.get(key, 0) + 1

        # Сортируем по заполненности, берем топ полей (>10% заполненности)
        total = len(examples)
        threshold = max(1, int(total * 0.1))  # минимум 10% или 1
        sorted_fields = sorted(field_stats.items(), key=lambda x: -x[1])
        relevant_fields = [k for k, v in sorted_fields if v >= threshold]

        # Поля для показа в разделе ПОЛЯ ЕСН (релевантные + ключевые)
        display_fields = relevant_fields[:15]  # максимум 15 полей

        examples_lines = []
        for i, ex in enumerate(sample_examples, 1):
            name = ex.get('полное_наименование') or ex.get('наименование', '')
            if not name:
                continue

            # Собираем все заполненные поля из ЕСН (только релевантные)
            filled_fields = []
            for field in display_fields:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    filled_fields.append(f"    {field}: {val}")

            # Показываем где параметры в строке (если значение входит в название)
            in_text = []
            for field in display_fields:
                val = ex.get(field)
                if val and str(val) in name:
                    in_text.append(f"      {field}='{val}' найдено в позиции {name.find(str(val))}")

            in_text_str = "\n".join(in_text) if in_text else "      (разбор не удался)"

            examples_lines.append(
                f"{i}. НОМЕНКЛАТУРА: \"{name}\"\n"
                f"   ПОЛЯ ЕСН:\n" + "\n".join(filled_fields) + "\n"
                f"   РАЗБОР ПО ПОЗИЦИЯМ:\n{in_text_str}"
            )

        examples_text = "\n\n".join(examples_lines)

        # Статистика уже собрана выше — используем relevant_fields
        stats_lines = [f"    {k}: {field_stats[k]} из {total}" for k in relevant_fields]

        # Имена параметров для JSON-примера (только реальные поля из ЕСН)
        params_list = json.dumps(relevant_fields, ensure_ascii=False)
        # Required — поля с 100% заполненностью
        required_fields = [k for k, v in sorted_fields if v == total and not k.startswith(('полное_наименование', 'наименование', 'код', 'mdm_key'))]
        if not required_fields:
            required_fields = [k for k, v in sorted_fields if v >= total * 0.8 and not k.startswith(('полное_наименование', 'наименование', 'код', 'mdm_key'))][:5]
        required_list = json.dumps(required_fields, ensure_ascii=False)

        context_text = ""
        if context:
            context_text = f"\nДоп. контекст: {json.dumps(context, ensure_ascii=False)}\n"

        return f"""Ты — эксперт по техническим стандартам ГОСТ/ОСТ/ТУ и регулярным выражениям Python.

ЗАДАЧА: Создай regex-паттерн с named groups (?P<name>...) для извлечения параметров из номенклатуры типа "{item_type}" по стандарту {standard}.

ВАЖНО: Ниже приведены реальные данные из Единого Справочника Номенклатуры (ЕСН).
Для КАЖДОЙ позиции показаны:
- Полное наименование (как в базе)
- ВСЕ заполненные поля ЕСН (материал, покрытие, диаметр, длина и т.д.)
- Разбор — где в строке находится каждый параметр

ИСПОЛЬЗУЙ эти данные чтобы понять СТРУКТУРУ строки и какие параметры где находятся.

=== ПРИМЕРЫ ИЗ ЕСН ({len(sample_examples)} шт.) ===

{examples_text}

=== СТАТИСТИКА ЗАПОЛНЕННОСТИ ПОЛЕЙ (по {total} примерам) ===
{chr(10).join(stats_lines)}

ВАЖНО: В ответе перечисли ТОЛЬКО эти параметры: {', '.join(relevant_fields)}

{context_text}
=== ТРЕБОВАНИЯ К РЕГЕКСУ ===

1. Используй named groups (?P<param_name>...) — имя группы = имя поля из списка выше
2. Regex ДОЛЖЕН матчить ПОЛНУЮ строку от начала до конца (используй ^ и $)
3. Учитывай специфику стандарта {standard}:
   - ГОСТ: обычно М<диаметр>х<длина>.<класс>.<покрытие>
   - Исполнение может быть числом перед М (например "2М12" = исп.2, диам.М12)
   - Поле допуска резьбы: 6g, 8g и т.д.
   - Материал: 40Х, А2, А4 и т.д.
   - Покрытие: 3-значный код (019, 016 и т.д.)
4. Поддерживай опциональные части через ?
5. Для разделителей используй [\.\s]* или [\s\-]*
6. НЕ используй кириллицу в именах групп — только ASCII
7. Все числа ДОЛЖНЫ включать возможность десятичной точки/запятой

=== ОТВЕТ ===

Ответь ТОЛЬКО в формате JSON (без markdown-форматирования):
{{
  "pattern": "ваш regex с (?P<name>) группами",
  "params": {params_list},
  "required": {required_list}
}}"""

    def _extract_json(self, text):
        """Извлечение JSON из текста."""
        for pattern in [r"```json\s*(.*?)```", r"```\s*(.*?)```", r"\{{[\s\S]*\}}"]:
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