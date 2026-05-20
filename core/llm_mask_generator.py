"""
LLM Mask Generator
Генерация regex-маски через LLM. Только default_service — никакого fallback на других провайдеров.

LAST_FIXES:
 2026-05-20 2026-05-20 12:07 UTC+3 — _parse_response: добавлен _fix_json_escapes для обработки LLM-ответов
   с невалидными JSON escape-последовательностями (backslash-s, backslash-d, backslash-w и т.д.).
   Ранее json.loads падал с Invalid backslash-escape при наличии backslash-s внутри pattern-строки,
   что приводило к retry и потере tokens на повторные запросы.
 2026-05-20 2026-05-20 12:07 UTC+3 — generate_mask: возвращает metadata (provider, model, temperature,
   tokens_prompt, tokens_completion, warnings).
 2026-05-20 2026-05-20 12:07 UTC+3 — _build_prompt: optional_params переведены на русские имена.
 2026-05-20 2026-05-20 12:07 UTC+3 — _build_prompt: visible_fields detection теперь требует standalone
   токен для чисто числовых значений (r"(?<!\d)8(?!\d)").
 2026-05-20 09:23 UTC+3 — generate_mask: строго только provider_priority (default_service).
 2026-05-18 21:45 UTC+3 — _build_provider_priority: только указанный в конфиге default_service.
"""
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from api_clients.base import BaseLLMClient
from config.settings import get_settings

logger = logging.getLogger(__name__)

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Извлечь JSON-объект из текста через markdown-блок или inline JSON.
    """
    if not text:
        return None

    # Попытка 1: Markdown code block ```json ... ``` / ```python ... ``` / ``` ... ```
    for lang in [r'(?:json)?', r'(?:python)?', r'']:
        pattern = rf'```{lang}\s*(.*?)\s*```'
        md_json = re.search(pattern, text, re.DOTALL)
        if md_json:
            try:
                return json.loads(md_json.group(1))
            except json.JSONDecodeError:
                pass

    # Попытка 3: Найти первый открытый {...} и балансировать скобки
    for start in re.finditer(r'(?m)^\s*\{', text):
        pos = start.start()
        brace_count = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[pos:], start=pos):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                if brace_count == 0:
                    candidate = text[pos:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Попытка 3: Простой fallback
    simple = re.search(r'\{.*?\}', text, re.DOTALL)
    if simple:
        try:
            return json.loads(simple.group())
        except json.JSONDecodeError:
            pass

    return None

def _fix_json_escapes(text: str) -> str:
    """
    Исправить невалидные JSON escape-последовательности, которые LLM генерирует
    внутри regex pattern (например: backslash-s, backslash-d, backslash-w, backslash-dot, backslash-minus).

    В JSON валидные escapes: backslash-quote, backslash-backslash, slash, b, f, n, r, t, uXXXX.
    Все остальное (backslash-s, backslash-d и т.д.) — невалидно и вызывает JSONDecodeError.

    Стратегия: заменяем одиночный backslash перед невалидным символом на двойной.
    """
    valid_escapes = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}

    result = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            # Проверяем, не является ли это уже двойным backslash
            if i > 0 and text[i-1] == '\\':
                result.append(text[i])
                i += 1
                continue
            nxt = text[i + 1]
            if nxt in valid_escapes:
                result.append(text[i])
                result.append(nxt)
                i += 2
                continue
            else:
                # Невалидный escape — удваиваем
                result.append('\\\\')
                result.append(nxt)
                i += 2
                continue
        result.append(text[i])
        i += 1
    return ''.join(result)

class LLMMaskGenerator:
    """Генерация regex-маски через LLM. Строго один провайдер — default_service из конфига."""

    def __init__(self, clients: Dict[str, BaseLLMClient], settings=None, max_retries: int = 3):
        self.clients = clients
        self.settings = settings
        self.max_retries = max_retries
        self.provider_priority = self._build_provider_priority()
        logger.info(
            "LLMMaskGenerator initialized with %d clients, priority: %s",
            len(clients), self.provider_priority
        )

    # --------------------------------------------------------------------------
    # TRANSIENT ERROR DETECTION
    # --------------------------------------------------------------------------

    @staticmethod
    def _is_transient_error(error_msg: str) -> bool:
        """Определить, является ли ошибка временной (retry-worthy)."""
        if not error_msg:
            return False
        lower = error_msg.lower()
        transient_keywords = [
            '503', '502', '504', 'timeout', 'connection',
            'temporary', 'unavailable', 'service unavailable',
            'too many requests', '429', 'rate limit',
            'internal server error', '500', 'read timed out',
            'connecttimeout', 'connection aborted'
        ]
        return any(kw in lower for kw in transient_keywords)

    # --------------------------------------------------------------------------
    # PROVIDER PRIORITY — строго только default_service
    # --------------------------------------------------------------------------

    def _build_provider_priority(self) -> List[str]:
        """Построить приоритет провайдеров: ТОЛЬКО default_service из конфига.
        Никакого fallback на других клиентов."""
        priority = []
        default_service = None

        if self.settings is not None:
            default_service = getattr(self.settings, 'default_service', None)
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
                raise ValueError(
                    f"default_service='{default_service}' not found in available clients: "
                    f"{list(self.clients.keys())}"
                )
        else:
            logger.error("[LLM] No default_service configured in mask_generation or settings!")
            raise ValueError(
                "No default_service configured. Set mask_generation.default_service in config.yaml "
                "or service in prompts.yaml"
            )

        # Fallback на других клиентов УДАЛЁН — только default_service
        logger.info("[LLM] Final provider_priority (strict, single): %s", priority)
        return priority

    # --------------------------------------------------------------------------
    # MASK GENERATION — строго provider_priority, retry с backoff на том же сервисе
    # --------------------------------------------------------------------------

    def generate_mask(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict[str, Any]],
        context: Optional[Dict] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Генерация regex-маски через LLM.

        Returns:
            (mask_dict, meta_dict) где meta_dict содержит:
            - provider, model, temperature
            - tokens_prompt, tokens_completion
            - warnings: список warning-сообщений (например, failed parse на attempt 1)
        """
        prompt = self._build_prompt(standard, item_type, examples, context)
        self._save_debug_prompt(standard, item_type, prompt)

        warnings_list: List[str] = []
        meta: Dict[str, Any] = {
            'provider': None,
            'model': None,
            'temperature': None,
            'tokens_prompt': None,
            'tokens_completion': None,
            'warnings': warnings_list,
        }

        for attempt in range(1, self.max_retries + 1):
            transient_occurred = False

            for provider in self.provider_priority:
                logger.info("Attempt %d/%d via %s", attempt, self.max_retries, provider)
                result = self._call_llm(provider, prompt, attempt)

                if result is None:
                    # Non-transient error (400 Bad Request, auth failure, invalid model и т.д.)
                    # Retry бессмысленен — та же ошибка повторится
                    msg = f"Non-transient failure from {provider} (attempt {attempt})"
                    warnings_list.append(msg)
                    logger.warning(msg + ", aborting retries")
                    return None, meta

                if result.get('_transient_error'):
                    # Transient error — worth retry с backoff
                    msg = (
                        f"Transient error from {provider} (attempt {attempt}): "
                        f"{result.get('error')}"
                    )
                    warnings_list.append(msg)
                    logger.warning(msg)
                    transient_occurred = True
                    continue  # к следующему provider (обычно их 1)

                # Успешный ответ от LLM
                response = result['content']
                self._save_debug_response(
                    standard, item_type, attempt, response,
                    provider=result['provider'],
                    model=result['model'],
                    temperature=result['temperature']
                )
                mask = self._parse_response(response, standard, item_type)
                if mask:
                    meta.update({
                        'provider': result.get('provider'),
                        'model': result.get('model'),
                        'temperature': result.get('temperature'),
                        'tokens_prompt': result.get('tokens_prompt'),
                        'tokens_completion': result.get('tokens_completion'),
                        'warnings': list(warnings_list),
                    })
                    logger.info("Generated mask via %s (attempt %d)", provider, attempt)
                    return mask, meta
                else:
                    msg = f"Failed to parse valid mask from {provider} (attempt {attempt})"
                    warnings_list.append(msg)
                    logger.warning(msg + ", retrying")
                    # Парсинг не удался — не transient, но retry с более высокой temperature
                    continue

            # Если была transient ошибка и ещё есть попытки — backoff
            if transient_occurred and attempt < self.max_retries:
                sleep_time = min(2 ** attempt, 30)
                logger.info(
                    "Transient errors on %s. Sleeping %ds before attempt %d...",
                    self.provider_priority, sleep_time, attempt + 1
                )
                time.sleep(sleep_time)

        msg = f"Failed to generate mask after {self.max_retries} attempts via {self.provider_priority}"
        warnings_list.append(msg)
        logger.error(msg)
        return None, meta

    # --------------------------------------------------------------------------
    # LLM CALL — transient vs non-transient
    # --------------------------------------------------------------------------

    def _call_llm(self, provider: str, prompt: str, attempt: int) -> Optional[Dict[str, Any]]:
        """
        Вызов LLM через конкретного провайдера.

        Returns:
        - dict с content/provider/model/temperature/tokens_prompt/tokens_completion при успехе
        - dict с _transient_error=True при transient ошибке (503, timeout и т.д.)
        - None при non-transient ошибке (400, auth error и т.д.) — retry бессмысленен
        """
        client = self.clients.get(provider)
        if not client:
            return None

        # Определение модели
        model = None
        if self.settings and hasattr(self.settings, 'api') and provider in self.settings.api:
            api_cfg = self.settings.api[provider]
            model = getattr(api_cfg, 'default_model', None)
            logger.debug("[LLM] Using model '%s' from api.%s.default_model", model, provider)

        if not model and self.settings and hasattr(self.settings, 'mask_generation'):
            model = getattr(self.settings.mask_generation, 'default_model', None)
            logger.debug("[LLM] Fallback to mask_generation.default_model: '%s'", model)

        if not model:
            model = "qwen2.5-72b-instruct"
            logger.debug("[LLM] Ultimate fallback model: '%s'", model)

        temperature = min(0.1 + attempt * 0.1, 0.5)

        try:
            response = client.complete(
                prompt=prompt,
                model=model,
                temperature=temperature
            )

            # response — dict с success, content, raw, error, model, tokens_prompt, tokens_completion
            if response and response.get('success'):
                content = response.get('content')
                if isinstance(content, dict):
                    logger.debug(
                        "[LLM] Provider %s returned pre-parsed dict, serializing to JSON",
                        provider
                    )
                    return {
                        'content': json.dumps(content, ensure_ascii=False),
                        'provider': provider,
                        'model': model,
                        'temperature': temperature,
                        'tokens_prompt': response.get('tokens_prompt'),
                        'tokens_completion': response.get('tokens_completion'),
                    }
                raw = response.get('raw', '')
                if raw:
                    return {
                        'content': raw,
                        'provider': provider,
                        'model': model,
                        'temperature': temperature,
                        'tokens_prompt': response.get('tokens_prompt'),
                        'tokens_completion': response.get('tokens_completion'),
                    }
                if content and not isinstance(content, dict):
                    return {
                        'content': str(content),
                        'provider': provider,
                        'model': model,
                        'temperature': temperature,
                        'tokens_prompt': response.get('tokens_prompt'),
                        'tokens_completion': response.get('tokens_completion'),
                    }
                logger.warning(
                    "[LLM] Provider %s returned success=True but no raw/content",
                    provider
                )
                return None
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                if self._is_transient_error(error_msg):
                    logger.warning("[LLM] Transient error from %s: %s", provider, error_msg)
                    return {'_transient_error': True, 'error': error_msg}
                logger.warning("LLM call failed (non-transient): %s", error_msg)
                return None

        except Exception as e:
            error_msg = str(e)
            if self._is_transient_error(error_msg):
                logger.warning("[LLM] Transient exception from %s: %s", provider, error_msg)
                return {'_transient_error': True, 'error': error_msg}
            logger.warning("LLM call failed (non-transient exception): %s", error_msg)
            return None

    # --------------------------------------------------------------------------
    # RESPONSE PARSING
    # --------------------------------------------------------------------------

    def _parse_response(self, response: str, standard: str, item_type: str) -> Optional[Dict[str, Any]]:
        """Извлечь и валидировать JSON из LLM ответа.

        FIX: при JSONDecodeError из-за невалидных escape (backslash-s, backslash-d и т.д. внутри pattern)
        пробуем _fix_json_escapes перед повторным парсингом.
        """
        if not response:
            logger.warning("Empty LLM response")
            return None

        # Попытка 1: Markdown code block ```json ... ```/ ```python ... ``` / ``` ... ```
        for lang in [r'(?:json)?', r'(?:python)?', r'']:
            pattern = rf'```{lang}\s*(.*?)\s*```'
            md_json = re.search(pattern, response, re.DOTALL)
            if md_json:
                candidate = md_json.group(1)
                try:
                    data = json.loads(candidate)
                    return self._validate_mask_dict(data, standard, item_type)
                except json.JSONDecodeError as e:
                    logger.debug("Markdown JSON parse failed: %s", e)
                    # FIX: пробуем исправить escapes
                    try:
                        fixed = _fix_json_escapes(candidate)
                        data = json.loads(fixed)
                        logger.debug("[LLM] JSON parsed after escape fix")
                        return self._validate_mask_dict(data, standard, item_type)
                    except json.JSONDecodeError:
                        pass

        # Попытка 2: Balanced braces
        for start in re.finditer(r'(?m)^\s*\{', response):
            pos = start.start()
            brace_count = 0
            in_string = False
            escape = False
            for i, ch in enumerate(response[pos:], start=pos):
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch == '{':
                        brace_count += 1
                    elif ch == '}':
                        brace_count -= 1
                    if brace_count == 0:
                        candidate = response[pos:i + 1]
                        try:
                            data = json.loads(candidate)
                            return self._validate_mask_dict(data, standard, item_type)
                        except json.JSONDecodeError:
                            try:
                                fixed = _fix_json_escapes(candidate)
                                data = json.loads(fixed)
                                logger.debug("[LLM] JSON (balanced) parsed after escape fix")
                                return self._validate_mask_dict(data, standard, item_type)
                            except json.JSONDecodeError:
                                break

        # Попытка 3: Простой fallback
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            candidate = json_match.group()
            try:
                data = json.loads(candidate)
                return self._validate_mask_dict(data, standard, item_type)
            except json.JSONDecodeError as e:
                try:
                    fixed = _fix_json_escapes(candidate)
                    data = json.loads(fixed)
                    logger.debug("[LLM] JSON (simple) parsed after escape fix")
                    return self._validate_mask_dict(data, standard, item_type)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse LLM response JSON: %s. Preview: %r",
                        e, response[:200]
                    )
                    return None

        logger.warning("No JSON found in LLM response. Preview: %r", response[:200])
        return None

    def _validate_mask_dict(
        self,
        data: Dict[str, Any],
        standard: str,
        item_type: str
    ) -> Optional[Dict[str, Any]]:
        """Валидировать и дополнить dict маски."""
        pattern = data.get('pattern', '')
        params = data.get('params', [])
        required = data.get('required', [])

        if not pattern:
            logger.warning("Mask dict missing 'pattern' field")
            return None

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

    # --------------------------------------------------------------------------
    # PROMPT BUILDER
    # --------------------------------------------------------------------------

    def _load_prompt_template(self) -> str:
        """Загрузить шаблон промпта из файла."""
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

    def _default_prompt_template(self) -> str:
        return (
            "Ты — эксперт по регулярным выражениям. Создай Python-совместимую regex-маску "
            "для извлечения параметров из наименований изделий.\n\n"
            "### КРИТИЧЕСКИЕ ПРАВИЛА\n"
            "1. ПОКРЫТИЕ/ТЕХНИЧЕСКИЕ_ХАРАКТЕРИСТИКИ: ВСЕГДА используй [\\w.]+. "
            "НИКОГДА не хардкоди конкретные значения покрытий (Хим.Пас, Ср, Ц, Кд, Окс, Фос, ЭФП). "
            "Эти значения меняются от изделия к изделию — regex должен матчить ЛЮБОЕ покрытие.\n"
            "2. ТОЧКА КАК РАЗДЕЛИТЕЛЬ: если в строке встречается число.число, "
            "и после точки ровно 2 цифры (например 100.58, 12.5), "
            "это НЕ десятичная дробь, а ДВА отдельных параметра: длина=100, группа_прочности=58. "
            "Используй (?P<длина>\\d+)\\.(?P<группа_прочности>\\d+), а НЕ (?P<длина>\\d+(?:[.,]\\d+)?).\n"
            "3. Используй named groups (?P<name>...) для захвата значений.\n\n"
            "### ВХОДНЫЕ ДАННЫЕ\n"
            "Тип изделия: {item_type}\n"
            "Стандарт: {standard}\n"
            "Примеры наименований (с параметрами):\n"
            "{examples_text}\n\n"
            "### ТРЕБУЕМЫЙ ВЫВОД\n"
            "```json\n"
            "{{\n"
            "  \"pattern\": \"...\",\n"
            "  \"params\": {params_list},\n"
            "  \"required\": {required_list}\n"
            "}}\n"
            "```\n"
        )

    def _load_skip_fields(self) -> set:
        """Загрузить skip_fields из ens_column_mapping.yaml."""
        default = {
            'id', 'mdm_key', 'created_at', 'updated_at', 'hash',
            'pattern_hash', 'source'
        }
        try:
            import yaml
            for path in ['config/ens_column_mapping.yaml', 'ens_column_mapping.yaml']:
                if Path(path).exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    fields = data.get('skip_fields', [])
                    if fields:
                        return set(fields)
                    break
        except Exception as e:
            logger.warning(f"[LLMMaskGenerator] Failed to load skip_fields: {e}")
        return default

    @staticmethod
    def _select_diverse_examples(examples: List[Dict], n: int = 10) -> List[Dict]:
        """Отобрать разнообразные примеры (с пропущенными и заполненными полями)."""
        if len(examples) <= n:
            return examples

        selected = []
        seen_patterns = set()
        selected.append(examples[0])
        seen_patterns.add('full')

        key_fields = [
            'diameter', 'length', 'width', 'height', 'thread',
            'material', 'coating', 'standard', 'item_type'
        ]

        for field in key_fields:
            for ex in examples[1:]:
                pattern_key = f"missing_{field}"
                if pattern_key in seen_patterns:
                    continue
                if not ex.get(field) or not str(ex.get(field)).strip():
                    has_some = any(
                        ex.get(f) and str(ex.get(f)).strip()
                        for f in key_fields if f != field
                    )
                    if has_some:
                        selected.append(ex)
                        seen_patterns.add(pattern_key)
                        break
                if len(selected) >= n // 2:
                    break

        random.seed(42)
        remaining = [ex for ex in examples if ex not in selected]
        random.shuffle(remaining)
        needed = n - len(selected)
        if remaining and needed > 0:
            selected.extend(remaining[:needed])

        return selected[:n]

    def _build_prompt(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict[str, Any]],
        context: Optional[Dict] = None
    ) -> str:
        """Построить промпт с field_stats, visible/invisible fields, примерами."""
        template = self._load_prompt_template()

        # --- Подготовка примеров ---
        sample_examples = self._select_diverse_examples(examples, n=10)

        # --- Получаем field_stats ---
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

        skip_fields = self._load_skip_fields()

        # --- Очистка имен для named groups ---
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

        if 'standard' not in field_name_map:
            field_name_map['standard'] = 'standard'
        if 'item_type' not in field_name_map:
            field_name_map['item_type'] = 'item_type'

        # --- visible vs invisible fields ---
        visible_fields = set()
        for ex in examples:
            name = ex.get('name') or ex.get('наименование', '')
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
                val_lower = val_str.lower()

                # FIX: для чисто числовых значений требуем standalone токен
                # (граница по не-цифре), чтобы избежать ложного совпадения
                # "8" внутри "80" или "31133-80"
                if val_lower.isdigit():
                    pattern = r'(?<!\d)' + re.escape(val_lower) + r'(?!\d)'
                    if re.search(pattern, name_lower):
                        visible_fields.add(field)
                    continue

                val_norm = val_lower.replace('.', '').replace(' ', '').replace(',', '')
                name_norm = name_lower.replace('.', '').replace(' ', '').replace(',', '')
                if val_lower in name_lower or val_norm in name_norm:
                    visible_fields.add(field)

        visible_field_names = [f for f in relevant_fields if f in visible_fields and f not in skip_fields]
        invisible_field_names = [f for f in relevant_fields if f not in visible_fields and f not in skip_fields]

        regex_fields = [field_name_map[f] for f in visible_field_names if f in field_name_map]

        # standard должен быть первым
        if 'standard' not in visible_field_names and 'standard' in field_name_map:
            visible_field_names.insert(0, 'standard')
            if field_name_map['standard'] not in regex_fields:
                regex_fields.insert(0, field_name_map['standard'])

        display_fields = [f for f in relevant_fields if f not in skip_fields][:15]

        # --- Формирование примеров ---
        examples_lines = []
        for i, ex in enumerate(sample_examples, 1):
            name = ex.get('name') or ex.get('наименование', '')
            if not name:
                continue

            filled_fields = [f" Наименование: {name}"]
            for field in display_fields:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    filled_fields.append(f" {field}: {val}")

            missing_fields = [f for f in display_fields if not ex.get(f) or not str(ex.get(f)).strip()]
            if missing_fields:
                filled_fields.append(f" [Отсутствуют: {', '.join(missing_fields)}]")

            visible_parts = []
            for field in visible_field_names:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    group_name = field_name_map.get(field, field)
                    visible_parts.append(f"(?P<{group_name}>{re.escape(str(val))})")

            invisible_parts = []
            for field in invisible_field_names:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    invisible_parts.append(f"{field}={val}")

            structure_lines = []
            if visible_parts:
                structure_lines.append(" Видимые: " + ' '.join(visible_parts))
            if invisible_parts:
                structure_lines.append(" Скрытые: " + ', '.join(invisible_parts))

            examples_lines.append(
                f"{i}. Исходное: \"{name}\"\n" +
                "\n".join(structure_lines) + "\n" +
                " Параметры:\n" +
                "\n".join(filled_fields)
            )

        examples_text = "\n\n".join(examples_lines) if examples_lines else "Нет примеров"

        # --- Статистика ---
        stats_lines = []
        for k in visible_field_names:
            if k in field_name_map:
                stats_lines.append(f" {field_name_map[k]}: {field_stats.get(k, total)} из {total}")
        if invisible_field_names:
            stats_lines.append(" --- Скрытые (не в regex): ---")
            for k in invisible_field_names:
                if k in field_name_map:
                    stats_lines.append(
                        f" {field_name_map[k]}: {field_stats.get(k, total)} из {total} [в статистике, не в шаблоне]"
                    )
        stats_text = "\n".join(stats_lines) if stats_lines else "Нет статистики"

        # --- JSON-список параметров ---
        params_list = json.dumps(regex_fields, ensure_ascii=False)
        # FIX: русские имена опциональных параметров (были английские — не совпадали с полями ENS)
        optional_params = {
            'покрытие', 'марка_материала', 'исполнение',
            'шаг_резьбы', 'технические_характеристики'
        }
        required_fields = [f for f in regex_fields if f not in optional_params]
        required_list = json.dumps(required_fields, ensure_ascii=False)
        params_hint = ", ".join(regex_fields[:10])
        context_text = context.get('context', '') if context else ''

        # --- Сборка промпта ---
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
            logger.warning(
                "[LLMMaskGenerator] Незаменённые placeholder'ы: %s",
                [m.group() for m in remaining[:5]]
            )

        return result

    # --------------------------------------------------------------------------
    # DEBUG SAVE
    # --------------------------------------------------------------------------

    def _sanitize_filename(self, text: str) -> str:
        sanitized = re.sub(r'[\\/:\\*?"<>|]', '_', str(text))
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')[:80]

    def _save_debug_prompt(self, standard: str, item_type: str, prompt: str):
        self._save_debug_file(standard, item_type, "prompt", prompt,
                              provider="PENDING", model="PENDING", temperature="PENDING")

    def _save_debug_response(self, standard: str, item_type: str, attempt: int, response: str,
                             provider: str = "N/A", model: str = "N/A", temperature: float = 0.0):
        self._save_debug_file(standard, item_type, f"response_a{attempt}", response,
                              provider=provider, model=model, temperature=temperature)

    def _save_debug_file(self, standard: str, item_type: str, suffix: str, content: str,
                         provider: str = "N/A", model: str = "N/A", temperature: Any = "N/A"):
        """Сохранить debug-файл (prompt/response) в prompts/debug."""
        save_enabled = False
        debug_dir = "prompts/debug"

        if self.settings and hasattr(self.settings, 'mask_generation'):
            mg = self.settings.mask_generation
            save_enabled = getattr(mg, 'save_debug_prompts', False)
            debug_dir = getattr(mg, 'debug_prompts_dir', 'prompts/debug')

        if not save_enabled:
            return

        try:
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
                f"# Тип изделия: {item_type}\n"
                f"# Стандарт: {standard}\n"
                f"# Провайдер: {provider}\n"
                f"# Модель: {model}\n"
                f"# Температура: {temperature}\n"
                f"# Время: {datetime.now().isoformat()}\n"
                f"# {'=' * 50}\n"
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write(content)

            logger.info("[LLMMaskGenerator] Сохранён debug: %s", filepath)
        except Exception as e:
            logger.warning("[LLMMaskGenerator] Ошибка сохранения debug: %s", e)