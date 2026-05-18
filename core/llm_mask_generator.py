"""
LLM Mask Generator
Генерация regex-масок для стандартов через LLM с fallback по провайдерам.

LAST_FIXES:
  2026-05-18 14:12 UTC+3 — _save_debug_file: заголовок с сервисом, моделью, температурой
  2026-05-18 13:10 UTC+3 — ВОССТАНОВЛЕН _build_prompt: field_stats, visible/invisible fields,
                           пропущенные поля, разнообразные примеры (как до 15.05)
  2026-05-18 12:45 UTC+3 — _build_prompt: наименования + значимые заполненные поля
  2026-05-18 11:50 UTC+3 — Восстановлены _save_debug_prompt/_save_debug_response
  2026-05-18 11:16 UTC+3 — _parse_response: поддержка ```python + balanced braces JSON extraction
  2026-05-18 10:35 UTC+3 — _call_llm: работа с Dict-ответами (success/content/raw/error/model)
"""

import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from api_clients.base import BaseLLMClient
from config.settings import get_settings

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Извлекает JSON-объект из текста с markdown-блоками или inline JSON.
    """
    if not text:
        return None

    # Стратегия 1: Markdown code block ```json ... ``` / ```python ... ``` / ``` ... ```
    for lang in [r'(?:json)?', r'(?:python)?', r'']:
        pattern = rf'```{lang}\s*(.*?)\s*```'
        md_json = re.search(pattern, text, re.DOTALL)
        if md_json:
            try:
                return json.loads(md_json.group(1))
            except json.JSONDecodeError:
                pass

    # Стратегия 2: Найти первый {...} с балансом скобок
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

    # Стратегия 3: Простой fallback
    simple = re.search(r'\{.*?\}', text, re.DOTALL)
    if simple:
        try:
            return json.loads(simple.group())
        except json.JSONDecodeError:
            pass

    return None


class LLMMaskGenerator:
    """Генератор масок через LLM с fallback по провайдерам."""

    def __init__(self, clients: Dict[str, BaseLLMClient], settings=None, max_retries: int = 3):
        self.clients = clients
        self.settings = settings
        self.max_retries = max_retries
        self.provider_priority = self._build_provider_priority()
        logger.info(
            "LLMMaskGenerator initialized with %d clients, priority: %s",
            len(clients), self.provider_priority
        )

    # ------------------------------------------------------------------
    # PROVIDER PRIORITY
    # ------------------------------------------------------------------

    def _build_provider_priority(self) -> List[str]:
        """Строит приоритет провайдеров: default_service первым, затем остальные."""
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

        for provider in self.clients.keys():
            if provider not in priority:
                priority.append(provider)

        logger.info("[LLM] Final provider_priority: %s", priority)
        return priority

    # ------------------------------------------------------------------
    # MASK GENERATION
    # ------------------------------------------------------------------

    def generate_mask(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict[str, Any]],
        context: Optional[Dict] = None
    ) -> tuple[Optional[Dict[str, Any]], None]:
        """Генерация маски через LLM с fallback по провайдерам."""
        prompt = self._build_prompt(standard, item_type, examples, context)
        self._save_debug_prompt(standard, item_type, prompt)

        for attempt in range(1, self.max_retries + 1):
            for provider in self.provider_priority:
                logger.info("Attempt %d/%d via %s", attempt, self.max_retries, provider)
                response_data = self._call_llm(provider, prompt, attempt)
                if response_data:
                    response = response_data['content']
                    self._save_debug_response(
                        standard, item_type, attempt, response,
                        provider=response_data['provider'],
                        model=response_data['model'],
                        temperature=response_data['temperature']
                    )
                    mask = self._parse_response(response, standard, item_type)
                    if mask:
                        logger.info("Generated mask via %s (attempt %d)", provider, attempt)
                        return mask, None
                    else:
                        logger.warning(
                            "Failed to parse response from %s (attempt %d)",
                            provider, attempt
                        )
                else:
                    logger.warning("No response from %s (attempt %d)", provider, attempt)

        logger.error("Failed to generate mask after %d attempts", self.max_retries)
        return None, None

    # ------------------------------------------------------------------
    # LLM CALL — возвращает Dict с metadata
    # ------------------------------------------------------------------

    def _call_llm(self, provider: str, prompt: str, attempt: int) -> Optional[Dict[str, Any]]:
        """
        Вызов LLM для генерации маски.
        Возвращает dict: {content: str, provider: str, model: str, temperature: float}
        или None при ошибке.
        """
        client = self.clients.get(provider)
        if not client:
            return None

        # Разрешаем модель
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

            # response — dict с success, content, raw, error, model
            if response and response.get('success'):
                # Если content уже распарсен (dict), сериализуем в JSON-строку
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
                        'temperature': temperature
                    }
                # Иначе вернуть raw текст
                raw = response.get('raw', '')
                if raw:
                    return {
                        'content': raw,
                        'provider': provider,
                        'model': model,
                        'temperature': temperature
                    }
                # Если raw пустой, но content есть (не dict) — вернуть content
                if content and not isinstance(content, dict):
                    return {
                        'content': str(content),
                        'provider': provider,
                        'model': model,
                        'temperature': temperature
                    }
                logger.warning(
                    "[LLM] Provider %s returned success=True but no raw/content",
                    provider
                )
                return None
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                logger.warning("LLM call failed: %s", error_msg)
                return None

        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # RESPONSE PARSING
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, standard: str, item_type: str) -> Optional[Dict[str, Any]]:
        """Парсинг ответа LLM с robust JSON extraction."""
        if not response:
            logger.warning("Empty LLM response")
            return None

        # Стратегия 1: Markdown code block ```json ... ``` / ```python ... ``` / ``` ... ```
        for lang in [r'(?:json)?', r'(?:python)?', r'']:
            pattern = rf'```{lang}\s*(.*?)\s*```'
            md_json = re.search(pattern, response, re.DOTALL)
            if md_json:
                try:
                    data = json.loads(md_json.group(1))
                    return self._validate_mask_dict(data, standard, item_type)
                except json.JSONDecodeError as e:
                    logger.debug("Markdown JSON parse failed: %s", e)

        # Стратегия 2: Balanced braces
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
                                break

        # Стратегия 3: Простой fallback
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_mask_dict(data, standard, item_type)
            except json.JSONDecodeError as e:
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
        """Валидация и нормализация словаря маски."""
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

    # ------------------------------------------------------------------
    # PROMPT BUILDER (ВОССТАНОВЛЕНО из 54019ee8)
    # ------------------------------------------------------------------

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

    def _default_prompt_template(self) -> str:
        return (
            "Ты — эксперт по техническим стандартам ГОСТ и регулярным выражениям Python.\n\n"
            "ЗАДАЧА: Создай regex-паттерн с named groups (?P...) "
            "для извлечения параметров из номенклатуры типа "{item_type}" по стандарту {standard}.\n\n"
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

    def _load_skip_fields(self) -> set:
        """Загрузка skip_fields из ens_column_mapping.yaml."""
        default = {
            'код', 'mdm_key', 'единицы_измерения', 'наименование_типа.1',
            'полное_наименование', 'наименование', 'нтд',
            'наименование_типа'
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
            logger.warning(f"[LLMMaskGenerator] Не удалось загрузить skip_fields: {e}")
        return default

    @staticmethod
    def _select_diverse_examples(examples: List[Dict], n: int = 10) -> List[Dict]:
        """Выбор разнообразных примеров (включая с пропущенными ключевыми полями)."""
        if len(examples) <= n:
            return examples

        selected = []
        seen_patterns = set()
        selected.append(examples[0])
        seen_patterns.add('full')

        key_fields = [
            'исполнение', 'покрытие', 'тип_резьбы',
            'марка_материала', 'шаг_резьбы', 'номинальный_диаметр_резьбы'
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
        """Построение промпта с field_stats, visible/invisible fields, пропущенные поля."""
        template = self._load_prompt_template()

        # --- Алиасы полей ---
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

        # --- Статистика по полям ---
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

        # --- Чистка имён полей для named groups ---
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

        # --- Видимые vs Невидимые поля ---
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

        # Тип изделия всегда первый
        if 'тип_изделия' not in visible_field_names and 'тип_изделия' in field_name_map:
            visible_field_names.insert(0, 'тип_изделия')
            if field_name_map['тип_изделия'] not in regex_fields:
                regex_fields.insert(0, field_name_map['тип_изделия'])

        display_fields = [f for f in relevant_fields if f not in skip_fields][:15]

        # --- Формирование примеров ---
        sample_examples = self._select_diverse_examples(examples, n=10)

        examples_lines = []
        for i, ex in enumerate(sample_examples, 1):
            name = ex.get('полное_наименование') or ex.get('наименование', '')
            if not name:
                continue

            filled_fields = [f"  полное_наименование: {name}"]
            for field in display_fields:
                val = ex.get(field)
                if val is not None and str(val).strip():
                    filled_fields.append(f"  {field}: {val}")

            missing_fields = [f for f in display_fields if not ex.get(f) or not str(ex.get(f)).strip()]
            if missing_fields:
                filled_fields.append(f"  [ПРОПУЩЕНЫ: {', '.join(missing_fields)}]")

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
                structure_lines.append("  ВИДИМЫЕ В СТРОКЕ: " + ' '.join(visible_parts))
            if invisible_parts:
                structure_lines.append("  МЕТАДАННЫЕ БД: " + ', '.join(invisible_parts))

            examples_lines.append(
                f"{i}. ИСХОДНАЯ СТРОКА: \"{name}\"\n" +
                "\n".join(structure_lines) + "\n" +
                "  ПОЛЯ ЕСН:\n" +
                "\n".join(filled_fields)
            )

        examples_text = "\n\n".join(examples_lines) if examples_lines else "Нет примеров"

        # --- Статистика ---
        stats_lines = []
        for k in visible_field_names:
            if k in field_name_map:
                stats_lines.append(f"  {field_name_map[k]}: {field_stats.get(k, total)} из {total}")
        if invisible_field_names:
            stats_lines.append("  --- МЕТАДАННЫЕ (не для regex): ---")
            for k in invisible_field_names:
                if k in field_name_map:
                    stats_lines.append(
                        f"  {field_name_map[k]}: {field_stats.get(k, total)} из {total} [в БД, не в строке]"
                    )
        stats_text = "\n".join(stats_lines) if stats_lines else "Нет статистики"

        # --- JSON-списки для шаблона ---
        params_list = json.dumps(regex_fields, ensure_ascii=False)
        optional_params = {'исполнение', 'покрытие', 'марка_материала'}
        required_fields = [f for f in regex_fields if f not in optional_params]
        required_list = json.dumps(required_fields, ensure_ascii=False)
        params_hint = ", ".join(regex_fields[:10])
        context_text = context.get('context', '') if context else ''

        # --- Подстановка в шаблон ---
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
                "[LLMMaskGenerator] Неподставленные placeholder'ы: %s",
                [m.group() for m in remaining[:5]]
            )

        return result

    # ------------------------------------------------------------------
    # DEBUG SAVE — с сервисом, моделью, температурой
    # ------------------------------------------------------------------

    def _sanitize_filename(self, text: str) -> str:
        sanitized = re.sub(r'[\\/:*?"<>|]', '_', str(text))
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
        """Сохранение debug-файла (prompt/response) с метаданными вызова."""
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
                f"# Тип: {item_type}\n"
                f"# Стандарт: {standard}\n"
                f"# Сервис: {provider}\n"
                f"# Модель: {model}\n"
                f"# Температура: {temperature}\n"
                f"# Дата: {datetime.now().isoformat()}\n"
                f"# {'=' * 50}\n\n"
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write(content)

            logger.info("[LLMMaskGenerator] Сохранён debug: %s", filepath)
        except Exception as e:
            logger.warning("[LLMMaskGenerator] Ошибка сохранения debug: %s", e)