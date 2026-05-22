# =============================================================================
# FILE: generators/llm_mask_generator.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 COMMITS (UTC+3):
# 2026-05-21 08:23:07 51f335da 21.05.2026
# 2026-05-21 08:05:56 ee843b22 21.05.2026
# 2026-05-20 17:47:49 19e8ca02 20.05.2026
# 2026-05-20 17:39:23 b00c4b25 20.05.2026
# 2026-05-20 17:31:34 66c66c93 20.05.2026
# =============================================================================
# FIX 2026-05-22 14:04 UTC+3:
# 1. RESTORED ENS examples injection into prompt.
# 2. FIXED return signature: generate_mask() now returns
#    (MaskGenerationResult, metadata_dict) instead of (result, int).
#    metadata_dict contains provider, model, temperature, tokens.
#    This matches cli.py expectations (meta.get("provider")).
# =============================================================================
"""
LLM Mask Generator Module
Generates regex masks using LLM with ENS examples context.

LAST_FIX: 2026-05-22 14:04 UTC+3 — ENS examples restored + return signature fixed.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.standard_utils import canonicalize_standard

logger = logging.getLogger(__name__)


@dataclass
class MaskGenerationResult:
    """Результат генерации маски."""
    pattern: str = ""
    params: List[str] = field(default_factory=list)
    required: List[str] = field(default_factory=list)
    standard: str = ""
    item_type: str = ""
    raw_response: str = ""
    service: str = ""
    model: str = ""
    temperature: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0


class LLMMaskGenerator:
    """Генератор масок через LLM с ENS-примерами."""

    def __init__(
        self,
        clients: Dict[str, Any],
        settings: Any = None,
        max_retries: int = 3,
    ):
        self.clients = clients
        self.settings = settings
        self.max_retries = max_retries
        self.validator = None
        logger.info("[LLMMaskGenerator] Initialized with %d clients", len(clients))

    def _get_validator(self):
        """Lazy init validator for ENS examples."""
        if self.validator is None:
            try:
                from core.auto_validator import AutoValidator
                ens_path = None
                if self.settings and hasattr(self.settings, "database"):
                    ens_path = getattr(self.settings.database, "ens_index_path", None)
                if not ens_path:
                    ens_path = "cache/ens_hardware.pkl"
                self.validator = AutoValidator(
                    ens_index_path=ens_path,
                    activation_threshold=0.85
                )
                logger.info("[LLMMaskGenerator] Validator initialized with %s", ens_path)
            except Exception as e:
                logger.warning("[LLMMaskGenerator] Failed to init validator: %s", e)
        return self.validator

    def _get_ens_examples(self, standard: str, item_type: str, max_examples: int = 10) -> List[Dict]:
        """Получить примеры из ЕНС для подстановки в промпт."""
        validator = self._get_validator()
        if not validator:
            logger.warning("[LLMMaskGenerator] No validator, returning empty examples")
            return []
        try:
            examples = validator._get_ens_examples(standard, item_type)
            if examples:
                logger.info("[LLMMaskGenerator] Loaded %d ENS examples for %s/%s",
                           len(examples), standard, item_type)
                return examples[:max_examples]
        except Exception as e:
            logger.warning("[LLMMaskGenerator] Failed to load examples: %s", e)
        return []

    def _format_examples(self, examples: List[Dict], standard: str, item_type: str) -> str:
        """Форматировать ENS-примеры для вставки в промпт."""
        if not examples:
            return ""
        lines = ["=== ПРИМЕРЫ ИЗ ЕНС (только ВИДИМЫЕ параметры) ===", ""]
        for i, ex in enumerate(examples[:10], 1):
            name = ex.get("наименование", ex.get("полное_наименование", ""))
            if not name:
                continue
            visible = []
            hidden = []
            for key in ["тип_изделия", "наименование_типа", "исполнение",
                       "номинальный_диаметр_резьбы", "длина", "шаг_резьбы",
                       "покрытие", "толщина_проката_стенки_полки",
                       "наружный_диаметр_диаметр_вписанного_круга_сторона_квадрата_стороны_поперечного_сечения",
                       "нтд_1", "нтд_2"]:
                val = ex.get(key)
                if val and str(val).strip():
                    val_str = str(val).strip()
                    if val_str in name or val_str.lower() in name.lower():
                        visible.append((key, val_str))
                    else:
                        hidden.append((key, val_str))
            lines.append(f'{i}. Исходное: "{name}"')
            if visible:
                vis_str = " ".join([f"(?P<{k}>{v})" for k, v in visible])
                lines.append(f"   Видимые: {vis_str}")
            if hidden:
                hid_str = ", ".join([f"{k}={v}" for k, v in hidden])
                lines.append(f"   Скрытые: {hid_str}")
            lines.append("")
        lines.append("=== СТАТИСТИКА ПО ПАРАМЕТРАМ ===")
        param_counts = {}
        for ex in examples:
            for key in ["исполнение", "номинальный_диаметр_резьбы", "длина",
                       "шаг_резьбы", "покрытие", "толщина_проката_стенки_полки"]:
                if ex.get(key) and str(ex.get(key)).strip():
                    param_counts[key] = param_counts.get(key, 0) + 1
        total = len(examples)
        for key, count in sorted(param_counts.items(), key=lambda x: -x[1]):
            lines.append(f" {key}: {count} из {total} ({count/total*100:.0f}%)")
        lines.append("")
        return "\n".join(lines)

    def _get_prompt_template(self) -> str:
        """Загрузить шаблон промпта."""
        if self.settings and hasattr(self.settings, "mask_generation"):
            mg = self.settings.mask_generation
            if hasattr(mg, "prompt_template") and mg.prompt_template:
                return mg.prompt_template
        for path in [
            "prompts/templates/mask_generation.txt",
            "prompts/mask_generation.txt",
            "config/mask_generation.txt",
        ]:
            p = Path(path)
            if p.exists():
                return p.read_text(encoding="utf-8")
        return self._default_template()

    def _default_template(self) -> str:
        """Default template with v3 rules."""
        return """Ты — эксперт по техническим стандартам ГОСТ/ОСТ/ТУ и регулярным выражениям Python 3 (re модуль).

### === ЖЁСТКИЕ ЗАПРЕТЫ (нарушение = брак) ===

1. **ТОЛЬКО ВИДИМЫЕ ПАРАМЕТРЫ**. Named group создаётся ТОЛЬКО если значение реально присутствует в исходной строке номенклатуры.
2. **ИМЯ ГРУППЫ ТИПА ИЗДЕЛИЯ — СТРОГО `тип_изделия`**.
3. **НЕТ ГРУППЕ `исполнение`**, если в примерах нет вариантов с `(N)`.
4. **НЕТ ГРУППЕ `технические_характеристики` как отдельной**.
5. **НЕТ `standard` в `required`**.
6. **НЕТ `толщина_покрытия` в regex**, если в строке нет явного числа толщины.

7. Разделители: `[-\s]+`. Не используй `.` как разделитель.
8. Десятичная точка: `(?:[.,]\d+)?` только внутри числовой группы.
9. Шаг резьбы: `[xXхХ×]` (включая кириллические х/Х).
10. Покрытие: `[\w.]+` матчит кириллицу (включая "ОСТ"!). Покрытие должно предшествовать `нтд_1`.
11. Порядок: `тип_изделия` → `исполнение` (опц.) → числовые → `покрытие` → `нтд_1`.
12. Полная строка: `^...$` обязательно.
13. Имена групп ≤30 символов.
14. Точка `.`: при сомнении разделяй `(?P<a>\d+)\.(?P<b>\d+)`.
15. **Группа `нтд_1` ДОЛЖНА матчить ПОЛНОЕ название стандарта**.
   - Для ОСТ: `(?P<нтд_1>ОСТ\s*1\s*\d+-\d+)`
   - Для ГОСТ: `(?P<нтд_1>ГОСТ\s*\d+-\d+)`
   - **ЗАПРЕЩЕНО** использовать `\d+` вместо `ОСТ`/`ГОСТ`.
16. **Пример правильного pattern**:
   ```
   ^(?P<тип_изделия>Гайка)\s*(?P<номинальный_диаметр_резьбы>\d+)(?:[xXхХ×](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?\s*[-\s]+(?P<покрытие>[\w.]+)\s*[-\s]+(?P<нтд_1>ОСТ\s*1\s*33049-80)$
   ```

### === ПРОВЕРКА ПЕРЕД ОТВЕТОМ ===
- [ ] Все группы реально видны в примерах?
- [ ] `тип_изделия` — первый и обязательный?
- [ ] `нтд_1` содержит полное название стандарта, а НЕ `\d+`?
- [ ] `^` и `$` присутствуют?"""

    def _build_prompt(self, standard: str, item_type: str, examples: List[Dict],
                     name: str = "", standard_info: Any = None) -> str:
        """Собрать промпт с ENS-примерами."""
        template = self._get_prompt_template()
        examples_text = self._format_examples(examples, standard, item_type)
        prompt = f"""# Тип изделия: {item_type}
# Стандарт: {standard}
# Провайдер: {{provider}}
# Модель: {{model}}
# Температура: {{temperature}}
# Время: {{timestamp}}
# ==================================================
{template}

{examples_text}

=== ЗАДАЧА ===

Создай regex-паттерн для стандарта {standard}, тип изделия {item_type}.
Используй ВИДИМЫЕ параметры из примеров выше.

=== ФОРМАТ ОТВЕТА ===

```json
{{
  "pattern": "^...$",
  "params": ["тип_изделия", ...],
  "required": ["тип_изделия", ...]
}}
```

Только JSON, без комментариев.
"""
        return prompt

    def generate_mask(
        self,
        standard: str,
        item_type: str,
        examples: Optional[List[Dict]] = None,
        name: str = "",
        standard_info: Any = None,
    ) -> Tuple[Optional[MaskGenerationResult], Optional[Dict]]:
        """
        Генерация маски через LLM с ENS-примерами.

        RETURNS:
            (MaskGenerationResult, metadata_dict) on success
            (None, None) on failure

        metadata_dict contains: provider, model, temperature,
                               tokens_prompt, tokens_completion
        """
        canon_std = canonicalize_standard(standard)
        if examples is None:
            examples = self._get_ens_examples(canon_std, item_type)
        prompt = self._build_prompt(canon_std, item_type, examples, name, standard_info)
        service, model, temperature = self._resolve_service()
        logger.info("[LLMMaskGenerator] Generating mask for %s/%s via %s (examples=%d)",
                   canon_std, item_type, service, len(examples))
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            for svc_name, client in self.clients.items():
                try:
                    result = self._call_llm(client, prompt, model, temperature)
                    if result:
                        mask = self._parse_mask_response(
                            result["text"], canon_std, item_type,
                            service=svc_name,
                            model=result.get("model", model),
                            temperature=temperature,
                            tokens_prompt=result.get("tokens_prompt", 0),
                            tokens_completion=result.get("tokens_completion", 0)
                        )
                        if mask:
                            meta = {
                                "provider": mask.service or svc_name,
                                "model": mask.model or model,
                                "temperature": mask.temperature or temperature,
                                "tokens_prompt": mask.tokens_prompt,
                                "tokens_completion": mask.tokens_completion,
                                "attempts": attempt,
                            }
                            logger.info(
                                "[LLMMaskGenerator] Generated mask via %s (attempt %d)",
                                svc_name, attempt
                            )
                            return mask, meta
                except Exception as e:
                    last_error = e
                    logger.debug("[LLMMaskGenerator] %s attempt %d failed: %s",
                               svc_name, attempt, e)
        logger.error("[LLMMaskGenerator] Failed after %d attempts: %s",
                     self.max_retries, last_error)
        return None, None

    def _resolve_service(self) -> Tuple[str, str, float]:
        """Определить сервис, модель и температуру."""
        service = ""
        model = ""
        temperature = 0.1
        if self.settings and hasattr(self.settings, "mask_generation"):
            mg = self.settings.mask_generation
            service = getattr(mg, "default_service", "")
            model = getattr(mg, "default_model", "")
            temperature = getattr(mg, "default_temperature", 0.1)
        if not service and self.settings and hasattr(self.settings, "default_service"):
            service = self.settings.default_service
        return service, model, temperature

    def _call_llm(self, client: Any, prompt: str, model: str, temperature: float) -> Optional[Dict]:
        """Вызвать LLM клиент с fallback на разные интерфейсы."""

        # DEBUG: log what we have
        client_type = type(client).__name__
        logger.debug("[LLMMaskGenerator] Calling %s with model=%s temp=%s", client_type, model, temperature)

        # Attempt 1: OpenAI-compatible messages format (most common)
        try:
            if hasattr(client, "chat") or hasattr(client, "generate"):
                method = getattr(client, "chat", None) or getattr(client, "generate", None)
                messages = [{"role": "user", "content": prompt}]

                # Try with messages first
                try:
                    response = method(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                    )
                except TypeError as te:
                    # Fallback: try with prompt= instead of messages=
                    logger.debug("[LLMMaskGenerator] messages failed, trying prompt: %s", te)
                    response = method(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                    )

                # Extract text from response
                text = None
                if isinstance(response, str):
                    text = response
                elif isinstance(response, dict):
                    # OpenAI format: choices[0].message.content
                    choices = response.get("choices", [])
                    if choices and isinstance(choices, list):
                        choice = choices[0]
                        if isinstance(choice, dict):
                            msg = choice.get("message", {})
                            text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                        else:
                            text = str(choice)
                    else:
                        text = response.get("text", "") or response.get("content", "")
                elif hasattr(response, "text"):
                    text = response.text
                elif hasattr(response, "content"):
                    text = response.content
                else:
                    text = str(response)

                if text and len(text) > 10:
                    tokens_prompt = getattr(client, "last_tokens_prompt", 0) or getattr(client, "_last_prompt_tokens", 0)
                    tokens_completion = getattr(client, "last_tokens_completion", 0) or getattr(client, "_last_completion_tokens", 0)
                    logger.info("[LLMMaskGenerator] %s success, response length=%d", client_type, len(text))
                    return {
                        "text": text,
                        "model": model,
                        "tokens_prompt": tokens_prompt,
                        "tokens_completion": tokens_completion,
                    }
        except Exception as e:
            logger.warning("[LLMMaskGenerator] %s call failed: %s", client_type, e)

        # Attempt 2: Direct HTTP-style complete()
        try:
            if hasattr(client, "complete"):
                response = client.complete(prompt, model=model, temperature=temperature)
                if response and len(str(response)) > 10:
                    return {
                        "text": str(response),
                        "model": model,
                        "tokens_prompt": 0,
                        "tokens_completion": 0,
                    }
        except Exception as e:
            logger.debug("[LLMMaskGenerator] complete() failed: %s", e)

        logger.error("[LLMMaskGenerator] All LLM call methods failed for %s", client_type)
        return None

    def _parse_mask_response(
        self,
        text: str,
        standard: str,
        item_type: str,
        service: str = "",
        model: str = "",
        temperature: float = 0.0,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
    ) -> Optional[MaskGenerationResult]:
        """Парсинг JSON-ответа LLM."""
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                logger.warning("[LLMMaskGenerator] No JSON found in response")
                return None
            data = json.loads(json_match.group())
            pattern = data.get("pattern", "")
            params = data.get("params", [])
            required = data.get("required", [])
            if not pattern or not pattern.startswith("^") or not pattern.endswith("$"):
                logger.warning("[LLMMaskGenerator] Invalid pattern: %s", pattern[:50])
                return None
            pattern = self._fix_pattern(pattern, standard, item_type)
            return MaskGenerationResult(
                pattern=pattern,
                params=params,
                required=required,
                standard=standard,
                item_type=item_type,
                raw_response=text,
                service=service,
                model=model,
                temperature=temperature,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
            )
        except Exception as e:
            logger.warning("[LLMMaskGenerator] Parse error: %s", e)
            return None

    def _fix_pattern(self, pattern: str, standard: str, item_type: str) -> str:
        """Исправить типичные ошибки LLM в regex."""
        if "ОСТ" in standard and r"(?P<нтд_1>\d+" in pattern:
            pattern = re.sub(
                r"\(\?P<нтд_1>\\d\+[^\)]*\)",
                f"(?P<нтд_1>{re.escape(standard)})",
                pattern
            )
            logger.info("[LLMMaskGenerator] Fixed нтд_1 for ОСТ standard")
        if "ГОСТ" in standard and r"(?P<нтд_1>\d+" in pattern:
            pattern = re.sub(
                r"\(\?P<нтд_1>\\d\+[^\)]*\)",
                f"(?P<нтд_1>{re.escape(standard)})",
                pattern
            )
            logger.info("[LLMMaskGenerator] Fixed нтд_1 for ГОСТ standard")
        if r"\\|" in pattern:
            pattern = pattern.replace(r"\\|", "|")
            logger.info("[LLMMaskGenerator] Fixed escaped alternation")
        if "наименование_типа" in pattern and "тип_изделия" not in pattern:
            pattern = pattern.replace("наименование_типа", "тип_изделия")
            logger.info("[LLMMaskGenerator] Fixed тип_изделия name")
        return pattern

