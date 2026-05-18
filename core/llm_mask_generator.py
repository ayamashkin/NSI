"""
LLM Mask Generator
Генерация regex-масок для стандартов через LLM с fallback по провайдерам.

LAST_FIXES:
  2026-05-18 12:45 UTC+3 — _build_prompt: наименования + значимые заполненные поля (без None и _-служебных)
  2026-05-18 11:50 UTC+3 — Восстановлены _save_debug_prompt/_save_debug_response
  2026-05-18 11:16 UTC+3 — _parse_response: поддержка ```python + balanced braces JSON extraction
  2026-05-18 10:35 UTC+3 — _call_llm: работа с Dict-ответами (success/content/raw/error/model)
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from api_clients.base import BaseLLMClient
from config.settings import get_settings

logger = logging.getLogger(__name__)


# Поля, которые НЕ несут смысловой нагрузки для генерации маски
_SKIP_FIELDS = {
    '_ens_category', '_source_file', '_available_columns', '_detected_prompt_id',
    '_detected_category', '_detection_confidence', '_implicit_тип',
    '_original_код', '_original_наименование', '_original_полное_наименование',
    '_original_нтд', '_original_нтд.1', '_original_марка_материала',
    '_original_длина', '_original_класс_(поле)_допуска', '_original_покрытие',
    '_original_группа_(класс)_прочности', '_original_шаг_резьбы',
    '_original_исполнение', '_original_наименование_типа', '_original_тип_резьбы',
    '_original_номинальный_диаметр_резьбы', '_original_толщина_покрытия',
    '_original_марка_материала.1', '_original_наименование.1',
    'код', 'mdm_key', 'ссылка', 'дата_создания', 'дата_последнего_изменения',
    'автор', 'автор_последнего_изменения', 'комментарий_эксперта',
    'базовая_единица_измерения', 'единицы_измерения', 'наличие_бп',
    'организация_корпорации', 'торговая_марка', 'торговая_марка.1',
    'гражданская_продукция', 'заблокировано', 'вести_учет_по_характеристикам',
    'специальная_приемка', 'соответствие_тр_тс', 'тр_тс', 'оквэд2', 'оквэд2_код',
    'окпд2', 'окпд2_код', 'оквэд2_код', 'окпд2_код', 'классификатор_енс',
    'классификатор_енс_код', 'пометка_удаления', 'тип', 'тип_позиции',
    'вид_специальной_приемки', 'технические_характеристики', 'свойства',
    'обозначение_тип_артикул', 'нтд_на_материал', 'нтд_на_материал.1',
    'каталожный_номер', 'класс_качества', 'типоразмер', 'особые_условия',
    'конструкция', 'состояние_поверхности', 'серия_товара_изделия_продукта',
    'категория_качества', 'видприемки', 'номинальный_диаметр',
    'обозначение_длины_резьбы', 'наличие_фаски', 'тип_шлица', 'наличие_отверстий',
    'класс_степень_точности', 'обозначение_болта', 'обозначение_винта',
    'обозначение_гайки', 'обозначение_шайбы', 'обозначение_заклепки',
    'вариант_исполнения', 'категория_размещения', 'марка_товара_изделия_продукта',
    'число_шагов_резьбы_на_дюйм', 'обозначение_размера_под_ключ',
    'тип_круга_головки', 'тип_болта', 'условное_обозначение_длины',
    'индекс_диаметра', 'климатическое_исполнение', 'предельные_отклонения',
    'обозначение_диаметра_корпуса', 'толщина_проката_стенки_полки',
    'состояние_материала', 'диаметр_наружный_режущей_части', 'шкала_твердости',
    'длина_хвостовика', 'состояние_поставки_металлопроката', 'диаметр_цилиндра',
    'обозначение_ширины_фаски', 'диаметр_проволоки', 'высота', 'марка_проволоки',
    'число_витков', 'температурный_диапазон_выкипания_эксплуатации',
    'расстояние_между_гранями', 'условное_обозначение_номера_цвета',
    'условное_обозначение_толщины', 'форма_поставки', 'вид_и_сторона_покрытия',
    'тип_зуба', 'лакокрасочное_и_полимерное_покрытие', 'давление', 'радиус', 'уклон',
    'способ_изготовления', 'категория_проката', 'направление_резьбы',
    'обозначение_толщины_покрытия', 'обозначение', 'внутренний_диаметр_условный_проход',
    'цвет', 'шаг_второй_резьбы', 'класс_поле_допуска_ввинчиваемого_конца',
    'расстояние_между_осями', 'условное_обозначение_марки_материала',
    'тип_отделки_концов', 'назначение_материала', 'способ_получения_стали',
    'длина_ввинчиваемого_конца', 'диаметр_посадочного_отверстия',
    'длина_общая_oal', 'твердость', 'комплектность', 'обозначение',
    'длина_резьбы', 'наружный_диаметр_диаметр_вписанного_круга_сторона_квадрата_стороны_поперечного_сечения',
}


def _extract_significant_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Извлекает значимые заполненные поля из записи ЕНС (без None и служебных)."""
    significant = {}
    for key, val in item.items():
        if key.startswith('_') or key in _SKIP_FIELDS:
            continue
        if val is None or val == '' or val == 'Нет':
            continue
        # Только простые типы
        if isinstance(val, (str, int, float)):
            significant[key] = val
    return significant


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

    def generate_mask(
        self,
        standard: str,
        item_type: str,
        examples: List[str]
    ) -> tuple[Optional[Dict[str, Any]], None]:
        """Генерация маски через LLM с fallback по провайдерам."""
        prompt = self._build_prompt(standard, item_type, examples)
        self._save_debug_prompt(standard, item_type, prompt)

        for attempt in range(1, self.max_retries + 1):
            for provider in self.provider_priority:
                logger.info("Attempt %d/%d via %s", attempt, self.max_retries, provider)
                response = self._call_llm(provider, prompt, attempt)
                if response:
                    self._save_debug_response(standard, item_type, attempt, response)
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

    def _build_prompt(self, standard: str, item_type: str, examples: List[Any]) -> str:
        """Строит промпт для LLM: наименования + значимые заполненные поля."""
        # examples может быть List[str] (только имена) или List[Dict] (полные записи ЕНС)
        entries = []
        for ex in examples[:20]:
            if isinstance(ex, dict):
                name = ex.get('наименование') or ex.get('полное_наименование') or ex.get('name', '')
                fields = _extract_significant_fields(ex)
                # Убираем дубли наименования из fields
                fields.pop('наименование', None)
                fields.pop('полное_наименование', None)
                entries.append((name, fields))
            else:
                entries.append((str(ex), {}))

        # Уникальные наименования (сохраняем порядок)
        seen = set()
        unique_entries = []
        for name, fields in entries:
            if name and name not in seen:
                seen.add(name)
                unique_entries.append((name, fields))
                if len(unique_entries) >= 15:
                    break

        # Формируем текст примеров
        examples_text = ""
        for i, (name, fields) in enumerate(unique_entries, 1):
            examples_text += f"  {i}. {name}"
            if fields:
                # Форматируем поля компактно: ключ=значение
                field_strs = [f"{k}={v}" for k, v in fields.items()]
                examples_text += f"  [{', '.join(field_strs)}]"
            examples_text += "\n"

        prompt = f"""На основе следующих примеров наименований изделий типа "{item_type}" по стандарту {standard}:

{examples_text}
Создай Python regex паттерн с именованными группами (?P<<name>...) для извлечения параметров.
Верни результат строго в формате JSON:

{{
    "pattern": "regex pattern here",
    "params": ["param1", "param2", ...],
    "required": ["param1", ...]
}}

Правила:
- pattern: валидный Python regex (флаг re.IGNORECASE будет применён при использовании)
- params: список имён групп из pattern
- required: список обязательных параметров (не может быть пустым)
"""
        return prompt

    def _call_llm(self, provider: str, prompt: str, attempt: int) -> Optional[str]:
        """
        Вызов LLM для генерации маски.
        """
        client = self.clients.get(provider)
        if not client:
            return None

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

        # Temperature с ростом по attempt
        temperature = min(0.1 + attempt * 0.1, 0.5)

        try:
            response = client.complete(
                prompt=prompt,
                model=model,
                temperature=temperature
            )

            # response — dict с success, content, raw, error, model
            if response and response.get('success'):
                # Если content уже распарсен (dict), вернуть его как JSON-строку
                content = response.get('content')
                if isinstance(content, dict):
                    logger.debug(
                        "[LLM] Provider %s returned pre-parsed dict, serializing to JSON",
                        provider
                    )
                    return json.dumps(content, ensure_ascii=False)
                # Иначе вернуть raw текст
                raw = response.get('raw', '')
                if raw:
                    return raw
                # Если raw пустой, но content есть (не dict) — вернуть content
                if content and not isinstance(content, dict):
                    return str(content)
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

        # Стратегия 2: Balanced braces (robust)
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
            # если brace_count не 0 — не валидный

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

    # ------------------------------------------------------------------
    # DEBUG SAVE METHODS
    # ------------------------------------------------------------------

    def _sanitize_filename(self, text: str) -> str:
        """Санитизация строки для использования в имени файла."""
        sanitized = re.sub(r'[\\/:*?"<>|]', '_', str(text))
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')[:80]

    def _save_debug_prompt(self, standard: str, item_type: str, prompt: str):
        self._save_debug_file(standard, item_type, "prompt", prompt)

    def _save_debug_response(self, standard: str, item_type: str, attempt: int, response: str):
        self._save_debug_file(standard, item_type, f"response_a{attempt}", response)

    def _save_debug_file(self, standard: str, item_type: str, suffix: str, content: str):
        """Сохранение debug-файла (prompt/response)."""
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
                f"# Дата: {datetime.now().isoformat()}\n"
                f"# {'=' * 50}\n\n"
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write(content)

            logger.info("[LLMMaskGenerator] Сохранён debug: %s", filepath)
        except Exception as e:
            logger.warning("[LLMMaskGenerator] Ошибка сохранения debug: %s", e)