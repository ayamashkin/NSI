"""
LLM Mask Generator Module
Generates regex masks using LLM with ENS examples context.
"""
# =============================================================================
# FIX 2026-05-25 14:11 UTC+3:
# 1. FIXED _get_prompt_template: now READS file content from path specified
#    in mask_generation.prompt_template instead of returning the path string.
# 2. FIXED _save_debug_prompt/_save_debug_response: use debug_prompts_dir
#    from config (mask_generation.debug_prompts_dir) with subdirs prompts/
#    and responses/. Respects save_debug_prompts flag.
# 3. REMOVED duplicate header from _build_prompt.
# 4. FIXED _format_examples: ENS field mapping (нтд_1->стандарт/нтд,
#    тип_изделия->наименование_типа/тип).
# 5. ADDED placeholder replacement for {provider},{model},{temperature},{timestamp}.
# =============================================================================
# FIX 2026-05-22 19:11 UTC+3:
# 1. ADDED yaml.safe_load fallback in _parse_mask_response for single-quoted
#    JSON and unquoted keys.
# 2. ADDED debug logging of raw LLM response for diagnostics.
# =============================================================================
# FIX 2026-05-22 14:04 UTC+3:
# 1. RESTORED ENS examples injection into prompt.
# 2. FIXED return signature: generate_mask() now returns
#    (MaskGenerationResult, metadata_dict).
# =============================================================================
# 2026-05-21 08:23:07 51f335da — canonicalize_standard fixes.
# 2026-05-20 17:47:49 19e8ca02 — generate-masks metadata & stats-output.
# =============================================================================

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.standard_utils import canonicalize_standard

logger = logging.getLogger(__name__)


@dataclass
class MaskGenerationResult:
    """Результат генерации маски.

    FIX 2026-05-25: Добавлен __getitem__ для совместимости с cli.py,
    который обращается к mask['pattern'], mask['params'] и т.д.
    """
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

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for cli.py compatibility: mask['pattern'] etc."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: 'pattern' in mask."""
        return hasattr(self, key)


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

    @staticmethod
    def _get_example_value(ex: Dict, key: str) -> Optional[str]:
        """Получить значение из примера ЕНС с учётом маппинга полей.

        FIX 2026-05-25: Добавлены маппинги для ГОСТ-специфичных полей:
        класс_поле_допуска, группа_класс_прочности, свойства, тип_резьбы, марка_материала.
        """
        # Прямой доступ
        val = ex.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

        # Альтернативные имена полей
        alt_map = {
            "нтд_1": ["стандарт", "нтд"],
            "тип_изделия": ["наименование_типа", "тип"],
            "исполнение": ["вариант_исполнения", "исполнение"],
            "класс_поле_допуска": ["класс_допуска", "поле_допуска", "допуск"],
            "группа_класс_прочности": ["группа_прочности", "класс_прочности", "прочность"],
            "свойства": ["свойство", "код_свойств"],
            "тип_резьбы": ["резьба", "вид_резьбы"],
            "марка_материала": ["материал", "марка", "марка_материала_1"],
            "марка_материала_1": ["марка_материала", "материал"],
            "толщина_покрытия": ["толщина_покр", "покрытие_толщина"],
            "шаг_резьбы": ["шаг", "шаг_резьбы_1"],
            "длина": ["длина_изделия", "l"],
            "номинальный_диаметр_резьбы": ["диаметр", "d", "номинальный_диаметр"],
        }

        for alt in alt_map.get(key, []):
            val = ex.get(alt)
            if val is not None and str(val).strip():
                return str(val).strip()
        return None

    def _format_stats(self, examples: List[Dict]) -> str:
        """Форматировать статистику параметров для вставки в промпт."""
        if not examples:
            return "(нет данных)"
        lines = []
        param_counts = {}
        for ex in examples:
            for key in ["исполнение", "номинальный_диаметр_резьбы", "длина",
                       "шаг_резьбы", "покрытие", "толщина_проката_стенки_полки"]:
                val = self._get_example_value(ex, key)
                if val and val.strip():
                    param_counts[key] = param_counts.get(key, 0) + 1
        total = len(examples)
        for key, count in sorted(param_counts.items(), key=lambda x: -x[1]):
            lines.append(f" {key}: {count} из {total} ({count/total*100:.0f}%)")
        return "\n".join(lines) if lines else "(нет параметров)"

    @staticmethod
    def _is_value_in_name(val: str, name: str) -> bool:
        """Проверить, что значение параметра присутствует в строке номенклатуры.

        FIX 2026-05-25: Robust fuzzy matching:
        - float 16.0 → матчит 16
        - покрытия Кд6-9.хр → матчит Кд, Кд.фос, Кд.фос.окс
        - покрытия Ц9.хр → матчит Ц
        - покрытия Ан.Окс.хр → матчит Ан, Окс, Ан.Окс
        """
        if not val or not name:
            return False
        val_str = str(val).strip().replace(",", ".")
        name_lower = name.lower().replace(",", ".")

        # 1. Точное совпадение
        if val_str.lower() in name_lower:
            return True

        # 2. Для float с .0: 16.0 матчит 16
        if '.' in val_str and val_str.endswith('.0'):
            int_part = val_str[:-2]
            if int_part and int_part.lower() in name_lower:
                return True

        # 3. Для покрытий и других составных значений:
        #    извлекаем все буквенные части и проверяем каждую
        letter_parts = re.findall(r"[a-zA-Zа-яА-Я]+", val_str)
        for part in letter_parts:
            if part.lower() in name_lower:
                return True

        # 4. Префикс до первой цифры (Кд6-9 → Кд)
        prefix_match = re.match(r"^([a-zA-Zа-яА-Я\.\-]+)", val_str)
        if prefix_match:
            prefix = prefix_match.group(1).rstrip('.-')
            # Убираем цифры из префикса
            prefix_clean = re.sub(r"[0-9]", "", prefix).rstrip('.-')
            if prefix_clean and prefix_clean.lower() in name_lower:
                return True

        # 5. Проверка без цифр вообще (Кд3.фос → Кд.фос)
        no_digits = re.sub(r"[0-9]", "", val_str).strip(".- ")
        if no_digits and no_digits.lower() in name_lower:
            return True

        return False

    def _format_examples(self, examples: List[Dict], standard: str, item_type: str) -> str:
        """Форматировать ENS-примеры для вставки в промпт.

        FIX 2026-05-25:
        1. Убран наименование_типа из видимых (дублирует тип_изделия).
        2. Убран нтд_1 из каждого примера (загромождает, стандарт и так известен).
        3. Добавлен fuzzy matching (_is_value_in_name) для float и покрытий.
        4. Расширен check_keys всеми возможными видимыми параметрами.
        5. Добавлена подсказка про структуру строки.
        """
        if not examples:
            return "(примеры отсутствуют)"

        lines = []
        lines.append(f"Структура: <{item_type}> [исполнение] <параметры> <покрытие> {standard}")
        lines.append("")

        for i, ex in enumerate(examples[:10], 1):
            name = ex.get("наименование", ex.get("полное_наименование", ""))
            if not name:
                continue
            visible = []
            hidden = []

            # Полный набор параметров, которые могут быть видимыми в строке
            check_keys = [
                "тип_изделия", "исполнение",
                "номинальный_диаметр_резьбы", "длина", "шаг_резьбы",
                "покрытие", "толщина_покрытия",
                "группа_класс_прочности", "класс_поле_допуска", "свойства",
                "марка_материала", "марка_материала_1", "тип_резьбы",
                "толщина_проката_стенки_полки",
                "наружный_диаметр_диаметр_вписанного_круга_сторона_квадрата_стороны_поперечного_сечения",
            ]

            for key in check_keys:
                val = self._get_example_value(ex, key)
                if not val:
                    continue
                val_str = val.strip()

                # наименование_типа — метаданные, не показываем как видимый параметр
                if key == "наименование_типа":
                    continue

                # тип_изделия: проверяем по началу строки
                if key == "тип_изделия":
                    if name.lower().startswith(val_str.lower()):
                        visible.append((key, val_str))
                    else:
                        hidden.append((key, val_str))
                    continue

                if self._is_value_in_name(val_str, name):
                    visible.append((key, val_str))
                else:
                    hidden.append((key, val_str))

            lines.append(f'{i}. Исходное: "{name}"')
            if visible:
                vis_str = " ".join([f"(?P<{k}>{v})" for k, v in visible])
                lines.append(f" Видимые: {vis_str}")
            if hidden:
                hid_str = ", ".join([f"{k}={v}" for k, v in hidden])
                lines.append(f" Скрытые: {hid_str}")
            lines.append("")

        return "\n".join(lines)

    def _get_prompt_template(self) -> str:
        """Загрузить шаблон промпта.

        FIX 2026-05-25: Читаем содержимое файла по пути из
        mask_generation.prompt_template, а не возвращаем сам путь как строку.
        """
        if self.settings and hasattr(self.settings, "mask_generation"):
            mg = self.settings.mask_generation
            template_path = getattr(mg, "prompt_template", "")
            if template_path:
                p = Path(template_path)
                if p.exists():
                    content = p.read_text(encoding="utf-8")
                    logger.info("[LLMMaskGenerator] Loaded prompt template from %s", template_path)
                    return content
                else:
                    logger.warning("[LLMMaskGenerator] prompt_template path not found: %s", template_path)
        # Fallback paths
        for path in [
            "prompts/templates/mask_generation.txt",
            "prompts/mask_generation.txt",
            "config/mask_generation.txt",
        ]:
            p = Path(path)
            if p.exists():
                logger.info("[LLMMaskGenerator] Loaded prompt template from fallback %s", path)
                return p.read_text(encoding="utf-8")
        logger.warning("[LLMMaskGenerator] No template file found, using default")
        return self._default_template()

    def _default_template(self) -> str:
        r"""Default template with v3 rules."""
        return r"""Ты — эксперт по техническим стандартам ГОСТ/ОСТ/ТУ и регулярным выражениям Python 3 (re модуль).

### === ЖЁСТКИЕ ЗАПРЕТЫ (нарушение = брак) ===

1. **ТОЛЬКО ВИДИМЫЕ ПАРАМЕТРЫ**. Named group создаётся ТОЛЬКО если значение реально присутствует в исходной строке номенклатуры.
2. **ИМЯ ГРУППЫ ТИПА ИЗДЕЛИЯ — СТРОГО `тип_изделия`**. Не `наименование_типа`, не `тип`, не `вид`. Только `тип_изделия`.
3. **НЕТ ГРУППЕ `исполнение`**, если в примерах нет вариантов с `(N)` или `N-` после типа изделия. Если исполнение есть — группа опциональная `(?:\(?(?P<исполнение>\d+)\)?)?`.
4. **НЕТ ГРУППЕ `технические_характеристики` как отдельной**. Если в строке только покрытие (Бп, Кд, Хим.Пас) — используй только `покрытие`.
5. **НЕТ `standard` в `required`**. `standard` — метаданные, не извлекается из строки.
6. **НЕТ `толщина_покрытия` в regex**, если в строке нет явного числа толщины.

### === ПРАВИЛА РАЗДЕЛИТЕЛЕЙ ===
7. Разделители между параметрами: `[-\s]+` (тире, пробел, таб). Не используй `.` как разделитель — это десятичная точка.
8. Десятичная точка/запятая: `(?:[.,]\d+)?` только внутри числовой группы. Пример: `\d+(?:[.,]\d+)?` для 12.5.

### === КИРИЛЛИЦА В REGEX ===
9. **Шаг резьбы**: используй `[xXхХ×]` (включая кириллические х/Х и × U+00D7). Пример: `M12х1.5` должно матчиться.
10. **Покрытие**: `[\w.]+` матчит кириллицу (включая "ОСТ"!). Поэтому покрытие должно строго предшествовать `нтд_1` в паттерне, или использовать негативный lookahead `(?!\w)`.

### === СТРУКТУРА ПАТТЕРНА ===
11. Порядок групп: `тип_изделия` → `исполнение` (опц.) → числовые параметры → `покрытие` → `нтд_1`.
12. Полная строка: `^...$` обязательно.
13. Имена групп ≤30 символов, только [a-zA-Zа-яА-Я0-9_].

### === ПРАВИЛО ТОЧКИ ===
14. Точка `.` в номенклатуре:
   - ДЕСЯТИЧНАЯ: если дробная часть имеет смысл (12.5 мм).
   - РАЗДЕЛИТЕЛЬ: если после точки ровно 2 цифры-кода (100.58 → длина=100, группа=58).
   - **ПРАВИЛО**: при сомнении разделяй: `(?P<a>\d+)\.(?P<b>\d+)` вместо `(?P<x>\d+(?:[.,]\d+)?)`.

### === КРИТИЧНО: ФОРМАТ НТД_1 ===
15. **Группа `нтд_1` ДОЛЖНА матчить ПОЛНОЕ название стандарта** из заголовка этого промпта.
   - Для ОСТ: `(?P<нтд_1>ОСТ\s*1\s*\d+-\d+)` или `(?P<нтд_1>ОСТ\s*1\s*33049-80)`
   - Для ГОСТ: `(?P<нтд_1>ГОСТ\s*\d+-\d+)` или `(?P<нтд_1>ГОСТ\s*7795-70)`
   - **ЗАПРЕЩЕНО** использовать `\d+` вместо `ОСТ`/`ГОСТ` — это сломает парсинг.
16. **Пример правильного pattern для ОСТ 1 33049-80 / Гайка**:
   ```
   ^(?P<тип_изделия>Гайка)\s*(?P<номинальный_диаметр_резьбы>\d+)(?:[xXхХ×](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?\s*[-\s]+(?P<покрытие>[\w.]+)\s*[-\s]+(?P<нтд_1>ОСТ\s*1\s*33049-80)$
   ```

### === ФОРМАТ ОТВЕТА ===

```json
{
  "pattern": "^...$",
  "params": ["тип_изделия", "номинальный_диаметр_резьбы", "покрытие", "нтд_1", "шаг_резьбы"],
  "required": ["тип_изделия", "номинальный_диаметр_резьбы", "покрытие", "нтд_1"]
}
```

- `params`: ВСЕ named group имена (только видимые в строке).
- `required`: обязательные параметры (все кроме опциональных: исполнение, шаг_резьбы).

### === ПРОВЕРКА ПЕРЕД ОТВЕТОМ ===

Перед выводом JSON проверь:
- [ ] Все группы из `params` реально видны в примерах?
- [ ] `тип_изделия` — первый и обязательный?
- [ ] `standard` НЕТ в `required`?
- [ ] `исполнение` есть только если в примерах есть `(N)`?
- [ ] Шаг резьбы использует `[xXхХ×]`?
- [ ] Покрытие не перехватывает `ОСТ`?
- [ ] `нтд_1` содержит полное название стандарта (`ОСТ 1 XXXXX-80` или `ГОСТ XXXX-XX`), а НЕ `\d+`?
- [ ] `^` и `$` присутствуют?

=== СТРОГОЕ СООТВЕТСТВИЕ ===

Выведи ОДИН JSON-объект. Без комментариев, markdown, объяснений — только JSON."""

    def _build_prompt(self, standard: str, item_type: str, examples: List[Dict],
                      name: str = "", standard_info: Any = None) -> str:
        """Собрать промпт с ENS-примерами.

        FIX 2026-05-25:
        1. Добавлен header с мета-информацией (provider, model, temperature, timestamp).
        2. Убрано дублирование: examples_text содержит только примеры,
           stats_text — только статистику (placeholder из template).
        """
        template = self._get_prompt_template()
        examples_text = self._format_examples(examples, standard, item_type)
        stats_text = self._format_stats(examples)

        service, model, temperature = self._resolve_service()

        # --- Замена placeholder'ов в template ---
        replacements = {
            "{examples_text}": examples_text,
            "{stats_text}": stats_text,
            "{item_type}": item_type,
            "{standard}": standard,
            "{provider}": service or "LLM",
            "{model}": model or "unknown",
            "{temperature}": str(temperature),
            "{timestamp}": datetime.now().isoformat(),
        }
        for placeholder, value in replacements.items():
            if placeholder in template:
                template = template.replace(placeholder, value)

        if "{params_list}" in template:
            visible = self._extract_visible_params(examples)
            template = template.replace("{params_list}", json.dumps(visible, ensure_ascii=False))
        if "{required_list}" in template:
            visible = self._extract_visible_params(examples)
            optional = {"исполнение", "шаг_резьбы", "толщина_покрытия", "variant"}
            required = [p for p in visible if p not in optional]
            template = template.replace("{required_list}", json.dumps(required, ensure_ascii=False))

        # --- Добавляем ЗАДАЧА/ФОРМАТ только если их нет в template ---
        has_task = "=== ЗАДАЧА ===" in template or "ЗАДАЧА:" in template.lower()
        has_format = "=== ФОРМАТ ОТВЕТА ===" in template or "```json" in template

        task_section = ""
        if not has_task:
            task_section = f"""

=== ЗАДАЧА ===

Создай regex-паттерн для стандарта {standard}, тип изделия {item_type}.
Используй ВИДИМЫЕ параметры из примеров выше."""

        format_section = ""
        if not has_format:
            format_section = """

=== ФОРМАТ ОТВЕТА ===

```json
{{
  "pattern": "^...$",
  "params": ["тип_изделия", ...],
  "required": ["тип_изделия", ...]
}}
```

Только JSON, без комментариев."""

        # Header с мета-информацией (не для regex — для отладки)
        header = f"""# Тип изделия: {item_type}
# Стандарт: {standard}
# Провайдер: {service or 'LLM'}
# Модель: {model or 'unknown'}
# Температура: {temperature}
# Время: {datetime.now().isoformat()}
# =================================================="""

        prompt = header + "\n" + template + task_section + format_section
        return prompt

    def _extract_visible_params(self, examples: List[Dict]) -> List[str]:
        """Извлечь список видимых параметров из ENS-примеров для подстановки в промпт."""
        if not examples:
            return ["тип_изделия", "номинальный_диаметр_резьбы", "покрытие", "нтд_1"]
        visible = set()
        for ex in examples[:5]:
            name = ex.get("наименование", ex.get("полное_наименование", ""))
            if not name:
                continue
            name_lower = name.lower()
            for key in ["тип_изделия", "исполнение", "номинальный_диаметр_резьбы",
                       "длина", "шаг_резьбы", "покрытие", "нтд_1", "нтд_2"]:
                val = self._get_example_value(ex, key)
                if val:
                    val_str = val.strip().lower()
                    if val_str in name_lower:
                        visible.add(key)
            # Обязательные поля всегда добавляем
            visible.add("тип_изделия")
            if any("нтд" in k for k in visible):
                visible.add("нтд_1")
        return list(visible)

    def _get_debug_dir(self) -> Optional[Path]:
        """Получить путь к debug-директории из конфига.

        FIX 2026-05-25: Используем mask_generation.debug_prompts_dir из config.
        Файлы сохраняются напрямую в эту директорию (без подпапок).
        Возвращает None если save_debug_prompts=false.
        """
        if not self.settings:
            return None
        mg = getattr(self.settings, "mask_generation", None)
        if not mg:
            return None
        if not getattr(mg, "save_debug_prompts", False):
            return None
        debug_dir = getattr(mg, "debug_prompts_dir", "prompts/debug")
        if not debug_dir:
            return None
        return Path(debug_dir)

    def _save_debug_prompt(self, standard: str, item_type: str, prompt: str) -> None:
        """Сохранить промпт в debug-директорию.

        Формат имени: {item_type}_{standard}.txt
        Пример: Болт_ОСТ 1 31141-80.txt
        """
        base_dir = self._get_debug_dir()
        if not base_dir:
            return
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{item_type}_{standard}.txt"
            path = base_dir / fname
            path.write_text(prompt, encoding='utf-8')
            logger.debug("[LLMMaskGenerator] Prompt saved to %s", path)
        except Exception as e:
            logger.debug("[LLMMaskGenerator] Failed to save prompt: %s", e)

    def _save_debug_response(self, standard: str, item_type: str, response: str,
                             service: str, attempt: int) -> None:
        """Сохранить ответ LLM в debug-директорию.

        Формат имени: {item_type}_{standard}_a{attempt}.txt
        Пример: Болт_ОСТ 1 31141-80_a1.txt
        """
        base_dir = self._get_debug_dir()
        if not base_dir:
            return
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{item_type}_{standard}_a{attempt}.txt"
            path = base_dir / fname
            path.write_text(response, encoding='utf-8')
            logger.debug("[LLMMaskGenerator] Response saved to %s", path)
        except Exception as e:
            logger.debug("[LLMMaskGenerator] Failed to save response: %s", e)

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

        # DEBUG: save prompt before sending
        self._save_debug_prompt(canon_std, item_type, prompt)

        service, model, temperature = self._resolve_service()
        logger.info("[LLMMaskGenerator] Generating mask for %s/%s via %s (examples=%d)",
                   canon_std, item_type, service, len(examples))
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            for svc_name, client in self.clients.items():
                try:
                    result = self._call_llm(client, prompt, model, temperature)
                    if result:
                        # DEBUG: save raw response
                        self._save_debug_response(canon_std, item_type, result["text"],
                                                  svc_name, attempt)

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
        client_type = type(client).__name__
        logger.debug("[LLMMaskGenerator] Calling %s with model=%s temp=%s", client_type, model, temperature)

        # Attempt 1: OpenAI-compatible messages format (most common)
        try:
            if hasattr(client, "chat") or hasattr(client, "generate"):
                method = getattr(client, "chat", None) or getattr(client, "generate", None)
                messages = [{"role": "user", "content": prompt}]

                try:
                    response = method(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                    )
                except TypeError as te:
                    logger.debug("[LLMMaskGenerator] messages failed, trying prompt: %s", te)
                    response = method(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                    )

                text = None
                if isinstance(response, str):
                    text = response
                elif isinstance(response, dict):
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
                    logger.debug("[LLM] Raw content from %s (len=%d): %r", client_type, len(text), text[:500])
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
        """Парсинг JSON-ответа LLM.

        FIX 2026-05-25: MTSAIClient возвращает Python dict repr (одинарные кавычки).
        Добавлен ast.literal_eval + извлечение из content/raw.
        """
        if not text:
            logger.warning("[LLMMaskGenerator] Empty response text")
            return None

        logger.debug("[LLM] Raw response for %s/%s (len=%d): %r",
                    standard, item_type, len(text), text[:800])

        data = None
        candidate = None

        # --- STRATEGY 1: Python dict repr (MTSAIClient format) ---
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                if "content" in parsed and isinstance(parsed["content"], dict):
                    data = parsed["content"]
                    logger.debug("[LLM] Parsed via ast.literal_eval -> content")
                elif "raw" in parsed and isinstance(parsed["raw"], str):
                    text = parsed["raw"]
                    logger.debug("[LLM] Parsed via ast.literal_eval -> raw, re-parsing")
                else:
                    data = parsed
                    logger.debug("[LLM] Parsed via ast.literal_eval -> direct dict")
        except (ValueError, SyntaxError, TypeError) as e:
            logger.debug("[LLM] ast.literal_eval failed: %s", e)

        # --- STRATEGY 2: YAML fallback (single quotes, unquoted keys) ---
        if data is None:
            try:
                import yaml
                data = yaml.safe_load(text)
                if isinstance(data, dict):
                    if "content" in data and isinstance(data["content"], dict):
                        data = data["content"]
                        logger.debug("[LLM] Parsed via yaml.safe_load -> content")
                    elif "raw" in data and isinstance(data["raw"], str):
                        text = data["raw"]
                        logger.debug("[LLM] Parsed via yaml.safe_load -> raw, re-parsing")
                    else:
                        logger.debug("[LLM] Parsed via yaml.safe_load -> direct dict")
                else:
                    data = None
            except Exception as e:
                logger.debug("[LLM] yaml.safe_load failed: %s", e)

        # --- STRATEGY 3: Markdown code block inside text ---
        if data is None:
            for prefix in ["```json", "```python", "```"]:
                start = text.find(prefix)
                if start >= 0:
                    start += len(prefix)
                    end = text.find("```", start)
                    if end >= 0:
                        candidate = text[start:end].strip()
                        break
            if candidate:
                try:
                    data = json.loads(candidate)
                    logger.debug("[LLM] Parsed markdown JSON via json.loads")
                except json.JSONDecodeError:
                    try:
                        import yaml
                        data = yaml.safe_load(candidate)
                        if isinstance(data, dict):
                            logger.debug("[LLM] Parsed markdown JSON via yaml.safe_load")
                        else:
                            data = None
                    except Exception:
                        pass

        # --- STRATEGY 4: Balanced braces ---
        if data is None:
            for start_match in re.finditer(r"(?m)^[ \t]*\{", text):
                pos = start_match.start()
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
                                    data = json.loads(candidate)
                                    logger.debug("[LLM] Parsed balanced JSON")
                                except json.JSONDecodeError:
                                    try:
                                        import yaml
                                        data = yaml.safe_load(candidate)
                                        if isinstance(data, dict):
                                            logger.debug("[LLM] Parsed balanced JSON via yaml")
                                        else:
                                            data = None
                                    except Exception:
                                        pass
                                break
                if data is not None:
                    break

        # --- STRATEGY 5: Simple regex fallback ---
        if data is None:
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if json_match:
                candidate = json_match.group()
                try:
                    data = json.loads(candidate)
                    logger.debug("[LLM] Parsed simple JSON")
                except json.JSONDecodeError:
                    try:
                        import yaml
                        data = yaml.safe_load(candidate)
                        if isinstance(data, dict):
                            logger.debug("[LLM] Parsed simple JSON via yaml")
                        else:
                            data = None
                    except Exception:
                        pass

        # --- Validate and build result ---
        if data is None or not isinstance(data, dict):
            logger.warning(
                "[LLMMaskGenerator] Failed to parse any JSON for %s/%s. Preview: %r",
                standard, item_type, text[:300]
            )
            return None

        pattern = data.get("pattern", "")
        params = data.get("params", [])
        required = data.get("required", [])
        if not pattern or not pattern.startswith("^") or not pattern.endswith("$"):
            logger.warning("[LLMMaskGenerator] Invalid pattern: %s", pattern[:50])
            return None
        pattern = self._fix_pattern(pattern, standard, item_type)
        result = MaskGenerationResult(
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
        return self._sanitize_mask_result(result)

    def _fix_pattern(self, pattern: str, standard: str, item_type: str) -> str:
        """Исправить типичные ошибки LLM в regex."""
        if "ОСТ" in standard and r"(?P<нтд_1>\d+" in pattern:
            pattern = re.sub(
                r"\(\?P<нтд_1>\d+[^\)]*\)",
                f"(?P<нтд_1>{re.escape(standard)})",
                pattern
            )
            logger.info("[LLMMaskGenerator] Fixed нтд_1 for ОСТ standard")
        if "ГОСТ" in standard and r"(?P<нтд_1>\d+" in pattern:
            pattern = re.sub(
                r"\(\?P<нтд_1>\d+[^\)]*\)",
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

    def _sanitize_mask_result(self, result: MaskGenerationResult) -> MaskGenerationResult:
        """Универсальная очистка маски от типичных LLM-ошибок.

        FIX 2026-05-25: Не хардкод под стандарт, а универсальные правила:
        1. Удалить наименование_типа, если есть тип_изделия.
        2. Исправить опечатки в именах групп (тип_2изделия → тип_изделия).
        3. Убедиться, что required ⊆ params.
        4. Удалить пустые опциональные группы из pattern.
        """
        pattern = result.pattern
        params = list(result.params)
        required = list(result.required)

        # 1. Удалить дублирующий наименование_типа
        if "тип_изделия" in params and "наименование_типа" in params:
            params.remove("наименование_типа")
            if "наименование_типа" in required:
                required.remove("наименование_типа")
            # Удалить группу наименование_типа из pattern (с разделителем)
            pattern = re.sub(r"\(\?P<наименование_типа>[^)]+\)(?:\s*[-\s]*)?", "", pattern)
            logger.info("[LLMMaskGenerator] Sanitized: removed duplicate наименование_типа")

        # 2. Исправить типичные опечатки
        typo_fixes = {
            "тип_2изделия": "тип_изделия",
            "тип_изделия2": "тип_изделия",
            "тип_изделеия": "тип_изделия",
        }
        for bad, good in typo_fixes.items():
            if bad in params:
                params = [good if p == bad else p for p in params]
                required = [good if p == bad else p for p in required]
                pattern = pattern.replace(f"(?P<{bad}>", f"(?P<{good}>")
                logger.info("[LLMMaskGenerator] Sanitized: fixed typo %s → %s", bad, good)

        # 3. required должен быть подмножеством params
        required = [p for p in required if p in params]

        # 4. Удалить пустые опциональные группы (?P<name>)? из pattern
        pattern = re.sub(r"\(\?P<[^>]+>\)\?", "", pattern)

        result.pattern = pattern
        result.params = params
        result.required = required
        return result