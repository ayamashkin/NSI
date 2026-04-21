# Промпт для генерации regex-маски (LLMMaskGenerator._build_prompt)

## Текущий промпт (шаблон с подстановками)

```
Ты - эксперт по техническим стандартам и регулярным выражениям.

Создай regex-паттерн для извлечения параметров из номенклатуры {item_type} по стандарту {standard}.

Примеры:
{examples_text}
{context_text}
Требования:
1. Используй named groups (?P<name>...)
2. Поддерживай разные варианты написания
3. Учитывай специфику стандарта {standard}

Ответь ТОЛЬКО в формате JSON:
{
  "pattern": "regex with named groups",
  "params": ["param1", "param2"],
  "required": ["required1"]
}
```

---

## Пример с реальными данными

**Входные параметры:**
- `standard` = `ГОСТ 7798-70`
- `item_type` = `болт`
- `examples` (первые 5 записей из ЕСН):

```
1. Болт М16х130.52.019 ГОСТ 7798-70
2. Болт М10х60.22.6.019 ГОСТ 7798-70
3. Болт М12х80.36.6.019 ГОСТ 7798-70
4. Болт М20х160.52.6.019 ГОСТ 7798-70
5. Болт М8х40.22.6.019 ГОСТ 7798-70
```

**Итоговый промпт, отправляемый в LLM:**

```
Ты - эксперт по техническим стандартам и регулярным выражениям.

Создай regex-паттерн для извлечения параметров из номенклатуры болт по стандарту ГОСТ 7798-70.

Примеры:
1. Болт М16х130.52.019 ГОСТ 7798-70
2. Болт М10х60.22.6.019 ГОСТ 7798-70
3. Болт М12х80.36.6.019 ГОСТ 7798-70
4. Болт М20х160.52.6.019 ГОСТ 7798-70
5. Болт М8х40.22.6.019 ГОСТ 7798-70

Требования:
1. Используй named groups (?P<name>...)
2. Поддерживай разные варианты написания
3. Учитывай специфику стандарта ГОСТ 7798-70

Ответь ТОЛЬКО в формате JSON:
{
  "pattern": "regex with named groups",
  "params": ["param1", "param2"],
  "required": ["required1"]
}
```

---

## Ожидаемый ответ от LLM

```json
{
  "pattern": "(?i)^(?P<item_type>Болт)\\s+(?P<thread_diameter>М\\d+)\\s*х\\s*(?P<length>\\d+(?:\\.\\d+)?)\\s*\\.\\s*(?P<accuracy_class>\\d+(?:\\.\\d+)?)\\s*\\.\\s*(?P<strength_class>\\d+(?:\\.\\d+)?)\\s*\\.\\s*(?P<coating>\\d+)\\s+(?P<standard>ГОСТ\\s+\\d+-\\d+)$",
  "params": ["item_type", "thread_diameter", "length", "accuracy_class", "strength_class", "coating", "standard"],
  "required": ["item_type", "thread_diameter", "length", "strength_class", "coating", "standard"]
}
```

---

## Архитектура: Как происходит генерация маски

```
+--------------------------------------------------+
|  AutomatedParametricProcessor.process(text)      |
|                                                  |
|  Level 0: StandardExtractor                      |
|    ├─ standard = "ГОСТ 7798-70"                  |
|    └─ item_type = "болт"                         |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  Level 1: MaskDatabase                           |
|    └─ get_mask("ГОСТ 7798-70", "болт")           |
|       └─ Нет активной маски → идём дальше        |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  Level 2: LLMMaskGenerator                       |
|                                                  |
|  ┌─ generate_mask(                               |
|  │    standard="ГОСТ 7798-70",                   |
|  │    item_type="болт",                          |
|  │    examples=[...],                            |
|  │    prompt_id=None          ← НЕ передан       |
|  │  )                                            |
|  │                                               |
|  │  ВНУТРИ:                                      |
|  │  ┌─ _build_retry_config(                     |
|  │  │    prompt_id=None,                         |
|  │  │    item_type="болт"                        |
|  │  │  )                                         |
|  │  │                                            |
|  │  │  1. _resolve_prompt_by_keywords("болт")    |
|  │  │     └─ Перебирает prompts.yaml:            |
|  │  │        ├─ hardware: keywords=["болт",...]  |
|  │  │        │   └─ "болт" in keywords → MATCH   |
|  │  │        └─ Возвращает: "hardware"           |
|  │  │                                            |
|  │  │  2. _get_prompt_config("hardware")         |
|  │  │     └─ Из prompts.yaml:                    |
|  │  │        ├─ service: "mws"                   |
|  │  │        ├─ model: "qwen2.5-72b-instruct"    |
|  │  │        ├─ temperature: 0.1                 |
|  │  │        └─ system_prompt: "Вы - эксперт..." |
|  │  │                                            |
|  │  │  3. Формируем retry-конфиг:                |
|  │  │     ├─ Попытка 1: mws/qwen2.5-72b, t=0.1  |
|  │  │     └─ Попытка 2: mws/qwen2.5-72b, t=0.3  |
|  │  │                                            |
|  │  └─ Возвращает: List[Dict] конфигов           |
|  │                                               |
|  ├─ _build_prompt("ГОСТ 7798-70", "болт", ...)   |
|  │   └─ Возвращает шаблон промпта (см. выше)    |
|  │                                               |
|  └─ _try_generate(config[0])                     |
|      ├─ client.complete(prompt, model, temp, ...)│
|      ├─ Парсим JSON из ответа                    |
|      └─ Возвращает: pattern, params, required    |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  Level 3: AutoValidator                          |
|    └─ Валидируем маску на примерах из ЕСН        |
|    └─ Score >= 0.85 → activate                   |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|  Level 5: MaskDatabase.save_mask()               |
|    └─ Сохраняем валидированную маску             |
+--------------------------------------------------+
```

---

## Сравнение: process_item vs generate_mask

| Этап | `processor.py` (основная обработка) | `llm_mask_generator.py` (генерация маски) |
|------|-------------------------------------|-------------------------------------------|
| **Ключевое слово** | `auto_process()` → `_check_category_match(name, prompt_cfg)` | `_resolve_prompt_by_keywords(item_type)` |
| **Что ищем** | keywords в `item.name` | keywords в `item_type` |
| **Результат** | Список `matching_prompts` | Один `prompt_id` |
| **Откуда config** | `settings.prompts[prompt_id]` | `settings.prompts[prompt_id]` |
| **Что берём** | service, model, temperature, system_prompt | service, model, temperature, system_prompt |
| **Использование** | Запрос к LLM для разбора номенклатуры | Запрос к LLM для генерации regex-маски |

---

## Ключевые изменения в обновлённом llm_mask_generator.py

### 1. Новый метод `_resolve_prompt_by_keywords(item_type)`
- Зеркало `_check_category_match()` из `processor.py`
- Поддерживает: `regex:`, glob, обычные подстроки
- Возвращает `prompt_id` (например: `"hardware"`, `"rolledMetal"`)

### 2. Обновлённый `_build_retry_config()`
- Новый параметр `item_type: Optional[str] = None`
- Если `prompt_id` не передан — определяет автоматически через `_resolve_prompt_by_keywords()`
- Логирует весь путь: `item_type → prompt_id → service/model/temp`

### 3. Обратная совместимость
- Если `prompt_id` передан явно — используется он (старая логика)
- Если `prompt_id=None` но `item_type` передан — автоопределение (новая логика)
- Fallback цепочка: keywords → 'hardware' → config.yaml defaults → hardcoded
