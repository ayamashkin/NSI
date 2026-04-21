"""
Демонстрация работы LLMMaskGenerator с keyword-based resolution.
Показывает, как item_type мапится на prompt_id, а оттуда на service/model.
"""

import re
from typing import Dict, List, Optional


# ============================================================================
# Минимальная имитация данных из prompts.yaml (для демо без зависимостей)
# ============================================================================

DEMO_PROMPTS = {
    "hardware": {
        "name": "Крепеж и метизы - полный разбор ГОСТ",
        "keywords": ["болт", "гайка", "шуруп", "винт", "заклепка", "шпилька",
                     "гвоздь", "штифт", "хомут", "анкер", "саморез", "шайба"],
        "service": "mws",
        "model": "qwen2.5-72b-instruct",
        "temperature": 0.1,
        "system_prompt": "Вы - эксперт по стандартам ГОСТ и техническим параметрам метизной продукции."
    },
    "hardware_washer": {
        "name": "Шайбы - полный разбор ГОСТ",
        "keywords": ["шайбы"],
        "service": "mws",
        "model": "qwen2.5-72b-instruct",
        "temperature": 0.1,
        "system_prompt": "Вы - эксперт по стандартам ГОСТ и техническим параметрам метизной продукции."
    },
    "rolledMetal": {
        "name": "Прокат - полный разбор ГОСТ",
        "keywords": ["труба", "швеллер", "уголок", "балка", "профиль", "лист",
                     "плита", "рулон", "круг", "квадрат", "шестигранник", "лента",
                     "полоса", "пруток", "прутки",
                     "regex:^ст\\.сорт\\.нерж\\.|ст\\.констр\\.калибр\\.|ст\\.сорт\\.|ст\\.констр\\.|ст\\.КАЧ\\."],
        "service": "gigachat",
        "model": "GigaChat-2",
        "temperature": 0.1,
        "system_prompt": "Вы - эксперт по стандартам ГОСТ и техническим параметрам проката, труб и профиля."
    }
}


# ============================================================================
# Логика маршрутизации (копия из обновлённого llm_mask_generator.py)
# ============================================================================

def resolve_prompt_by_keywords(item_type_or_name: str, prompts_data: dict) -> Optional[str]:
    """
    Определение prompt_id по keywords.
    Точная копия логики из processor.py _check_category_match().
    """
    if not item_type_or_name:
        return 'hardware'

    name_lower = item_type_or_name.lower()

    for prompt_id, cfg in prompts_data.items():
        keywords = cfg.get('keywords', [])
        if match_keywords(name_lower, keywords):
            return prompt_id

    return 'hardware'


def match_keywords(name_lower: str, keywords: List[str]) -> bool:
    """Проверка совпадения keywords."""
    for keyword in keywords:
        keyword = keyword.strip()

        # regex: префикс
        if keyword.startswith('regex:') or keyword.startswith('re:'):
            pattern = keyword.split(':', 1)[1].strip()
            try:
                if re.search(pattern, name_lower, re.IGNORECASE):
                    return True
            except re.error:
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


# ============================================================================
# ДЕМО
# ============================================================================

def demo_resolution():
    """Демонстрация разрешения prompt_id по item_type."""

    test_cases = [
        # (item_type, ожидаемый_prompt_id, ожидаемый_service, ожидаемая_model)
        ("болт", "hardware", "mws", "qwen2.5-72b-instruct"),
        ("БОЛТ", "hardware", "mws", "qwen2.5-72b-instruct"),
        ("гайка", "hardware", "mws", "qwen2.5-72b-instruct"),
        ("шайба", "hardware", "mws", "qwen2.5-72b-instruct"),
        ("шайбы", "hardware_washer", "mws", "qwen2.5-72b-instruct"),
        ("труба", "rolledMetal", "gigachat", "GigaChat-2"),
        ("швеллер", "rolledMetal", "gigachat", "GigaChat-2"),
        ("уголок стальной", "rolledMetal", "gigachat", "GigaChat-2"),
        ("ст.сорт.нерж.12х18н10т", "rolledMetal", "gigachat", "GigaChat-2"),  # regex match
        ("ст.констр.калибр.20", "rolledMetal", "gigachat", "GigaChat-2"),      # regex match
        ("неизвестный_тип", "hardware", "mws", "qwen2.5-72b-instruct"),        # fallback
    ]

    print("=" * 80)
    print("ДЕМО: Keyword-based prompt resolution для LLMMaskGenerator")
    print("=" * 80)
    print()

    for item_type, expected_pid, expected_svc, expected_model in test_cases:
        resolved_pid = resolve_prompt_by_keywords(item_type, DEMO_PROMPTS)
        config = DEMO_PROMPTS.get(resolved_pid, {})

        status = "✓" if resolved_pid == expected_pid else "✗"

        print(f"{status} item_type='{item_type}'")
        print(f"    └─ prompt_id: '{resolved_pid}' (ожидалось: '{expected_pid}')")
        print(f"       ├─ service: {config.get('service', 'N/A')} (ожидалось: {expected_svc})")
        print(f"       ├─ model:   {config.get('model', 'N/A')} (ожидалось: {expected_model})")
        print(f"       ├─ temperature: {config.get('temperature', 'N/A')}")
        print(f"       └─ system:  {config.get('system_prompt', 'N/A')[:50]}...")
        print()

    # Показываем как выглядит итоговый промпт
    print("=" * 80)
    print("ПРИМЕР: Промпт для генерации маски (болт / ГОСТ 7798-70)")
    print("=" * 80)
    print()

    example_prompt = build_example_prompt("ГОСТ 7798-70", "болт")
    print(example_prompt)


def build_example_prompt(standard: str, item_type: str) -> str:
    """Построение примера промпта для генерации маски."""
    examples_text = """1. Болт М16х130.52.019 ГОСТ 7798-70
2. Болт М10х60.22.6.019 ГОСТ 7798-70
3. Болт М12х80.36.6.019 ГОСТ 7798-70
4. Болт М20х160.52.6.019 ГОСТ 7798-70
5. Болт М8х40.22.6.019 ГОСТ 7798-70"""

    return f"""Ты - эксперт по техническим стандартам и регулярным выражениям.

Создай regex-паттерн для извлечения параметров из номенклатуры {item_type} по стандарту {standard}.

Примеры:
{examples_text}

Требования:
1. Используй named groups (?P<name>...)
2. Поддерживай разные варианты написания
3. Учитывай специфику стандарта {standard}

Ответь ТОЛЬКО в формате JSON:
{{
  "pattern": "regex with named groups",
  "params": ["param1", "param2"],
  "required": ["required1"]
}}"""


if __name__ == "__main__":
    demo_resolution()
