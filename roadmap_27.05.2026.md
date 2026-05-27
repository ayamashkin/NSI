# Дорожная карта: Domain-based ENS Index Architecture
## Контекст: NSI / Nomenclature Matching System
## Дата: 2026-05-27

---

## 1. Проблема

Текущая архитектура использует единый плоский индекс ENS (`cache/ens_hardware.pkl`) и захардкоженный список параметров (`CHECK_KEYS`, `FIELD_NAME_MAP`, `SERVICE_FIELDS`) в `llm_mask_generator.py`.

**Что сломано:**
- Поля определяются хардкодом — при смене категории (крепеж → ЭРИ → прокат) код мертв.
- Близнецы (параметры с одинаковыми visible-значениями) определяются при генерации маски, что дорого и ненадёжно.
- Пустые/служебные/константные поля не удаляются — индекс раздут.
- Нет изоляции предметных областей: поля ЭРИ смешиваются с полями крепежа.

---

## 2. Цель

Вынести всю предметную логику (какие поля skip/retain/meta, какие близнецы, нормализация имён) на этап формирования индекса. На этапе генерации маски — только проверка однозначности оставшихся параметров.

---

## 3. Архитектура решения

```
config/domains/
├── hardware.yaml          # крепеж
├── rolled_metal.yaml      # прокат
├── eri.yaml               # ЭРИ
└── __init__.py            # DomainConfig loader

core/
├── domain_config.py       # dataclass DomainConfig
├── ens_index_builder.py   # построитель индекса (НОВЫЙ)
├── llm_mask_generator.py  # упрощённый, читает готовый индекс
└── auto_validator.py      # multi-domain поиск

cache/
├── ens_hardware.pkl       # структурированный индекс
├── ens_rolled_metal.pkl
└── ens_eri.pkl
```

---

## 4. Структура DomainConfig (YAML)

```yaml
domain: hardware
description: "Крепежные изделия"

index:
  skip_fields:          # удаляются полностью
    - "Пометка удаления"
    - "Автор"
    - ...

  meta_fields:          # сохраняются в _meta, не в regex
    - "Код"
    - "Наименование"
    - "НТД"
    - "Наименование типа"

  retain_fields:        # не видны в наименовании, но нужны для валидации
    - "Марка материала"
    - "Группа (класс) прочности"
    - "Класс (поле) допуска"
    - "Твердость"

  field_aliases:        # ENS header → canonical name
    "Наружный диаметр (диаметр вписанного круга)...": "наружный_диаметр"
    "Номинальный диаметр резьбы": "номинальный_диаметр_резьбы"

  twin_threshold: 1.0
  visible_threshold: 0.05
  max_field_name_len: 30
```

---

## 5. Структура индекса (ens_{domain}.pkl)

```python
{
  "ОСТ 1 31133-80": {
    "Болт": {
      "examples": [
        {
          "_meta": {
            "ens_code": "1000614651",
            "name": "Болт (2)-9-36-Кд.фос.окс-ОСТ 1 31133-80",
            "full_name": "...",
            "standard": "ОСТ 1 31133-80",
            "item_type": "Болт",
          },
          "наружный_диаметр": "9",
          "длина": "36",
          "покрытие": "Кд3.фос.окс",
          "исполнение": "2",
          "марка_материала": "30ХГСА",   # retain_fields → is_metadata=True
        }
      ],
      "twin_groups": [
        ["наружный_диаметр", "номинальный_диаметр_резьбы"]
      ],
      "field_meta": {
        "наружный_диаметр": {
          "original_name": "Наружный диаметр (диаметр вписанного круга)...",
          "visible_count": 20,
          "total_count": 20,
          "is_metadata": False,
        },
        "марка_материала": {
          "original_name": "Марка материала",
          "visible_count": 0,
          "is_metadata": True,
        }
      },
      "stats": {
        "total": 20,
        "visible_fields": ["наружный_диаметр", "длина", "покрытие", "исполнение"],
        "metadata_fields": ["марка_материала"],
      }
    }
  }
}
```

---

## 6. Алгоритм ens_index_builder.py

Для каждой пары (стандарт, тип):

1. **Нормализация заголовков** — aliases + auto (`пробел → _`, дубли → `_1`, `_2`)
2. **Удалить `skip_fields`** — полностью из всех записей
3. **Удалить всегда пустые** — None/""/" "/0 во всех записях стандарта
4. **Удалить константные** — одно значение во всех записях; если две колонки идентичны — удалить вторую
5. **Вычислить `visible_count`** — `_is_value_in_name(value, name)` для каждого поля каждой записи
6. **Удалить невидимые** — `visible_count == 0` и не в `retain_fields`
7. **Определить близнецов** — Union-Find по visible values (threshold=1.0)
8. **Разрешить близнецов** — заменить на canonical (первый в группе), подставить значения
9. **Сформировать `_meta` + `field_meta`** — сохранить original_name, статистику, флаги

---

## 7. Упрощение llm_mask_generator.py

**Удалить:**
- `SERVICE_FIELDS` → в домен
- `FIELD_NAME_MAP` → в домен
- `_canonicalize_field_name` → делается при индексации
- `_auto_detect_visible` → данные уже отфильтрованы
- `_detect_twin_groups` → читать из индекса
- `_resolve_twins` → близнецы разрешены

**Оставить:**
- `_filter_unambiguous` — проверка однозначности оставшихся параметров
- `_get_global_visible` — threshold на однозначных примерах
- `_format_examples`, `_format_stats` — форматирование промпта из готовых данных

**Добавить:**
- `__init__(..., domain: str)` — загружает `ens_{domain}.pkl`

---

## 8. Обновление auto_validator.py / matcher

- `_get_ens_examples(standard, item_type, domain)` — читает из `ens_{domain}.pkl`
- При сопоставлении без домена: перебирать все `ens_*.pkl`, выбирать лучший score
- Post-validation: читать `retain_fields` из записи (`_meta` + `field_meta`)

---

## 9. CLI обновления

```bash
# Построение индекса
python cli.py build-index --source _ЕНС_Крепеж.xlsx --domain hardware

# Генерация масок
python cli.py generate-masks --domain hardware --standard "ОСТ 1 31133-80"

# Сопоставление
python cli.py match --input data.csv --domain hardware
python cli.py match --input data.csv --auto-domain  # пробует все домены
```

---

## 10. Порядок реализации (приоритет)

| # | Задача | Файлы | Сложность |
|---|--------|-------|-----------|
| 1 | Создать `DomainConfig` + загрузчик | `core/domain_config.py`, `config/domains/hardware.yaml` | Низкая |
| 2 | Создать `ens_index_builder.py` | `core/ens_index_builder.py` | Средняя |
| 3 | Команда CLI `build-index` | `cli.py` | Низкая |
| 4 | Построить индекс hardware и проверить | `cache/ens_hardware.pkl`, логи | — |
| 5 | Рефактор `llm_mask_generator.py` | `core/llm_mask_generator.py` | Средняя |
| 6 | Обновить `auto_validator.py` | `core/auto_validator.py` | Средняя |
| 7 | Команды CLI `generate-masks` и `match` с `--domain` | `cli.py` | Низкая |
| 8 | Создать домен `rolled_metal` / `eri` (эксперимент) | `config/domains/rolled_metal.yaml` | Низкая |
| 9 | Тестирование: сравнение score до/после рефактора | тестовые данные | — |

---

## 11. Критерии приёмки

- [ ] `ens_hardware.pkl` содержит ~8–12 полей на стандарт (вместо 100+)
- [ ] `field_meta` содержит `original_name`, `visible_count`, `is_metadata`
- [ ] `_meta` содержит `ens_code`, `name`, `standard`, `item_type`
- [ ] `twin_groups` определены и сохранены
- [ ] `llm_mask_generator.py` не содержит `SERVICE_FIELDS` / `FIELD_NAME_MAP`
- [ ] Генерация маски для ОСТ 1 31133-80 даёт score ≥ 0.85
- [ ] Команда `--domain` работает для hardware
- [ ] Команда `--auto-domain` перебирает все домены

---

## 12. Открытые вопросы (решить в процессе)

1. **Формат индекса:** `.pkl` или SQLite? (пользователь: «как будет работать быстрее»)
2. **retain_fields:** пока 4 поля. Будем расширять методом экспериментов.
3. **Инкрементальная перестройка:** `--standard` для обновления одного стандарта без полного rebuild.
4. **Длина canonical имён:** `наружный_диаметр_диаметр_вписанного_кру...` = 85 символов. Нужно сокращение до 30?

---

*Сформировано на основе обсуждения 2026-05-27.*
*Следующий шаг: реализация `DomainConfig` + `ens_index_builder.py`.*
