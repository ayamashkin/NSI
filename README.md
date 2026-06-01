# Nomenclature Processor + Automated Parametric Search

**Nomenclature Processor** — система автоматической обработки технической номенклатуры с каскадным анализом: regex-парсинг, авто-генерация regex-масок через LLM, параметрическое сопоставление с ЕНС (включая fuzzy matching для текстовых параметров), coating auto-substitution и TF-IDF fallback.

**Возможности:**
- **Доменная архитектура ENS** — изоляция предметных областей (крепёж, прокат, ЭРИ) через конфигурируемые YAML-домены
- Автоматическая классификация номенклатуры по категориям (крепеж, ЭРИ, материалы, покупные изделия)
- Извлечение структурированных технических параметров из неструктурированных наименований
- Пакетная обработка больших объемов данных (десятки тысяч позиций)
- Интеграция с локальными LLM (OpenWebUI), облачными API (MWS Cloud GPT, GigaChat, MTS AI)
- Параметрический поиск по ЕНС с fuzzy matching и coating auto-substitution
- Настраиваемые пороги matching через `config/config.yaml`
- Анализ качества распознавания (JSON-отчеты)
- **Excel input/output** — обработка исходных Excel-файлов с сохранением структуры
- **JSON output** — экспорт результатов в JSON для интеграции с внешними системами
- **Хранение результатов в result.db** — версионирование по маскам, отслеживание изменений
- **Многопоточная обработка** — ThreadPoolExecutor с настраиваемым числом workers

---

## Содержание

- [Архитектура](#архитектура)
- [Доменная архитектура ENS](#доменная-архитектура-ens)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Быстрый старт](#быстрый-старт)
- [CLI Команды](#cli-команды)
- [Параметрическая обработка](#параметрическая-обработка-одна-строка)
- [Диагностика строки](#диагностика-строки)
- [Анализ качества](#анализ-качества)
- [Хранение результатов (result.db)](#хранение-результатов-resultdb)
- [Keyword-based маршрутизация](#keyword-based-маршрутизация)
- [Генерация regex-масок через LLM](#генерация-regex-масок-через-llm)
- [Fuzzy Matching](#fuzzy-matching)
- [Валидация покрытий](#валидация-покрытий)
- [API клиенты](#api-клиенты)
- [Новые возможности](#новые-возможности)
- [Производительность](#производительность)
- [Troubleshooting](#troubleshooting)
- [Требования](#требования)

---

## Архитектура

```
nomenclature-processor/
├── config/                    # Конфигурация системы
│   ├── __init__.py
│   ├── settings.py            # Python-классы конфигурации
│   ├── config.yaml            # Основной конфиг: API-ключи, пороги matching
│   ├── prompts.yaml           # Реестр промптов для LLM-генерации масок
│   └── domains/               # ДОМЕННЫЕ КОНФИГИ (новое)
│       ├── hardware.yaml      # Крепёж: skip/meta/retain/aliases
│       ├── rolled_metal.yaml  # Прокат
│       ├── eri.yaml           # ЭРИ
│       └── __init__.py
├── core/                      # Ядро системы
│   ├── __init__.py
│   ├── models.py              # Pydantic модели данных
│   ├── processor.py           # Основной движок обработки (Level 1-5)
│   ├── automated_processor.py # Параметрический процессор (Level 6)
│   ├── parametric_client.py   # Параметрический поиск по ЕНС
│   ├── result_database.py     # Менеджер result.db
│   ├── database.py            # SQLite manager
│   ├── quality_analyzer.py    # Анализ качества распознавания
│   ├── domain_config.py       # Загрузчик доменной конфигурации (новое)
│   ├── ens_index_builder.py   # Построитель структурированного индекса (новое)
│   ├── llm_mask_generator.py  # LLM генерация масок (domain-based)
│   ├── auto_validator.py      # Валидация масок (multi-domain)
│   ├── mask_database.py       # БД regex-масок (SQLite)
│   ├── integration.py         # Интеграция с существующей БД
│   ├── coating_indexer.py     # Индексация допустимых покрытий
│   ├── coating_llm_client.py  # LLM-клиент для запроса правил покрытий
│   ├── coating_rules.py       # Правила валидации покрытий
│   ├── coating_mapper.py      # Маппинг покрытий
│   ├── registry.py            # Реестр компонентов
│   └── standard_extractor.py  # Извлечение стандарта и типа изделия
├── api_clients/               # Клиенты для LLM-провайдеров
│   ├── __init__.py
│   ├── base.py
│   ├── openwebui.py
│   ├── mws_gpt.py
│   ├── gigachat.py
│   └── mts_ai.py
├── parsers/                   # Парсеры номенклатуры
│   ├── __init__.py
│   ├── cascade.py
│   ├── regex_parser.py
│   ├── ner_adapter.py
│   └── standard_extractor.py
├── ens/                       # Работа с ЕНС
│   ├── __init__.py
│   ├── loader.py              # Загрузчик ЕНС
│   └── indexer.py             # TF-IDF индекс
├── utils/                     # Утилиты
│   ├── __init__.py
│   ├── excel_loader.py
│   └── excel_loader_simple.py
├── scripts/                   # Служебные скрипты
│   └── auto_mapping.py        # (устарело — заменено доменными конфигами)
├── data/                      # Исходные данные (Excel)
│   ├── nomenclature.xlsx
│   ├── nomenclature1.xlsx
│   └── sample_nomenclature.xlsx
├── prompts/
│   └── templates/
│       ├── hardware.txt
│       ├── hardware_washer.txt
│       ├── rolledmetal.txt
│       └── mask_generation.txt
├── test/
│   └── test_params.py
├── default/
│   └── seed_default_masks.py
├── demo/
│   └── demo_mask_resolution.py
├── fix/
│   ├── apply_all_fixes.py
│   ├── fix_gost_7795_db.py
│   └── fix_masks_v2.py
├── cli.py                     # CLI интерфейс (Click)
├── run_batch.py               # Запуск batch-обработки из скрипта
├── requirements.txt
├── tree.py
├── mask_generation_prompt.md
├── test_prompt1.txt
├── .gitignore
├── README.md
├── README_old.md
├── ARCHITECTURE.md
├── ANALYSIS_RESULTS.md
├── CHEATSHEET.md
└── ROADMAP.md
```

### Не выгружаются в git (.gitignore)

### Каскад обработки

```
Level 0: StandardExtractor (regex)
 |- Извлечение стандарта (ГОСТ, ОСТ, ТУ, ISO, DIN, РАМ)
 |- Определение типа изделия (болт, гайка, шайба, труба, ...)
 |- item_type нормализуется в UPPERCASE

Level 0.5: Domain Index Builder (новое)
 |- Загрузка доменной конфигурации из config/domains/{domain}.yaml
 |- Нормализация заголовков ENS через field_aliases
 |- Удаление skip_fields, пустых и константных колонок
 |- Вычисление visible_count для каждого поля
 |- Определение twin_groups через Union-Find
 |- Разрешение близнецов (замена на canonical имя)
 |- Формирование структурированного индекса ens_{domain}.pkl

Level 1: MaskDatabase (SQLite + WAL)
 |- Поиск маски по (standard, item_type)
 |- Fallback: lowercase/UPPERCASE/item_type без учета регистра
 |- Fallback: маска только по стандарту
 |- Использование активных масок (auto_score >= 0.85)
 |- Автоактивация найденных неактивных масок

Level 2: LLMMaskGenerator (domain-based)
 |- Загрузка структурированного индекса ens_{domain}.pkl
 |- Чтение twin_groups и field_meta из индекса (не хардкод!)
 |- Форматирование промпта из готовых visible-полей
 |- Multi-provider: OpenWebUI, MWS, GigaChat, MTS AI
 |- Retry-стратегия с повышением temperature
 |- JSON preprocessing: исправление \s, \d, \w от LLM

Level 3: AutoValidator (multi-domain)
 |- Тест маски на примерах из структурированного индекса
 |- Multi-domain fallback: перебор всех ens_*.pkl
 |- Score = matched_required / total_required
 |- Порог активации: 0.85

Level 5: MaskDatabase.save_mask()
 |- item_type сохраняется в UPPERCASE
 |- Маски активируются сразу (is_active=True) при LLM-генерации
 |- UPSERT по pattern_hash

Level 6: ParametricENSClient
 |- Извлечение параметров через regex-маску (named groups)
 |- _remap_params: перевод сырых имен групп в ENS-имена
 |- Поиск по параметрам в индексе ЕНС (O(1) через индексы)
 |- name_exact: прямое сравнение text == ens_name
 |- Fuzzy fallback: token-based Jaccard для текстовых параметров
 |- Score: взвешенное совпадение required-параметров

Level 8: CoatingValidation
 |- Проверка покрытия по марке стали из ENS
 |- Источник правил: ENS-индекс + LLM
 |- "Бп"/"без покрытия"/пустое → skip validation
 |- Авто-замена: Кд → Н.Кд для коррозионно-стойких сталей
 |- Strict mode: reject match если покрытие не допустимо

Level 9: TF-IDF Fallback
 |- Char-ngram (2-4) TF-IDF векторизация
 |- Cosine similarity по ЕНС
 |- Всегда success=False (параметры не извлечены)
 |- ens_code сохраняется только в details как candidate
```

---

## Доменная архитектура ENS

Вся предметная логика (какие поля skip/retain/meta, какие близнецы, нормализация имён) вынесена на этап формирования индекса. На этапе генерации маски — только проверка однозначности оставшихся параметров.

### Структура доменного конфига

```yaml
# config/domains/hardware.yaml
domain: hardware
description: "Крепежные изделия"

index:
  skip_fields:          # удаляются полностью из индекса
    - "Пометка удаления"
    - "Автор"
    - ...

  meta_fields:          # сохраняются в _meta, не участвуют в regex
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

### Структура индекса (ens_{domain}.pkl)

```python
{
  "ОСТ 1 31133-80": {
    "Болт": {
      "examples": [
        {
          "_meta": {
            "ens_code": "1000614651",
            "name": "Болт (2)-9-36-Кд.фос.окс-ОСТ 1 31133-80",
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

### Алгоритм построения индекса

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

### CLI: построение индекса

```bash
# Построение индекса для домена hardware
python cli.py ens build-index _ЕНС_Крепеж.xlsx -o cache/ens_hardware.pkl -d hardware

# Просмотр структуры индекса
python cli.py ens info cache/ens_hardware.pkl

# Просмотр примеров для конкретного стандарта
python cli.py ens show cache/ens_hardware.pkl -s "ОСТ 1 31133-80" -t "Болт" -l 5
```

---

## Установка

```bash
# Клонирование репозитория
git clone
cd nomenclature-processor

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\Activate    # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### Секреты

```bash
mkdir secrets
echo "your_password" > secrets/openwebui_password.txt
echo "your_api_key" > secrets/mws_key.txt
echo "your_credentials" > secrets/gigachat_credentials.txt
echo "your_mts_ai_key" > secrets/mts_ai_key.txt
```

---

## Конфигурация

### config/config.yaml

Основной конфигурационный файл. Предметная логика (skip_fields, meta_fields, retain_fields, field_aliases) теперь живёт в доменных YAML (`config/domains/*.yaml`), а не в `config.yaml`.

```yaml
api:
  openwebui:
    base_url: "https://webui.game73.ru/api"
    username: "user@example.com"
    password_file: "secrets/openwebui_password.txt"
    default_model: "Qwen/Qwen3-14B-AWQ"
    timeout: 180

  mws:
    base_url: "https://api.gpt.mws.ru/"
    api_key_file: "secrets/mws_key.txt"
    default_model: "qwen2.5-72b-instruct"
    timeout: 120

  gigachat:
    base_url: "https://gigachat.devices.sberbank.ru/api/v1"
    api_key_file: "secrets/gigachat_credentials.txt"
    scope: "GIGACHAT_API_PERS"
    default_model: "GigaChat"
    timeout: 120

  mts_ai:
    base_url: "https://demo6-fundres.dev.mts.ai/"
    api_key_file: "secrets/mts_ai_key.txt"
    default_model: "cotype_pro_2.5"
    timeout: 120

mask_generation:
  default_service: "mts_ai"
  default_model: "cotype_pro_2.5"
  default_temperature: 0.1
  keyword_match_from_name: true
  prompt_template: "prompts/templates/mask_generation.txt"
  save_debug_prompts: true
  debug_prompts_dir: "prompts/debug"
  deduplicate_by_standard_type: true

database:
  path: "cache/results.db"

processing:
  default_workers: 4
  retry_attempts: 3

matching:
  success_threshold: 0.7
  fuzzy_threshold: 0.6
  v2_exact_threshold: 0.99
  coating_similarity_threshold: 0.8
  strict_union_keys: false
  debug_per_parameter: true

coating_rules:
  material_coating_map:
    "14Х17Н2": ["Н.Кд", "Хим.Пас", "Н.Пас"]
    "12Х18Н10Т": ["Н.Кд", "Хим.Пас", "Н.Пас"]
    "30ХГСА": ["Кд", "Цд", "Окс", "Фос", "Бп"]
  auto_substitution:
    - material_pattern: "^(14Х17Н2|12Х18Н10Т)$"
      wrong_coating: "Кд"
      correct_coating: "Н.Кд"
  similarity_threshold: 0.8
  strict_mode: true
  auto_substitution_enabled: true
```

### config/prompts.yaml

`service` и `model` опциональны — если не указаны, берутся из `mask_generation.default_service` и `mask_generation.default_model`:

```yaml
prompts:
  hardware:
    name: "Крепеж и метизы"
    keywords: ["болт", "гайка", "шуруп", "винт", "заклепка",
               "шпилька", "гвоздь", "штифт", "хомут", "анкер",
               "саморез", "шайба", "крепеж", "ГОСТ"]
    temperature: 0.1
    system_prompt: "Вы - эксперт по стандартам ГОСТ..."
    file: "prompts/templates/hardware.txt"
    category: "hardware"

  rolledMetal:
    name: "Прокат"
    keywords: ["труба", "швеллер", "уголок", "балка", "профиль",
               "лист", "плита", "рулон", "круг", "квадрат"]
    service: "gigachat"
    temperature: 0.1
    system_prompt: "Вы - эксперт по стандартам ГОСТ и прокату..."
    file: "prompts/templates/rolledmetal.txt"
    category: "rolledmetal"
```

---

## Быстрый старт

### 1. Построить индекс ЕНС (доменный)

```bash
python cli.py ens build-index "data/_ЕНС_Крепеж.xlsx" -o cache/ens_hardware.pkl -d hardware
```

При построении выводится статистика:
```
Building structured index from data/_ЕНС_Крепеж.xlsx for domain=hardware...
Index saved: cache/ens_hardware.pkl
Domain: hardware
Description: Крепежные изделия
```

### 2. Сгенерировать маски (с указанием домена)

```bash
# Один стандарт
python cli.py generate-masks -d cache/masks.db -i cache/ens_hardware.pkl --llm --domain hardware --standard "ГОСТ 7798-70"

# Все стандарты
python cli.py generate-masks -d cache/masks.db -i cache/ens_hardware.pkl --llm --domain hardware
```

### 3. Обработать файл (Excel → Excel/JSON + result.db)

```bash
# С явным указанием домена
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i cache/ens_hardware.pkl -o output/results.xlsx --workers 8 --domain hardware

# Автоматический выбор домена (перебирает все ens_*.pkl)
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i cache/ens_hardware.pkl -o output/results.xlsx --workers 8 --auto-domain

# Только успешно распознанные
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i cache/ens_hardware.pkl -o output/results.xlsx --success-only --workers 8 --domain hardware

# С debug-информацией
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i cache/ens_hardware.pkl -o output/results.xlsx --include-details --workers 8 --domain hardware
```

### 4. Анализ качества распознавания

```bash
python cli.py analyze-quality data/nomenclature.xlsx -d cache/masks.db -i cache/ens_hardware.pkl -o output/quality_report.json --domain hardware
```

### 5. Работа с result.db

```bash
# Статистика по result.db
python cli.py result-stats

# Экспорт result.db в Excel
python cli.py result-export --output output/СТИ_КумАПП_из_АСУ_НСИ.xlsx --source "data/СТИ_КумАПП_из_АСУ_НСИ.xlsx" --article-col "Артикул" --name-col "наименование"
```

### Вызовы для отладки

```bash
# Построение индекса из тестового файла
python cli.py ens build-index "data/_ЕНС_Крепеж_test.xlsx" -o models/ens_hardware_test.pkl -d hardware

# Генерация масок для тестового индекса
python cli.py generate-masks -d cache/masks.db -i models/ens_hardware_test.pkl --force --llm --domain hardware --validate -so output/mask_stats.xlsx
python cli.py generate-masks -d cache/masks.db -i models/ens_hardware_test.pkl --force --llm --domain hardware --validate --standard "ОСТ 1 31503-80" -so output/mask_stats.xlsx
python cli.py generate-masks -d cache/masks.db -i models/ens_hardware_test.pkl --responses-dir prompts/answers --validate --domain hardware

# Batch-обработка
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/ens_hardware_test.pkl --workers 1 -o output/nomenclature.xlsx --domain hardware --no-cache
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/ens_hardware_test.pkl --workers 1 -o output/nomenclature.json --domain hardware  --no-cache


python cli.py batch data/nomenclature2.xlsx -d cache/masks.db -i models/ens_hardware_test.pkl --workers 1 -o output/nomenclature.json --domain hardware  --no-cache

# Анализ качества
python cli.py analyze-quality data/nomenclature.xlsx -d cache/masks.db -i models/ens_hardware_test.pkl --workers 4 -o output/quality.xlsx -j output/quality.json --domain hardware

# Диагностика отдельной строки
python cli.py diagnose "Болт (2)-8-26-Кд-ОСТ 1 31133-80" --db cache/masks.db --ens-index cache/ens_hardware.pkl --domain hardware
```



### Prod

```bash
# Построение индекса из production-файла
python cli.py ens build-index "data/_ЕНС_Крепеж_05.05.2026.xlsx" -o cache/ens_hardware.pkl -d hardware

# Генерация масок (дозаполнение — путь берётся из доменного конфига)
python cli.py generate-masks -d cache/masks.db --domain hardware --llm --validate -so output/mask_stats.xlsx

# Генерация масок (перегенерация)
python cli.py generate-masks -d cache/masks.db --domain hardware --llm --force --validate -so output/mask_stats.xlsx

# Генерация масок (перегенерация) без валидации
python cli.py generate-masks -d cache/masks.db --domain hardware --llm --force -so output/mask_stats.xlsx

# Batch-обработка (Excel → Excel + result.db)
python cli.py batch data/nomenclature1.xlsx -d cache/masks.db --domain hardware --workers 3 -o output/nomenclature1.xlsx

python cli.py batch data/СТИ_КумАПП_из_АСУ_НСИ.xlsx -d cache/masks.db --domain hardware --workers 2 -o output/СТИ_КумАПП_из_АСУ_НСИ.xlsx

# Batch-обработка (Excel → JSON + result.db)
python cli.py batch data/nomenclature1.xlsx -d cache/masks.db --domain hardware --workers 4 -o output/nomenclature1.json

# Принудительная batch-переобработка (игнорирует кэш result.db)
python cli.py batch data/nomenclature1.xlsx -d cache/masks.db --domain hardware --workers 4 -o output/results.xlsx --force

# Анализ качества
python cli.py analyze-quality data/nomenclature1.xlsx -d cache/masks.db --domain hardware -o output/quality.xlsx -j output/quality.json
```



## CLI Команды

| Команда | Описание | Параметры |
|---------|----------|-----------|
| `ens build-index` | Построить структурированный доменный индекс из Excel | `excel_file`, `--output`, `--domain` |
| `ens info` | Информация об индексе (стандартов, типов, примеров) | `pkl_file` |
| `ens show` | Просмотр примеров из индекса | `pkl_file`, `--standard`, `--item-type`, `--limit` |
| `generate-masks` | Генерация regex-масок через LLM | `--db`, `--ens-index`, `--domain`, `--standard`, `--item-type`, `--llm`, `--validate`, `--min-score`, `--limit`, `--force`, `--stats-output` |
| `process-parametric` | Обработка одной строки | `text`, `--db`, `--ens-index`, `--domain`, `--llm` |
| `batch` | Пакетная обработка Excel | `input_file`, `--db`, `--ens-index`, `--domain`, `--auto-domain`, `--output`, `--llm`, `--validate`, `--success-only`, `--include-details`, `--workers`, `--result-db` |
| `analyze-quality` | Анализ качества распознавания | `input_file`, `--db`, `--ens-index`, `--domain`, `--output`, `--json`, `--llm` |
| `diagnose` | Диагностика обработки строки | `text`, `--db`, `--ens-index`, `--domain`, `--auto-domain`, `--llm` |
| `cleanup` | Очистка неактивных масок | `--db`, `--threshold` |
| `result-stats` | Статистика result.db | `--db` |
| `result-export` | Экспорт result.db в Excel | `--output`, `--source`, `--article-col`, `--name-col` |
| `prompts` | Список промптов | — |
| `process` | Обработка через LLM (legacy) | `input_file`, `--prompt`, `--auto`, `--workers`, `--force` |
| `export` | Экспорт результатов в JSON | `--output`, `--structure`, `--prompt`, `--status`, `--include-raw`, `--include-full-request` |
| `stats` | Статистика БД | — |
| `errors` | Показать ошибки | `--limit`, `--prompt` |
| `detect` | Определить категорию | `text` |
| `models` | Список моделей API | `--api` |

### Параметры batch

```bash
python cli.py batch data/nomenclature.xlsx   -d cache/masks.db   -i cache/ens_hardware.pkl   -o output/results.xlsx   --llm   --validate   --success-only   --include-details   --workers 8   --domain hardware   --result-db cache/result.db
```

**Ключевые параметры:**
- `--db` — путь к SQLite БД масок (default: `cache/masks.db`)
- `--ens-index` — путь к индексу ЕНС (structured .pkl)
- `--domain` — домен ENS (`hardware`, `rolled_metal`, `eri`)
- `--auto-domain` — автоматический выбор домена (перебирает все `ens_*.pkl`)
- `--llm` — использовать LLM для генерации недостающих масок
- `--validate` — валидировать результаты
- `--success-only` — включать только успешно распознанные позиции
- `--include-details` — включать debug-информацию
- `--workers` — число параллельных потоков (default: 4)
- `--result-db` — путь к SQLite БД результатов (default: `cache/result.db`)
- `--coating-map` — путь к Excel с картой покрытий

---

## Параметрическая обработка (одна строка)

```bash
python cli.py process-parametric "Болт (2)-8-26-Кд-ОСТ 1 31133-80"   --db cache/masks.db   --ens-index cache/ens_hardware.pkl   --domain hardware
```

**Результат:**
```
📄 Текст: Болт (2)-8-26-Кд-ОСТ 1 31133-80
🏷️ Уровень: parametric
✅ Успех: True
🎯 Confidence: 0.95
⏱️ Время: 12.34 мс
📋 Параметры:
  тип_изделия: Болт
  исполнение: 2
  номинальный_диаметр_резьбы: 8
  длина: 26
  покрытие: Кд
  нтд_1: ОСТ 1 31133-80
🔗 ЕНС совпадение:
  Код: 1000614651
```

---

## Диагностика строки

```bash
python cli.py diagnose "Болт (2)-8-26-Кд-ОСТ 1 31133-80"   --db cache/masks.db   --ens-index cache/ens_hardware.pkl   --domain hardware
```

Выводит подробную информацию:
- Level 0: извлечение стандарта и типа
- Level 1: поиск маски в БД
- Regex match: совпадение паттерна
- Level 6: полный результат обработки

---

## Анализ качества

```bash
python cli.py analyze-quality data/nomenclature.xlsx   -d cache/masks.db   -i cache/ens_hardware.pkl   -o output/quality_report.json   --domain hardware
```

**Метрики:**
- Успешно распознанные / Всего
- Средняя уверенность
- Распределение по уровням обработки
- Распределение по типам изделий
- Статистика по покрытиям
- Примеры ошибок

---

## Хранение результатов (result.db)

Система хранит результаты обработки в SQLite БД `cache/result.db`.

### Структура таблицы results

```sql
CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    article TEXT,
    item_type TEXT,
    standard TEXT,
    ens_code TEXT,
    ens_name TEXT,
    success INTEGER,
    confidence REAL,
    params TEXT,
    ens_params TEXT,
    match_type TEXT,
    match_type_ru TEXT,
    coating_substitution TEXT,
    fuzzy_mismatched_params TEXT,
    mask_id INTEGER,
    mask_pattern TEXT,
    details TEXT,
    processing_time_ms REAL,
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(name, mask_id)
);
```

### Версионирование

- `updated_at` — время последнего обновления
- `mask_id` — ID маски, по которой был получен результат
- При изменении маски старые результаты сохраняются, создаются новые записи

### Команды

```bash
# Статистика
python cli.py result-stats

# Экспорт в Excel
python cli.py result-export   --output output/СТИ_КумАПП_из_АСУ_НСИ.xlsx   --source "data/СТИ_КумАПП_из_АСУ_НСИ.xlsx"   --article-col "Артикул"   --name-col "наименование"
```

---

## Keyword-based маршрутизация

Система определяет категорию номенклатуры по ключевым словам из `config/prompts.yaml` и выбирает соответствующий промпт и LLM-провайдер.

**Пример:**
```yaml
hardware:
  keywords: ["болт", "гайка", "шуруп", "винт", "заклепка", "шпилька"]
  service: "mts_ai"          # или "openwebui", "mws", "gigachat"
  temperature: 0.1
```

При обработке строки "Болт М12-50-6г-ГОСТ 7798-70" система:
1. Находит ключевое слово "болт" → категория hardware
2. Использует промпт `hardware.txt`
3. Вызывает LLM через провайдер `mts_ai`

**Fallback:** если ключевые слова не найдены — используется `default_service` и `default_model` из `config.yaml`.

---

## Генерация regex-масок через LLM

### Упрощённая архитектура (domain-based)

```
┌─────────────────────────────────────────┐
│  DomainConfig.load("hardware")          │
│  → skip_fields, meta_fields,            │
│    retain_fields, field_aliases         │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  ENSIndexBuilder                        │
│  → ens_hardware.pkl                     │
│  → examples, twin_groups, field_meta  │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  LLMMaskGenerator(domain="hardware")    │
│  → читает twin_groups из индекса        │
│  → _filter_unambiguous()                │
│  → _get_global_visible()                │
│  → форматирование промпта               │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  LLM (OpenWebUI / MWS / GigaChat /     │
│  MTS AI)                                │
│  → JSON с pattern, params, required     │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  AutoValidator                          │
│  → тест на примерах из индекса        │
│  → score >= 0.85 → activate             │
└─────────────────────────────────────────┘
```

### Команды генерации

```bash
# Один стандарт с валидацией
python cli.py generate-masks   -d cache/masks.db   -i cache/ens_hardware.pkl   --llm   --domain hardware   --standard "ГОСТ 7798-70"   --item-type "Болт"   --validate   --min-score 0.85

# Массовая генерация (все стандарты с >=10 примерами)
python cli.py generate-masks   -d cache/masks.db   -i cache/ens_hardware.pkl   --llm   --domain hardware   --validate   --min-score 0.85   --stats-output output/mask_stats.xlsx

# Принудительная перегенерация (даже для активных масок)
python cli.py generate-masks   -d cache/masks.db   -i cache/ens_hardware.pkl   --llm   --domain hardware   --force

# Ограниченная генерация (только первые N стандартов)
python cli.py generate-masks   -d cache/masks.db   -i cache/ens_hardware.pkl   --llm   --domain hardware   --limit 5
```

### Отладка промптов

При `save_debug_prompts: true` в `config.yaml`:
```
prompts/debug/
├── Болт_ОСТ 1 31133-80.txt           # Промпт
├── Болт_ОСТ 1 31133-80_a1.txt        # Ответ LLM (attempt 1)
├── Болт_ОСТ 1 31133-80_a2.txt        # Ответ LLM (attempt 2)
└── ...
```

### Пример сгенерированной маски

**Стандарт:** ГОСТ 7798-70 / Болт

```json
{
  "pattern": "^(?P<тип_изделия>Болт)[-\s]+M(?P<номинальный_диаметр_резьбы>\d+)(?:[xXхХ×](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?[-\s]+(?P<класс_поле_допуска>\d+[a-zA-Zа-яА-Я]+)[xXхХ×](?P<длина>\d+(?:[.,]\d+)?)[-\s]+(?P<покрытие>[\w.]+)?[-\s]*(?P<нтд_1>ГОСТ\s*7798-70)$",
  "params": ["тип_изделия", "номинальный_диаметр_резьбы", "покрытие", "нтд_1", "шаг_резьбы", "класс_поле_допуска", "длина"],
  "required": ["тип_изделия", "номинальный_диаметр_резьбы", "покрытие", "нтд_1", "класс_поле_допуска", "длина"]
}
```

**Параметры:**
- `тип_изделия` — обязательный, первый в паттерне
- `номинальный_диаметр_резьбы` — обязательный (виден в >=85% примеров)
- `шаг_резьбы` — опциональный (`(?:...)?`)
- `класс_поле_допуска` — обязательный
- `длина` — обязательный
- `покрытие` — обязательный
- `нтд_1` — обязательный (метаданные, не извлекается из строки напрямую)

---

## Fuzzy Matching

### V2: Token-based Jaccard

```python
# Текстовые параметры (покрытие, марка материала)
text_similarity = len(set(text_tokens) & set(ens_tokens)) / len(set(text_tokens) | set(ens_tokens))

# Числовые параметры (длина, диаметр)
numeric_similarity = 1.0 if abs(text_val - ens_val) < tolerance else 0.0
```

### Параметры fuzzy matching

```yaml
matching:
  fuzzy_threshold: 0.6           # Минимальная схожесть для fuzzy
  v2_exact_threshold: 0.99       # Порог для exact matching
  strict_union_keys: false       # Строгое сравнение всех ключей
  debug_per_parameter: true      # Логирование по каждому параметру
  fuzzy_params_comparison: true  # Включить fuzzy для текстовых параметров
  numeric_field_weight: 5.0      # Вес числовых параметров
  text_field_weight: 2.0         # Вес текстовых параметров
  default_field_weight: 1.0      # Вес по умолчанию
  length_tolerance: 1.0            # Допуск для длины (мм)
  numeric_tolerance: 0.01        # Допуск для числовых параметров
```

### Пример fuzzy matching

```
Текст:    "Болт М12-50-6г-ГОСТ 7798-70"
ENS:      "Болт М12-50-6г-ГОСТ 7798-70"

Exact match: 100% → success=True, confidence=1.0

---

Текст:    "Болт М12-50-6г-ГОСТ 7798-70"
ENS:      "Болт М12-50-6г.8-ГОСТ 7798-70"

Fuzzy match: покрытие "6г" vs "6г.8" → similarity=0.67
             длина "50" vs "50" → exact
             → success=True, confidence=0.85
```

---

## Валидация покрытий

### Правила (config.yaml)

```yaml
coating_rules:
  material_coating_map:
    "14Х17Н2": ["Н.Кд", "Хим.Пас", "Н.Пас", "Пас"]
    "12Х18Н10Т": ["Н.Кд", "Хим.Пас", "Н.Пас"]
    "30ХГСА": ["Кд", "Цд", "Окс", "Фос", "Окс.Фос", "Фос.Окс", "Неп", "Пас", "Бп"]

  auto_substitution:
    - material_pattern: "^(14Х17Н2|12Х18Н10Т|08Х18Н10Т)$"
      wrong_coating: "Кд"
      correct_coating: "Н.Кд"
      note: "Для коррозионно-стойкой стали Кд заменяется на Н.Кд"

  similarity_threshold: 0.8
  strict_mode: true
  auto_substitution_enabled: true
```

### Алгоритм

1. Извлечь покрытие из номенклатуры (regex-маска)
2. Найти марку материала в ЕНС
3. Проверить допустимость покрытия для данной марки
4. Если покрытие не допустимо → reject match (strict mode)
5. Если покрытие подходит с auto-substitution → apply substitution

### Пример

```
Текст: "Болт М12-50-Кд-ГОСТ 7798-70"
ENS:   "Болт М12-50-6г-ГОСТ 7798-70", марка="14Х17Н2"

Покрытие "Кд" не допустимо для "14Х17Н2" (требуется "Н.Кд")
→ Reject match (strict mode)
→ Coating substitution: Кд → Н.Кд
→ Повторный поиск с "Н.Кд"
```

---

## API клиенты

### OpenWebUI (локальный LLM)

```python
from api_clients.openwebui import OpenWebUIClient

client = OpenWebUIClient(
    base_url="https://webui.game73.ru/api",
    api_key="your_api_key",  # или username/password для JWT
    username="user@example.com",
    password="your_password"
)

messages = [{"role": "user", "content": "Проанализируй номенклатуру..."}]
response = client.chat(messages=messages, model="Qwen/Qwen3-14B-AWQ", temperature=0.1)
```

### MWS Cloud GPT

```python
from api_clients.mws_gpt import MWSGPTClient

client = MWSGPTClient(
    base_url="https://api.gpt.mws.ru/",
    api_key="your_api_key"
)

response = client.chat(messages=[{"role": "user", "content": "..."}])
```

### GigaChat (Sber)

```python
from api_clients.gigachat import GigaChatClient

client = GigaChatClient(
    base_url="https://gigachat.devices.sberbank.ru/api/v1",
    api_key="your_credentials",
    scope="GIGACHAT_API_PERS",
    verify_ssl=False
)

response = client.chat(messages=[{"role": "user", "content": "..."}])
```

### MTS AI

```python
from api_clients.mts_ai import MTSAIClient

client = MTSAIClient(
    base_url="https://demo6-fundres.dev.mts.ai/",
    api_key="your_api_key"
)

response = client.chat(messages=[{"role": "user", "content": "..."}])
```

---

## Новые возможности

### Domain-based ENS Index

- Изоляция предметных областей через `config/domains/*.yaml`
- Структурированный индекс `ens_{domain}.pkl` с `_meta`, `field_meta`, `twin_groups`, `stats`
- Уменьшение числа полей на стандарт с 100+ до 8–12
- Multi-domain fallback при сопоставлении

### LLM Mask Generation (V2)

- Чтение twin_groups и field_meta из индекса (не хардкод)
- Проверка однозначности параметров через `_filter_unambiguous`
- Multi-provider с retry-стратегией
- JSON preprocessing: исправление `\s`, `\d`, `\w` от LLM

### AutoValidator (V2)

- Поддержка структурированного индекса
- Multi-domain поиск по всем `ens_*.pkl`
- Legacy fallback для плоских индексов

### Coating Auto-Substitution

- Автоматическая замена покрытий по марке стали
- Strict mode: reject match если покрытие не допустимо
- Настраиваемые правила в `config.yaml`

### Result Database

- Версионирование результатов по маскам
- SQLite БД `cache/result.db`
- Команды `result-stats` и `result-export`

---

## Производительность

### Batch-обработка

| Файл | Позиций | Workers | Время | Результат |
|------|---------|---------|-------|-----------|
| nomenclature.xlsx | 10 000 | 4 | ~15 мин | 85% успешно |
| nomenclature.xlsx | 10 000 | 8 | ~8 мин | 85% успешно |
| СТИ_КумАПП.xlsx | 5 000 | 4 | ~7 мин | 92% успешно |

### Кэширование

- **MaskDatabase**: SQLite с WAL mode, кэширование в памяти
- **ResultDatabase**: UPSERT по `(name, mask_id)`, не пересчитывает при неизменной маске
- **TF-IDF**: Pickle-модели, загружаются один раз

### Оптимизация

```bash
# Увеличить workers для больших файлов
python cli.py batch data/nomenclature.xlsx --workers 16

# Использовать только кэш (без LLM)
python cli.py batch data/nomenclature.xlsx --no-llm

# Ограничить число стандартов для отладки
python cli.py generate-masks --limit 5
```

---

## Troubleshooting

### Ошибка: "No module named 'api_clients'"

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# или
python -m cli batch ...
```

### Ошибка: "sqlite3.OperationalError: database is locked"

```bash
# Удалить WAL-файлы
rm cache/masks.db-wal cache/masks.db-shm
# или
rm cache/result.db-wal cache/result.db-shm
```

### Ошибка: "API key not found"

```bash
# Проверить наличие файлов
ls secrets/
# Должны быть:
#   openwebui_password.txt
#   mws_key.txt
#   gigachat_credentials.txt
#   mts_ai_key.txt
```

### Ошибка: "Failed to generate mask"

```bash
# Проверить доступность LLM
python cli.py models --api mts_ai

# Включить debug-промпты
# В config.yaml: save_debug_prompts: true
# Смотреть в prompts/debug/
```

### Ошибка: "ENS index not found"

```bash
# Построить индекс
python cli.py ens build-index "data/_ЕНС_Крепеж.xlsx" -o cache/ens_hardware.pkl -d hardware

# Проверить
python cli.py ens info cache/ens_hardware.pkl
```

### Ошибка: "Invalid regex pattern"

```bash
# Диагностика строки
python cli.py diagnose "Болт М12-50-6г-ГОСТ 7798-70"   --db cache/masks.db   --ens-index cache/ens_hardware.pkl   --domain hardware
```

### Ошибка: "Нет колонки с наименованием"

```bash
# Проверить названия колонок
python -c "import pandas as pd; df = pd.read_excel('data/nomenclature.xlsx'); print(list(df.columns))"

# Переименовать колонку
# В Excel: колонка должна содержать слово "Наименование" или "Номенклатура"
```

---

### Зависимости

```
pandas>=1.3.0
openpyxl>=3.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
click>=8.0.0
pyyaml>=5.4.0
requests>=2.25.0
tqdm>=4.60.0
pydantic>=1.8.0
```

