# Nomenclature Processor + Automated Parametric Search

**Nomenclature Processor** — система автоматической обработки технической номенклатуры с каскадным анализом: regex-парсинг, авто-генерация regex-масок через LLM, параметрическое сопоставление с ЕНС (включая fuzzy matching для текстовых параметров), coating auto-substitution и TF-IDF fallback.

**Возможности:**
- Автоматическая классификация номенклатуры по категориям (крепеж, ЭРИ, материалы, покупные изделия)
- Извлечение структурированных технических параметров из неструктурированных наименований
- Пакетная обработка больших объемов данных (десятки тысяч позиций)
- Интеграция с локальными LLM (OpenWebUI), облачными API (MWS Cloud GPT, GigaChat, MTS AI)
- Параметрический поиск по ЕСН с fuzzy matching и coating auto-substitution
- Настраиваемые пороги matching через `config/config.yaml`
- Анализ качества распознавания (JSON-отчеты)
- **Excel input/output** — обработка исходных Excel-файлов с сохранением структуры
- **Хранение результатов в result.db** — версионирование по маскам, отслеживание изменений
- **Многопоточная обработка** — ProcessPoolExecutor с настраиваемым числом workers

---

## Содержание

- [Архитектура](#архитектура)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Matching Configuration](#configconfigyaml--matching-configuration)
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
  - [MTS AI](#mts-ai)
- [Новые возможности](#новые-возможности)
- [Производительность](#производительность)
- [Troubleshooting](#troubleshooting)
- [Требования](#требования)

---

## Архитектура

```
nomenclature-processor/
├── config/                          # Конфигурация системы
│   ├── __init__.py
│   ├── settings.py                  # Python-классы конфигурации (Settings, PromptConfig, APIConfig, MatchingConfig и др.)
│   ├── config.yaml                # Основной конфиг: API-ключи, пороги matching, coating_rules
│   ├── prompts.yaml               # Реестр промптов для LLM-генерации масок (service/model опциональны)
│   └── ens_column_mapping.yaml    # Маппинг колонок ЕНС при импорте
├── core/                            # Ядро системы
│   ├── __init__.py
│   ├── models.py                  # Pydantic модели данных (MaskRecord, ProcessingResult и др.)
│   ├── processor.py               # Основной движок обработки (Level 1-5)
│   ├── automated_processor.py     # Параметрический процессор (Level 6) — fuzzy matching, coating auto-substitution
│   ├── parametric_client.py       # Параметрический поиск по ЕНС с индексацией и кэшированием
│   ├── result_database.py         # Менеджер result.db — upsert с отслеживанием версий масок
│   ├── database.py                # SQLite manager — кэширование результатов и upsert
│   ├── quality_analyzer.py        # Анализ качества распознавания (JSON-отчеты)
│   ├── llm_mask_generator.py      # LLM генерация regex-масок через промпты
│   ├── auto_validator.py          # Автоматическая валидация сгенерированных масок
│   ├── mask_database.py           # БД regex-масок (SQLite) — CRUD, поиск, версионирование
│   ├── integration.py             # Интеграция с существующей БД номенклатуры
│   ├── coating_indexer.py         # Индексация допустимых покрытий (марка → покрытия) из ЕНС
│   ├── coating_llm_client.py      # LLM-клиент для запроса правил покрытий
│   ├── coating_rules.py           # Правила валидации покрытий по маркам материалов
│   ├── coating_mapper.py          # Маппинг покрытий между текстом и ЕНС
│   ├── registry.py                # Реестр компонентов системы
│   └── standard_extractor.py      # Извлечение стандарта (ГОСТ/ОСТ/ТУ) и типа изделия
├── api_clients/                     # Клиенты для LLM-провайдеров
│   ├── __init__.py
│   ├── base.py                    # Абстрактный базовый класс для API-клиентов
│   ├── openwebui.py               # OpenWebUI API (JWT/API key, локальные LLM)
│   ├── mws_gpt.py                 # MWS Cloud GPT API
│   ├── gigachat.py                # GigaChat API (Sber, OAuth2)
│   └── mts_ai.py                  # MTS AI API (OpenAI-compatible, модель cotype_pro_2.5)
├── parsers/                         # Парсеры номенклатуры
│   ├── __init__.py
│   ├── cascade.py                 # Каскадный парсер: regex → NER → LLM fallback
│   ├── regex_parser.py            # Regex-уровень парсинга (именованные группы)
│   ├── ner_adapter.py             # NER адаптер (именованные сущности)
│   └── standard_extractor.py      # Извлечение ГОСТ/ОСТ/ТУ и типа изделия
├── ens/                             # Работа с Единой Номенклатурной Системой (ЕНС)
│   ├── __init__.py
│   ├── loader.py                  # Загрузчик ЕНС с авто-маппингом колонок
│   └── indexer.py                 # TF-IDF индекс для семантического поиска по ЕНС
├── utils/                           # Утилиты
│   ├── __init__.py
│   ├── excel_loader.py            # Загрузка Excel через pandas/openpyxl
│   └── excel_loader_simple.py     # Загрузка Excel только через openpyxl (без pandas)
├── scripts/                         # Служебные скрипты
│   └── auto_mapping.py            # Авто-генерация ens_column_mapping.yaml из Excel
├── data/                            # Исходные данные (Excel)
│   ├── nomenclature.xlsx          # Тестовая номенклатура
│   ├── nomenclature1.xlsx         # Расширенная тестовая номенклатура
│   └── sample_nomenclature.xlsx   # Пример данных для демо
├── prompts/
│   └── templates/                   # Шаблоны промптов для LLM
│       ├── hardware.txt           # Промпт для крепежа и метизов
│       ├── hardware_washer.txt    # Промпт для шайб
│       ├── rolledmetal.txt        # Промпт для проката
│       └── mask_generation.txt    # Базовый шаблон генерации масок
├── test/                            # Тесты
│   └── test_params.py             # Тесты извлечения параметров
├── default/                         # Данные по умолчанию
│   └── seed_default_masks.py      # Скрипт первоначального заполнения БД масок
├── demo/                            # Демонстрационные скрипты
│   └── demo_mask_resolution.py    # Демо разрешения масок (разбор конфликтов)
├── fix/                             # Скрипты исправления данных
│   ├── apply_all_fixes.py         # Применение всех фиксов
│   ├── fix_gost_7795_db.py        # Исправление ГОСТ 7795 в БД
│   └── fix_masks_v2.py            # Исправление масок v2
├── cli.py                           # CLI интерфейс (Click) — точка входа
├── run_batch.py                     # Запуск batch-обработки из скрипта
├── requirements.txt                 # Зависимости Python
├── tree.py                          # Генерация дерева файлов проекта
├── mask_generation_prompt.md        # Документация по промптам для масок
├── test_prompt1.txt                 # Тестовый промпт
├── .gitignore                       # Исключаемые из git файлы и директории
├── README.md                        # Этот файл
├── README_old.md                    # Предыдущая версия README (устаревшая)
├── ARCHITECTURE.md                  # Подробное описание архитектуры системы
├── ANALYSIS_RESULTS.md              # Результаты анализа качества распознавания
├── CHEATSHEET.md                    # Шпаргалка по командам и конфигурации
└── ROADMAP.md                       # Дорожная карта развития проекта
```

### Не выгружаются в git (`.gitignore`)

Следующие директории и файлы создаются автоматически при работе и не хранятся в репозитории:

```
# Данные и кэш
secrets/                           # API ключи и пароли (ключевые файлы аутентификации)
logs/                              # Логи процессора (ротируются, большой объем)
*.log                              # Отдельные лог-файлы
output/                            # JSON результаты обработки
results/                           # Резервные копии результатов
models/                            # Pickle модели TF-IDF и индексы ЕНС (большие файлы)
cache/                             # SQLite кэш: маски, результаты, статистика
*.db                               # Файлы SQLite баз данных
result.db                          # БД результатов сопоставления (создается автоматически)
prompts/debug/                     # Отладочные промпты (создаются автоматически при генерации)

# Python
.venv/                             # Виртуальное окружение
venv/                              # Альтернативное виртуальное окружение
.env                               # Переменные окружения
__pycache__/                       # Кэш скомпилированных Python-модулей
*.pyc                              # Скомпилированные Python-файлы
*.pyo                              # Оптимизированные Python-файлы
*.egg-info/                        # Метаданные установленных пакетов

# IDE
.idea/                             # Файлы конфигурации PyCharm/IntelliJ
```

### Каскад обработки

```
Level 0: StandardExtractor (regex)
  |- Извлечение стандарта (ГОСТ, ОСТ, ТУ, ISO, DIN, РАМ)
  |- Определение типа изделия (болт, гайка, шайба, труба, ...)
  |- item_type нормализуется в UPPERCASE (БОЛТ, ВИНТ, ШАЙБА)
  |- Keyword-based routing: тип -> prompt_id -> service/model

Level 1: MaskDatabase (SQLite + WAL)
  |- Поиск маски по (standard, item_type)
  |- Fallback: lowercase/UPPERCASE/item_type без учета регистра
  |- Fallback: маска только по стандарту (любой item_type)
  |- Использование активных масок (auto_score >= 0.85)
  |- Автоактивация найденных неактивных масок

Level 2: LLMMaskGenerator
  |- Автоопределение prompt_id по keywords из item_type/name (5 источников)
  |- Загрузка конфига из prompts.yaml (service, model, temperature, system_prompt)
  |- Fallback: раздел mask_generation в config.yaml
  |- Multi-provider: OpenWebUI, MWS, GigaChat, MTS AI
  |- Retry-стратегия с повышением temperature
  |- JSON preprocessing: исправление \s, \d, \w от LLM
  |- Ограничение имен групп: max 30 символов
  |- skip_fields из ens_column_mapping.yaml (исключение служебных полей)

Level 3: AutoValidator
  |- Тест маски на примерах из ЕСН
  |- Score = matched_required / total_required
  |- Порог активации: 0.85 (настраивается)

Level 5: MaskDatabase.save_mask()
  |- item_type сохраняется в UPPERCASE
  |- Маски активируются сразу (is_active=True) при LLM-генерации
  |- UPSERT по pattern_hash

Level 6: ParametricENSClient
  |- Извлечение параметров через regex-маску (named groups)
  |- _remap_params: перевод сырых имен групп в ENS-имена
  |- Поиск по параметрам в индексе ЕНС (O(1) через индексы по стандарту/типу)
  |- name_exact: прямое сравнение text == ens_name
  |- Fuzzy fallback: token-based Jaccard для текстовых параметров
     (покрытие, материал) — учитывает перестановку токенов
  |- Score: взвешенное совпадение required-параметров

Level 8: CoatingValidation
  |- Проверка покрытия по марке стали из ENS
  |- Источник правил: ENS-индекс + LLM (coating_indexer.py, coating_llm_client.py)
  |- "Бп"/"без покрытия"/пустое → skip validation
  |- Авто-замена: Кд → Н.Кд для коррозионно-стойких сталей
  |- Strict mode: reject match если покрытие не допустимо

Level 9: TF-IDF Fallback
  |- Char-ngram (2-4) TF-IDF векторизация
  |- Cosine similarity по ЕСН
  |- Всегда success=False (параметры не извлечены)
  |- ens_code сохраняется только в details как candidate
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
.venv\Scripts\activate   # Windows

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
    # MTS AI — OpenAI-compatible API для модели cotype_pro_2.5
    # Веб-интерфейс: https://cotype6.dev.mts.ai/
    # Документация: https://demo6-fundres.dev.mts.ai/
    base_url: "https://demo6-fundres.dev.mts.ai/"
    api_key_file: "secrets/mts_ai_key.txt"
    default_model: "cotype_pro_2.5"
    timeout: 120

mask_generation:
  default_service: "mws"
  default_model: "qwen2.5-72b-instruct"
  default_temperature: 0.1
  keyword_match_from_name: true
  prompt_template: "prompts/templates/mask_generation.txt"
  save_debug_prompts: true
  debug_prompts_dir: "prompts/debug"

database:
  path: "cache/results.db"

processing:
  default_workers: 4
  retry_attempts: 3

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
    # service: "mws"      # можно не указывать — берется из mask_generation
    # model: "qwen2.5-72b-instruct"  # можно не указывать — берется из mask_generation
    temperature: 0.1
    system_prompt: "Вы - эксперт по стандартам ГОСТ..."
    file: "prompts/templates/hardware.txt"
    category: "hardware"

  rolledMetal:
    name: "Прокат"
    keywords: ["труба", "швеллер", "уголок", "балка", "профиль",
               "лист", "плита", "рулон", "круг", "квадрат",
               "regex:^ст\.сорт\.нерж\.|ст\.констр\.калибр\."]
    service: "gigachat"  # явно указан — используется gigachat вместо default
    temperature: 0.1
    system_prompt: "Вы - эксперт по стандартам ГОСТ и прокату..."
    file: "prompts/templates/rolledmetal.txt"
    category: "rolledmetal"
```

### config/ens_column_mapping.yaml

```yaml
categories:
  hardware:
    'полное наименование': 'полное_наименование'
    'наименование типа': 'тип_изделия'
    'код': 'код'
    'нтд': 'стандарт'
    'марка материала': 'материал'
    'покрытие': 'покрытие'

skip_fields:
  - 'код'
  - 'mdm_key'
  - 'единицы_измерения'
  - 'наименование_типа.1'
  - 'полное_наименование'
  - 'наименование'
  - 'нтд'
  - 'тип'
  - 'наименование_типа'

auto_mapping_patterns:
  'диаметр': 'диаметр'
  'длина': 'длина'
  'исполнение': 'исполнение'
  'покрытие': 'покрытие'
  'класс прочности': 'класс_прочности'
  'марка материала': 'марка_материала'
```

### config/config.yaml — Matching Configuration

```yaml
matching:
  # Порог для считать match успешным (0.0..1.0)
  # 1.0 = только идеальное совпадение всех параметров
  # 0.7 = fuzzy match с совпадением ключевых параметров
  success_threshold: 0.7

  # Порог fuzzy matching кандидатов из ЕНС
  # Fuzzy активируется когда точный parametric match не дал результата
  fuzzy_threshold: 0.6

  # Порог V2 exact matching (params vs ens_params_mask)
  v2_exact_threshold: 0.99

  # Порог similarity для покрытия (token-based Jaccard)
  coating_similarity_threshold: 0.8

  # Режим сравнения параметров:
  # false = по пересечению ключей (ключ отсутствующий в одном наборе игнорируется)
  # true = строгий режим по объединению (не рекомендуется)
  strict_union_keys: false

  # Детальный debug per-parameter в лог (true = выводить matched/mismatched)
  debug_per_parameter: true
```

| Параметр | Дефолт | Описание |
|----------|--------|----------|
| `success_threshold` | 0.7 | Score >= порога → success=true |
| `fuzzy_threshold` | 0.6 | Минимальный fuzzy score для кандидата |
| `v2_exact_threshold` | 0.99 | Порог подтверждения V2 exact match |
| `coating_similarity_threshold` | 0.8 | Порог fuzzy-match для покрытия |
| `strict_union_keys` | false | Режим сравнения ключей параметров |
| `debug_per_parameter` | true | Детальный debug per-parameter в лог |

---

## Быстрый старт

### 1. Построить индекс ЕСН

```bash
python cli.py ens build-index "data/ENS_Крепеж.xlsx" -o models/hardware/ens_hardware.pkl
```

При построении выводится статистика:
```
ENS columns: 129 total | 26 explicitly mapped | 103 auto-mapped | 0 unmapped
```

Все 129 колонок Excel автоматически получают нормализованные snake_case ключи через транслитерацию, даже если они не прописаны в `ens_column_mapping.yaml`.

### 2. Сгенерировать маски

```bash
# Один стандарт
python cli.py generate-masks -d cache/masks.db -i models/hardware/ens_hardware.pkl --llm --standard "ГОСТ 7798-70"

# Все стандарты
python cli.py generate-masks -d cache/masks.db -i models/hardware/ens_hardware.pkl --llm
```

### 3. Обработать файл (Excel → Excel + result.db)

```bash
# Полная обработка с сохранением в result.db и экспортом в Excel
python cli.py batch data/nomenclature.xlsx \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl \
  -o output/results.xlsx \
  --result-db result.db \
  --workers 8

# Только успешно распознанные
python cli.py batch data/nomenclature.xlsx \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl \
  -o output/results.xlsx \
  --result-db result.db \
  --success-only \
  --workers 8

# С debug-информацией (для анализа)
python cli.py batch data/nomenclature.xlsx \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl \
  -o output/results.xlsx \
  --result-db result.db \
  --include-details \
  --workers 8
```

### 4. Анализ качества распознавания

```bash
python cli.py analyze-quality data/nomenclature.xlsx \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl \
  -o output/quality_report.json
```

### 5. Работа с result.db

```bash
# Статистика по result.db
python cli.py result-stats --result-db result.db

# Экспорт result.db в Excel (с обогащением исходного файла)
python cli.py result-export \
  --result-db result.db \
  --output output/enriched.xlsx \
  --source data/nomenclature.xlsx \
  --article-col "Артикул" \
  --name-col "наименование"

# Записи, измененные после перегенерации масок
python cli.py result-stats --result-db result.db --since 2026-05-14T10:00:00
```

### Вызовы для отладки

```bash
# Построение индекса из тестового файла
python cli.py ens build-index "data/_ЕНС_Крепеж_test.xlsx" -o models/hardware2/ens_hardware.pkl

# Генерация масок для тестового индекса
python cli.py generate-masks -d cache/masks.db -i models/hardware2/ens_hardware.pkl --llm

# Batch-обработка через тестовый индекс (Excel → Excel)
python cli.py batch data/nomenclature.xlsx \
  -d cache/masks.db \
  -i models/hardware2/ens_hardware.pkl \
  -o output/results.xlsx \
  --result-db result.db \
  --workers 4

# Анализ качества
python cli.py analyze-quality data/nomenclature.xlsx \
  -d cache/masks.db \
  -i models/hardware2/ens_hardware.pkl \
  --workers 4 \
  -o output/quality.xlsx \
  -j output/quality.json

# Запуск с логированием DEBUG — видно все шаги PARAM_MATCH, Fallback, _apply_mask
python -u cli.py batch data/nomenclature.xlsx \
  --db cache/masks.db \
  --ens-index models/hardware2/ens_hardware.pkl \
  --output output/results.xlsx \
  --result-db result.db \
  2>&1 | grep -E "(PARAM_MATCH|Fallback|_apply_mask)"

# Диагностика отдельной строки + паттерна
python cli.py diagnose "Болт (2)-8-26-Кд-ОСТ 1 31133-80" \
  --db cache/masks.db \
  --ens-index models/hardware2/ens_hardware.pkl
```

### Prod

```bash
# Построение индекса из production-файла
python cli.py ens build-index "data/_ЕНС_Крепеж_05.05.2026.xlsx" -o models/hardware/ens_hardware.pkl

# Генерация масок
python cli.py generate-masks -d cache/masks.db -i models/hardware/ens_hardware.pkl --llm

# Batch-обработка (Excel → Excel + result.db)
python cli.py batch data/nomenclature1.xlsx \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl \
  --workers 8 \
  -o output/results.xlsx \
  --result-db result.db

# Анализ качества
python cli.py analyze-quality data/nomenclature1.xlsx \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl \
  -o output/quality.xlsx \
  -j output/quality.json
```

---

## CLI Команды

### Обработка

| Команда | Описание |
|---------|----------|
| `batch <input.xlsx> -d <db> -i <ens> -o <output.xlsx>` | Пакетная обработка (Excel → Excel + result.db) |
| `batch ... --success-only` | Только успешно распознанные в Excel |
| `batch ... --include-details` | Включить debug-информацию |
| `batch ... --workers 8` | Многопоточная обработка (8 процессов) |
| `batch ... --result-db result.db` | Сохранять результаты в result.db |
| `process-parametric <text> -d <db> -i <ens>` | Обработка одной строки |
| `process <file> --auto` | LLM-обработка (Legacy Mode) |

### Генерация масок

| Команда | Описание |
|---------|----------|
| `generate-masks -d <db> -i <ens> --llm` | Все стандарты |
| `generate-masks -d <db> -i <ens> --llm --standard "ГОСТ 7798-70"` | Один стандарт |
| `generate-masks -d <db> -i <ens> --llm --limit 5` | Первые 5 (отладка) |
| `cleanup -d <db> -t 0.5` | Удалить маски с низким score |

### Анализ качества

| Команда | Описание |
|---------|----------|
| `analyze-quality <file> -d <db> -i <ens>` | JSON-отчет в stdout |
| `analyze-quality <file> -d <db> -i <ens> -o <file>` | Сохранить JSON в файл |

### ENS индекс

| Команда | Описание |
|---------|----------|
| `ens build-index <excel> -o <pkl>` | Построить индекс |
| `ens search <query> -i <pkl>` | Поиск по индексу |
| `ens analyze <excel> -i <pkl>` | Анализ покрытия |

### Результаты (result.db)

| Команда | Описание |
|---------|----------|
| `result-stats --result-db result.db` | Статистика по result.db |
| `result-stats --result-db result.db --since <ISO>` | Записи, измененные после даты |
| `result-export --result-db result.db -o <excel>` | Экспорт result.db в Excel |
| `result-export --result-db result.db -o <excel> --source <input>` | Экспорт с обогащением исходного файла |

### Утилиты

| Команда | Описание |
|---------|----------|
| `prompts` | Список промптов с keywords, service, model |
| `models [--api <name>]` | Доступные модели у API-провайдера |
| `detect <text>` | Определить категорию по keywords |
| `export -o <file>` | Экспорт результатов (Legacy JSON) |
| `stats` | Статистика results.db (Legacy) |
| `errors -l <n>` | Последние ошибки |

```bash
# Список моделей по провайдерам
python cli.py models --api openwebui
python cli.py models --api mws
python cli.py models --api gigachat
python cli.py models --api mts_ai

# Автоопределение API
python cli.py models
```

### Параметрическая обработка (одна строка)

```bash
python cli.py process-parametric "Болт (2)-12-96-Окс.Фос.ЭФП-ОСТ 1 31133-80" \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl
```

Результат: JSON с извлечёнными параметрами, ENS match и confidence score.

### Диагностика строки

```bash
python cli.py diagnose "Болт (2)-8-26-Кд-ОСТ 1 31133-80" \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl
```

Диагностика показывает:
1. Извлечённые standard и item_type
2. Найденную маску и regex-паттерн
3. Извлечённые params (named groups)
4. Parametric match (score, ENS-кандидаты)
5. Fuzzy match (если parametric не дал результата)
6. V2 scoring (params vs ens_params_mask)
7. Coating validation (проверка покрытия)

---

## Анализ качества

Команда `analyze-quality` собирает статистику распознавания по группам `(item_type, standard)`:

```bash
python cli.py analyze-quality data/nomenclature.xlsx \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl \
  -o output/quality_report.json
```

### Формат JSON-отчета

```json
{
  "summary": {
    "total": 155,
    "ens_code": { "found": 120, "percent": 77.42 },
    "params": { "found": 98, "percent": 63.23 },
    "ens_params": { "found": 75, "percent": 48.39 },
    "both": { "found": 65, "percent": 41.94 }
  },
  "groups": [
    {
      "item_type": "БОЛТ",
      "standard": "ГОСТ 7795-70",
      "total": 15,
      "ens_code": { "found": 12, "percent": 80.0 },
      "params": { "found": 10, "percent": 66.67 },
      "ens_params": { "found": 8, "percent": 53.33 },
      "both": { "found": 7, "percent": 46.67 }
    }
  ]
}
```

### Метрики

| Метрика | Описание | Источник |
|---------|----------|----------|
| `ens_code` | Определен код ЕСН | TF-IDF или regex-маска |
| `params` | Распознаны параметры из текста | Парсинг regex-маской |
| `ens_params` | Распознаны ENS-параметры | Модель ENS (только при `ens_code != null`) |
| `both` | И `params`, и `ens_params` | Комбинированная |

---

## Хранение результатов (result.db)

Система автоматически сохраняет результаты сопоставления в SQLite-базу `result.db` с отслеживанием версий масок.

### Структура таблицы

```sql
CREATE TABLE nomenclature_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article TEXT NOT NULL,          -- Артикул из исходного Excel
    name TEXT NOT NULL,             -- Наименование из исходного Excel
    standard TEXT,                  -- Стандарт (ГОСТ/ОСТ)
    item_type TEXT,                 -- Тип изделия
    level TEXT,                     -- Уровень обработки
    success INTEGER NOT NULL,       -- Успешно распознано (0/1)
    params TEXT,                    -- Извлеченные параметры (JSON)
    ens_code TEXT,                  -- Код ЕНС
    ens_name TEXT,                  -- Наименование ЕНС
    ens_params TEXT,                -- Параметры из индекса ЕНС (JSON)
    ens_params_mask TEXT,           -- Параметры из ENS-имени по маске (JSON)
    confidence REAL,                -- Уверенность (0.0..1.0)
    match_type TEXT,                -- Тип сопоставления (eng)
    match_type_ru TEXT,             -- Тип сопоставления (rus)
    coating_substitution TEXT,      -- Информация о подстановке покрытия (JSON)
    fuzzy_mismatched_params TEXT,   -- Несовпавшие параметры fuzzy (JSON)
    mask_id INTEGER,                -- ID маски в БД масок
    mask_pattern TEXT,              -- Regex-паттерн маски
    mask_pattern_hash TEXT,         -- MD5-хеш паттерна (для отслеживания изменений)
    details TEXT,                   -- Дополнительная информация (JSON)
    processing_time_ms REAL,        -- Время обработки в мс
    created_at TEXT,                -- Время создания записи
    updated_at TEXT,                -- Время последнего обновления
    UNIQUE(article, name)           -- Уникальность по (артикул, наименование)
);
```

### Логика upsert

```
При обработке номенклатуры:
  ├─ Запись (article, name) не существует → INSERT (new_record)
  ├─ Запись существует, mask_pattern_hash совпадает → UPDATE updated_at только (mask_unchanged)
  └─ Запись существует, mask_pattern_hash отличается → UPDATE всех полей (mask_changed)
```

Это позволяет:
- **Повторно запускать batch** — существующие записи не дублируются
- **Отслеживать изменения масок** — записи обновляются только если маска изменилась
- **Получать дельту** — `result-stats --since <дата>` показывает записи, измененные после перегенерации масок

### Excel output

При обработке batch создается Excel-файл с исходными колонками + дополнительными:

| Колонка | Источник | Пример |
|---------|----------|--------|
| `Код ЕНС` | `ens_code` | `1000613872` |
| `Наименование ЕНС` | `ens_name` | `Болт 2М12х1,25-6gx60.109.40Х.016 ГОСТ 7798-70` |
| `Уровень` | `level` | `parametric_match` |
| `Распознано` | `success` | `Да` / `Нет` |
| `Уверенность` | `confidence` | `0.95` |
| `Тип сопоставления` | `match_type_ru` | `Полное совпадение параметров с индексом` |
| `Подстановка покрытия` | `coating_substitution` | `{"original": "Кд", "corrected": "Н.Кд"}` |
| `Несовпавшие параметры` | `fuzzy_mismatched_params` | `{"покрытие": "'Кд' vs 'Хим.Пас' (sim=0.00)"}` |

---

## Keyword-based маршрутизация

```
Номенклатура: "Болт М16х130.52.019 ГОСТ 7798-70"
  |
  v
StandardExtractor: standard="ГОСТ 7798-70", item_type="БОЛТ" (UPPERCASE)
  |
  v
Keywords lookup: "болт" in hardware.keywords -> True
  |
  v
prompt_id="hardware" -> prompts.yaml:
  service="mws", model="qwen2.5-72b-instruct", temp=0.1
  |
  v
Маска найдена в БД? -> Да -> ParametricMatch (Level 6)
  |
  v
Извлечены params: {тип_изделия: "Болт", диаметр: "М16", длина: "130", ...}
  |
  v
Поиск в ЕСН:
  1. Точное совпадение параметров (с _remap_params)
  2. name_exact: text == ens_name?
  3. Fuzzy fallback: token-based matching для покрытия/материала
     "Окс.Фос.ЭФП" ~ "Фос.Окс.ЭФП" = 100% (перестановка токенов)
  |
  v
ens_code="1000613872", score=0.95
```

### Каскадный поиск по 5 источникам

При `item_type=unknown` или нет совпадения по keywords:

1. `item_type` — "болт", "гайка", "труба"...
2. `standard_type` — "ГОСТ", "ОСТ", "ТУ", "ISO", "DIN", "РАМ"
3. `standard_normalized` — "ГОСТ 7798-70" (конкретный стандарт)
4. `examples` — первые 3 примера из ЕСН (содержимое)
5. `name` — полное наименование номенклатуры

### Каскадный поиск маски в БД

```
1. Точное совпадение (standard, item_type.upper())
2. Fallback: исходный регистр item_type
3. Fallback: маска только по стандарту (любой item_type)
4. Fallback: case-insensitive SQL-поиск
5. Если найдена неактивная маска -> автоактивация
```

---

## Генерация regex-масок через LLM

### Промпт

Шаблон загружается из `prompts/templates/mask_generation.txt`:

```
Ты — эксперт по техническим стандартам ГОСТ/ОСТ/ТУ и регулярным выражениям Python.

=== ПРИМЕРЫ ИЗ ЕСН ===

1. ИСХОДНАЯ СТРОКА: "Болт 2М12х1,25-6gx60.109.40Х.016 ГОСТ 7798-70"
   СТРУКТУРА: [Болт] (?P<тип_изделия>Болт) [ ] (?P<номинальный_диаметр_резьбы>2М12) ...
   ПОЛЯ ЕСН:
     полное_наименование: Болт 2М12х1,25-6gx60.109.40Х.016 ГОСТ 7798-70
     тип_изделия: Болт
     исполнение: 2
     ...

=== СТАТИСТИКА ===
  тип_изделия: 659 из 659
  номинальный_диаметр_резьбы: 659 из 659

Ответ в формате:
```json
{
  "pattern": "ваш regex",
  "params": ["тип_изделия", "номинальный_диаметр_резьбы", ...],
  "required": ["тип_изделия", "номинальный_диаметр_резьбы"]
}
```
```

### Требования к regex (встроены в промпт)

| # | Правило | Пример |
|---|---------|--------|
| 1 | Имена групп max 30 символов | `наружный_диаметр_диаметр_вписа` |
| 2 | Первое поле — `тип_изделия` | `^(?P<тип_изделия>\w+)` |
| 3 | Полная строка: `^...$` | с `^` и `$` |
| 4 | Дублирующиеся значения = 1 параметр | если диаметр=номинальный_диаметр=10, один параметр |
| 5 | Покрытие: `[\w.]+` | матчит `Окс.Фос.ЭФП` |
| 6 | Разделители: `[\s\-]*` или `[\.\s]*` | гибко |
| 7 | Опциональные части: `?` | для необязательных полей |

### Релаксация паттернов (runtime)

Маски из БД динамически адаптируются перед применением:

| Rule | Проблема | Фикс |
|------|----------|------|
| 1 | Латинские буквы в типе | `Болt` → `Болт` |
| 2 | Нет пробела между группами | `)?(?P<` → `)?\s*(?P<` |
| 3 | Нет пробела перед `(` | `(?:\(` → `(?:\s*\(` |
| 4 | Точка вместо `[.,]` | `\d+\.\d+` → `\d+(?:[.,]\d+)?` |
| 5 | `ОСТ1` без пробела | `ОСТ1` → `ОСТ\s*1` |
| 6 | Опциональная группа | `)?\s*-(?P<` → `)?` |
| 7 | Шайба: лишний диаметр | `\d+\-?\d+` → skip intermediate |
| 8 | Винт: нет `\s*` перед `(` | `(?:\(` → `(?:\s*\(` |
| 9 | Исполнение без скобок | `\((\d+)\)` → `\(?(\d+)\)?` |
| 10 | Нет шага резьбы | `M\d+` → `M\d+(?:[xX×]\d+(?:[.,]\d+)?)?` |
| 11 | Класс допуска жадный | `[\w]*` → `[a-zA-Zа-яА-Я]*` |
| 12 | `tipo_rezby` крадет `M` | убрать группу `tipo_rezby` |
| 13 | Группа прочности жадная | `\d+\.\d+` → `\d{1,2}(?:\.\d+)?` |

### Отладка

Промпты сохраняются в `prompts/debug/`:
```
prompts/debug/
|-- БОЛТ_ГОСТ_7798-70.txt           # промпт
|-- БОЛТ_ГОСТ_7798-70_a1.txt        # ответ попытки 1
|-- БОЛТ_ГОСТ_7798-70_a2.txt        # ответ попытки 2 (retry)
|-- БОЛТ_ГОСТ_7798-70_failed_a1.txt # при ошибке парсинга
```

---

## Fuzzy Matching

Проблема: покрытие в номенклатуре (`Окс.Фос.ЭФП`) может отличаться от ЕСН (`Фос.Окс.ЭФП`) — перестановка токенов.

Решение: token-based Jaccard similarity:
- Токены извлекаются по `[a-zA-Zа-яА-Я0-9]+`
- Цифры удаляются из токенов (`Кд3` -> `кд`)
- Сравниваются множества (порядок не важен)
- Порог совпадения: >= 80%

Примеры:

| Входная строка | ЕСН | Similarity | Результат |
|----------------|-----|------------|-----------|
| `Окс.Фос.ЭФП` | `Фос.Окс.ЭФП` | **1.00** | MATCH |
| `Кд.фос.окс` | `Кд3.фос.окс` | **1.00** | MATCH (цифра нормализована) |
| `Ц.фос.окс` | `Ц3.хр` | **0.25** | MISMATCH |

Активируется при `parametric_match score < 0.7` или отсутствии ENS match.

---

## Валидация покрытий

Гибридная система валидации покрытий: **фактические данные из ЕСН + LLM-дополнение**.

### Проблема

Покрытие в номенклатуре (`"Кд"`) может семантически не совпадать с покрытием в ЕСН (`"Хим.Пас"`), хотя fuzzy similarity высокий (общий токен). Для коррозионно-стойких сталей (`14Х17Н2`) простое кадмиевое покрытие `"Кд"` некорректно — требуется `"Н.Кд"` (никелевый подслой).

### Архитектура валидации

```
Level 6: ParametricMatch
  └─ ENS найден, параметры извлечены
       |
       v
Level 8: CoatingValidation
  1. Извлекаем покрытие из текста + марку стали из ENS
  2. Если покрытие = "" / "Бп" / "без покрытия" → skip (валидно)
  3. Загружаем coating_rules из config.yaml
  4. Проверяем: покрытие допустимо для марки?
     ├── Да → match подтвержден
     ├── Нет, но есть auto_substitution → заменяем в params
     └── Нет, strict_mode=true → REJECT match (success=false)
```

### Построение правил (гибридный подход)

**Phase 1 — Индексация ЕСН** (`coating_indexer.py`):
```python
from coating_indexer import build_coating_rules_for_standard

rules, llm_used = build_coating_rules_for_standard(
    standard="ОСТ 1 31509-80",
    item_type="винт",
    llm_generator=coating_llm_client  # None для offline-режима
)
# Результат: {"14Х17Н2": ["Н.Кд", "Хим.Пас", "Н.Пас"]}
```

**Phase 2 — LLM-дополнение** (`coating_llm_client.py`):
- Активируется, если в ЕСН < 2 примеров на марку
- LLM получает: стандарт + марки стали + контекст
- Возвращает: допустимые покрытия с пояснениями
- ENS-данные имеют приоритет над LLM

**Phase 3 — Валидация при сопоставлении** (`automated_processor.py`):
```python
# Пример: "Винт 3-6-Кд" + ENS "Винт 3-6-Н.Кд" (14Х17Н2)
# "Кд" не в списке допустимых для 14Х17Н2
# → auto_substitution: "Кд" → "Н.Кд" в params
# → match подтвержден с исправленным покрытием
```

### Специальные значения "без покрытия"

Следующие значения покрытия пропускают валидацию (валидны для любой марки):
- `""` (пустая строка)
- `"Бп"`, `"бп"`, `"без покрытия"`
- `"нет"`, `"-"`, `"none"`, `"н/п"`

### Конфигурация coating_rules

```yaml
coating_rules:
  # Марка → допустимые покрытия (из ЕСН + LLM)
  # Определяет какие покрытия допустимы для каждой марки материала
  material_coating_map:
    # Коррозионно-стойкие стали: кадмиевое (Кд) НЕДОПУСТИМО → только Н.Кд
    "14Х17Н2": ["Н.Кд", "Хим.Пас", "Н.Пас", "Пас"]
    "12Х18Н10Т": ["Н.Кд", "Хим.Пас", "Н.Пас"]
    "08Х18Н10Т": ["Н.Кд", "Хим.Пас", "Н.Пас"]
    # Конструкционные стали: кадмиевое, цинковое, оксидные и др.
    "30ХГСА": ["Кд", "Цд", "Окс", "Фос", "Окс.Фос", "Фос.Окс", "Неп", "Пас", "Бп"]
    "40Х": ["Кд", "Цд", "Окс", "Фос", "Окс.Фос", "Фос.Окс", "Неп", "Пас", "Бп"]

  # Авто-замена покрытия: wrong → correct (если матчит material_pattern)
  # Работает на этапе fuzzy matching (НЕ на этапе индексации)
  auto_substitution:
    # Для коррозионно-стойких сталей: Кд → Н.Кд
    - material_pattern: "^(14Х17Н2|12Х18Н10Т|08Х18Н10Т)$"
      wrong_coating: "Кд"
      correct_coating: "Н.Кд"
      note: "Для коррозионно-стойкой стали Кд заменяется на Н.Кд"

  similarity_threshold: 0.8    # порог fuzzy-match для покрытий
  strict_mode: true            # true=reject при недопустимом, false=penalty
  auto_substitution_enabled: true  # включить авто-замену при matching
```

### Как работает auto_substitution в matching

Pipeline coating auto-substitution (automated_processor.py):

1. **Первый fuzzy pass** с исходным покрытием (например, "Кд")
2. **Определение марки материала** из лучшего кандидата ENS
3. **Проверка правил** auto_substitution: если material_pattern совпадает
   и wrong_coating ≈ извлечённому покрытию → применяем substitution
4. **Второй fuzzy pass** с исправленным покрытием ("Н.Кд")
5. **Выбор лучшего результата** между исходным и исправленным

### Результаты для типовых кейсов

| Входная строка | ENS | Марка | Результат |
|---|---|---|---|
| `Винт 3-6-Кд` | `Винт 3-6-Н.Кд` | `14Х17Н2` | **AUTO-SUBSTITUTION** Кд→Н.Кд, потом fuzzy match |
| `Винт 3-6-Кд` | `Винт 2,5-6-Хим.Пас` | `14Х17Н2` | **REJECTED** — Кд не допустимо, substitution не помогла |
| `Винт 3-6-Бп` | `Винт 3-6-Н.Кд` | `14Х17Н2` | **OK** — "без покрытия" валидно |
| `Винт 3-6-` | `Винт 3-6-Н.Кд` | `14Х17Н2` | **OK** — пустое покрытие валидно |
| `Болт 12-44-Кд` | `Болт 12-44-Кд` | `30ХГСА` | **OK** — Кд допустимо для конструкционной стали |

---

## API клиенты

| Провайдер | Аутентификация | Модели |
|-----------|---------------|--------|
| **OpenWebUI** | JWT (login/password) или API key | `Qwen/Qwen3-14B-AWQ` и др. |
| **MWS Cloud GPT** | API key | `qwen2.5-72b-instruct`, `gpt-oss-120b` |
| **GigaChat** | OAuth2 (Client Credentials) | `GigaChat`, `GigaChat-2`, `GigaChat-Pro` |
| **MTS AI** | API key (Bearer) | `cotype_pro_2.5` |

### Добавление нового API клиента

1. Создать класс в `api_clients/`, наследующий `BaseLLMClient`
2. Реализовать методы: `complete()`, `health_check()`, `get_models()`
3. Добавить инициализацию в `core/processor.py::_init_api_clients()`
4. Добавить в CLI в `cli.py` (команда `models`)
5. Обновить `config/config.yaml` с настройками нового API

---

## MTS AI

MTS AI — OpenAI-compatible API для моделей cotype. Использует стандартные endpoint'ы `/v1/chat/completions` и `/v1/models`.

### Конфигурация

```yaml
api:
  mts_ai:
    base_url: "https://demo6-fundres.dev.mts.ai/"
    api_key_file: "secrets/mts_ai_key.txt"
    default_model: "cotype_pro_2.5"
    timeout: 120
```

### Проверка подключения

```bash
# Список доступных моделей
python cli.py models --api mts_ai

# Пример ответа:
# MTS AI: ✓ available
# Models: cotype_pro_2.5, cotype_pro_1.5
```

### Использование в prompts.yaml

```yaml
prompts:
  my_prompt:
    name: "Тест MTS AI"
    keywords: ["тест"]
    service: "mts_ai"  # используем MTS AI вместо default
    # model: "cotype_pro_2.5"  # можно не указывать — берется из api.mts_ai.default_model
    temperature: 0.1
    system_prompt: "Вы — эксперт..."
    file: "prompts/templates/test.txt"
```

`service` и `model` в `prompts.yaml` опциональны. Если не указаны — берутся из `mask_generation.default_service` и `mask_generation.default_model`.

---

## Новые возможности

### Авто-маппинг колонок ЕНС (ens/loader.py)

Загрузчик автоматически преобразует все 129 колонок Excel в нормализованные snake_case ключи, даже если они не прописаны в `ens_column_mapping.yaml`:

```python
from ens.loader import create_ens_loader

loader = create_ens_loader(
    "data/ENS_Крепеж.xlsx",
    mapping_yaml="config/ens_column_mapping.yaml"  # опционально!
)
index = loader.get_index()
# Все колонки доступны по нормализованным ключам:
# "Марка стали" → "marka_stali"
# "D, мм" → "d_mm"
# "Класс прочности" → "klass_prochnosti"
```

Механизм:
1. **Explicit mapping** — колонки из `ens_column_mapping.yaml` (26 шт.)
2. **Regex patterns** — типовые паттерны (`D, мм` → `d`, `покрытие` → `покрытие`)
3. **Auto snake_case** — транслитерация + нормализация для всех остальных (103 шт.)

### Авто-генерация ens_column_mapping.yaml (scripts/auto_mapping.py)

```bash
# Сгенерировать маппинг из Excel
python scripts/auto_mapping.py "data/ENS_Крепеж.xlsx" -o config/ens_column_mapping.yaml

# Дополнить существующий маппинг
python scripts/auto_mapping.py "data/ENS_Крепеж.xlsx" --append -o config/ens_column_mapping.yaml
```

### Match type в выходном JSON

Каждая запись в `results.json` содержит поле `match_type` — тип сопоставления:

| `match_type` | `match_type_ru` | Когда |
|---|------|---|
| `name_exact` | Сопадение по наименованию | text == ens_name (прямое равенство) |
| `parametric_full` | Полное совпадение параметров с индексом | Все params совпали с ENS индексом |
| `v2_exact` | Полное совпадение параметров с маской ENS | Params совпали с ens_params_mask |
| `fuzzy_fallback` | Нечеткое совпадение (fuzzy matching) | Fuzzy matching нашёл кандидата выше threshold |
| `coating_substituted` | Совпадение после подбора правильного покрытия | Покрытие было авто-заменено |

### Fuzzy mismatched params

При `match_type=fuzzy_fallback` в выходном JSON добавляется `fuzzy_mismatched_params` — параметры, которые не совпали у лучшего кандидата:

```json
{
  "match_type": "fuzzy_fallback",
  "match_type_ru": "Нечеткое совпадение (fuzzy matching)",
  "fuzzy_mismatched_params": {
    "покрытие": "'Кд' vs 'Хим.Пас' (sim=0.00)",
    "марка_материала": "'30ХГСА' vs '14Х17Н2'"
  }
}
```

Значения `fuzzy_mismatched_params`:
- `null` — fuzzy не применялся (exact/parametric match)
- `{}` — fuzzy применялся, все параметры совпали
- `{"покрытие": "..."}` — есть несовпавшие параметры с пояснением

---

## Производительность

### Оптимизации (2026-05-14)

| Оптимизация | Эффект | Реализация |
|-------------|--------|-----------|
| **Индексация ENS** | 10–50× ускорение поиска | `Dict[(std, type), List[item]]` в `parametric_client.py` |
| **O(1) поиск по коду** | Мгновенный lookup ENS | `Dict[code, item]` в `parametric_client.py` |
| **Кэш compiled regex** | 1.5–2× ускорение | `Dict[pattern, compiled]` в `parametric_client.py` |
| **Кэш coating_rules** | 1.2–1.5× ускорение | Одна загрузка YAML на сессию в `automated_processor.py` |
| **Lazy debug-логи** | 1.3–2× ускорение | `logger.debug("%s", arg)` вместо `f-string` в `automated_processor.py` |
| **Multiprocessing** | 4–8× ускорение | `ProcessPoolExecutor` в `cli.py` |
| **Кэш ENS candidates** | 2–3× ускорение | `Dict[(std, type), List[item]]` в `automated_processor.py` |
| **Итого** | **50–200×** | С 8.57 сек/запись → 0.1–0.3 сек/запись |

### Индексация ENS

При загрузке индекса ЕНС (245 000 записей) строятся два индекса:

```python
# Индекс по (стандарт, тип) — поиск O(100-500) вместо O(245K)
self._ens_by_standard_type: Dict[Tuple[str, str], List[Dict]]

# Индекс по коду — lookup O(1)
self._ens_by_code: Dict[str, Dict]
```

### Многопоточная обработка

```bash
# Однопоточный режим (для отладки)
python cli.py batch input.xlsx ... --workers 1

# Авто (число CPU ядер)
python cli.py batch input.xlsx ... --workers 8

# Рекомендуемые значения:
#   OpenWebUI (локальные модели): 4-8 workers
#   MWS Cloud GPT: 2-4 workers
#   GigaChat API: 2-4 workers (лимиты API)
#   MTS AI: 2-4 workers
```

Каждый worker-процесс самостоятельно инициализирует `Processor` и загружает ENS index (lazy init), избегая сериализации 5 ГБ через pickle.

### Рекомендации по производительности

- **ENS индекс 5 ГБ**: загружается 1 раз при старте, занимает ~6 ГБ RAM
- **SQLite WAL mode**: `cache/masks.db` и `result.db` используют WAL для параллельного чтения/записи
- **Chunk size**: для очень больших файлов (>10K записей) используйте `--chunk-size 100`

---

## Troubleshooting

### "settings недоступен, читаем prompts.yaml напрямую"
Проверьте что `settings` передаётся в `AutomatedParametricProcessor` и `LLMMaskGenerator`.

### "Fallback клиент mws не инициализирован!"
Проверьте что в `cli.py` создаются MWS/GigaChat/MTS AI клиенты, не только OpenWebUI.

### "Все N попытки неудачны"
Проверьте `prompts/debug/*_a1.txt` — там сохраняется raw ответ от LLM.

### "JSON не найден" (после LLM-ответа)
LLM генерирует `\s`, `\d`, `\w` внутри JSON-строки pattern. Система автоматически предобрабатывает такие ответы, но старые маски могут требовать перегенерации.

### Маска не используется (tfidf_fallback)
- Проверьте `item_type` маски — должен быть в UPPERCASE (`БОЛТ`, не `болт` или `Болт`)
- Удалите старые маски и пересоздайте:
  ```sql
  DELETE FROM masks WHERE standard = 'ГОСТ 7798-70';
  ```
- Перегенерируйте: `generate-masks ... --standard "ГОСТ 7798-70" --llm`

### TF-IDF fallback возвращает success=true
Исправлено — теперь `tfidf_fallback` всегда `success: false`. Параметры не извлечены, `ens_code` только в `details.tf_idf_ens_candidate`.

### Покрытие не матчится (разные обозначения)
Fuzzy matching автоматически обрабатывает перестановку токенов. Если не срабатывает:
- Проверьте `--include-details` в выводе — там будет `fuzzy_used: true/false`
- Убедитесь что `parametric_match score < 0.7` (триггер fuzzy)

### Coating validation REJECTED корректный match
Если в логах `[PARAM_MATCH] REJECTED: coating ... not allowed for material`:
- Проверьте `config.yaml → coating_rules.material_coating_map` — марка есть в справочнике?
- Запустите индексацию: `python -c "from coating_indexer import build_coating_rules_for_standard; build_coating_rules_for_standard('СТАНДАРТ', 'ТИП')"`
- Временно отключите strict_mode: `coating_rules.strict_mode: false`

### "Бп" (без покрытия) отклоняется
Убедитесь что значение покрытия в номенклатуре точно `"Бп"` — без пробелов, без дополнительных символов. Система распознает: `Бп`, `бп`, `без покрытия`, `""`, `-`, `нет`.

### Кд не заменяется на Н.Кд для 14Х17Н2
Проверьте в логах `[PARAM_MATCH] Coating auto-substitution`. Если не срабатывает:
- Проверьте `coating_rules.auto_substitution` — material_pattern покрывает вашу марку?
- Проверьте `coating_rules.auto_substitution_enabled: true`
- Марка стали должна быть в ENS-записи (поле `марка_материала` или `марка_стали`)

### Низкое качество распознавания
Используйте `analyze-quality` для диагностики:
```bash
python cli.py analyze-quality data/nomenclature.xlsx -d cache/masks.db -i models/hardware/ens_hardware.pkl -o quality.json
```

Анализируйте `params_percent` и `ens_params_percent` по группам. Низкие значения указывают на проблемы с масками — перегенерируйте через LLM.

### MTS AI не подключается
- Проверьте `secrets/mts_ai_key.txt` — API key должен быть в формате `Bearer sk-...` или просто `sk-...`
- Проверьте доступность endpoint: `curl -H "Authorization: Bearer $(cat secrets/mts_ai_key.txt)" https://demo6-fundres.dev.mts.ai/v1/models`
- Убедитесь что в `config.yaml` base_url не содержит пробелов в конце (фикс: `__post_init__` делает `strip()`)

### Медленная обработка (>5 сек/запись)
- Проверьте что индексы ENS построены: `python -c "from core.parametric_client import ParametricENSClient; c = ParametricENSClient('models/hardware/ens_hardware.pkl'); print(len(c._ens_by_standard_type), 'groups')"`
- Используйте multiprocessing: `--workers 8`
- Проверьте логи на наличие повторных загрузок coating_rules (должна быть 1 раз)

### result.db не обновляется после перегенерации масок
- Проверьте `mask_pattern_hash` в таблице: `sqlite3 result.db "SELECT article, mask_pattern_hash FROM nomenclature_results LIMIT 5"`
- При изменении маски hash должен измениться → запись обновится
- Используйте `result-stats --since <дата>` для проверки

---

## Upsert семантика

### Legacy (results.db)
База данных SQLite гарантирует уникальность записей по составному ключу `(article, prompt_id)`:
- Если запись существует — она обновляется
- Если записи нет — она создается
- Повторный запуск безопасен и не создает дубликатов

### result.db (новая)
База `result.db` использует upsert по `(article, name)` с отслеживанием версий масок:
- Новая запись → INSERT
- Существующая, маска не изменилась → soft touch (только updated_at)
- Существующая, маска изменилась → полное UPDATE всех полей

---

## Тестирование

```bash
# Тест определения категории
python cli.py detect "Болт М12х1.25-6gx100.58 ГОСТ 7795-70"

# Проверка доступности API
python cli.py models --api mts_ai

# Тест с небольшой выборкой
python cli.py process test_sample.xlsx --auto -w 2

# Проверка статистики
python cli.py stats

# Проверка result.db
python cli.py result-stats --result-db result.db
```

---

## Логирование

Логи сохраняются в `logs/processor.log` и выводятся в консоль:
- Уровень логирования настраивается в `config.yaml` (`logging.level`)
- Ротация логов: 5 файлов по 10MB каждый

### Уровни логирования по модулям

| Модуль | INFO | DEBUG |
|---|------|-------|
| `automated_processor` | Инициализация, `Processing:`, `REJECTED:` | `[PARAM_MATCH]`, `[FUZZY]`, coating checks |
| `parametric_client` | Индексация ENS (info) | `[_calculate_match_score]` fuzzy details |
| `coating_indexer` | `Built map for`, `LLM augmented` | Сканирование ENS |
| `ens.loader` | Загрузка индекса, статистика колонок | Auto-mapped колонки |

При `level: "INFO"` в логе не будет пер-айтемных записей (каждый кандидат, каждый match). Только ключевые события: REJECTED, auto-substitution, итоговый score.

```yaml
logging:
  level: "INFO"        # INFO или DEBUG
  file: "logs/processor.log"
  max_size: 10485760   # 10MB ротация
  backup_count: 5      # 5 файлов истории
```

---

## Требования

- Python 3.11+
- SQLite 3.39+
- RAM: 4GB минимум, 8GB рекомендуется (6GB для ENS индекса 5GB)
- Диск: ~2GB для моделей и индексов

### Зависимости

```
pandas>=1.5
openpyxl>=3.0        # Чтение Excel
pyyaml>=6.0          # YAML конфиги
scikit-learn>=1.2    # TF-IDF
numpy>=1.24
click>=8.0           # CLI
requests>=2.28       # API клиенты
aiohttp>=3.8         # Async API (опционально)
```

---

```bash
# 1. Размер файла и тип
dir cache\masks.db /Q

# 2. WAL файлы (если WAL-режим — -wal/-shm могут быть рассинхронизированы)
dir cache\masks.db*

# 3. Режим журналирования
sqlite3 cache/masks.db "PRAGMA journal_mode;"
sqlite3 cache/masks.db "PRAGMA wal_checkpoint;"

# 4. Попробуйте простой запрос к БД через Python
python -c "import sqlite3; c=sqlite3.connect('cache/masks.db'); print(c.execute('SELECT count(*) FROM masks').fetchone())"

# 5. Проверка result.db
sqlite3 result.db "SELECT COUNT(*), SUM(success) FROM nomenclature_results;"
sqlite3 result.db "SELECT match_type, COUNT(*) FROM nomenclature_results GROUP BY match_type;"
```
