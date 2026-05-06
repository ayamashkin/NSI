# Nomenclature Processor + Automated Parametric Search

**Nomenclature Processor** — система автоматической обработки технической номенклатуры с каскадным анализом: regex-парсинг, авто-генерация regex-масок через LLM, параметрическое сопоставление с ЕСН (включая fuzzy matching для текстовых параметров) и TF-IDF fallback.

**Возможности:**
- Автоматическая классификация номенклатуры по категориям (крепеж, ЭРИ, материалы, покупные изделия)
- Извлечение структурированных технических параметров из неструктурированных наименований
- Пакетная обработка больших объемов данных (десятки тысяч позиций)
- Интеграция с локальными LLM (OpenWebUI), облачными API (MWS Cloud GPT, GigaChat)
- Параметрический поиск по ЕСН с fuzzy matching
- Анализ качества распознавания (JSON-отчеты)

---

## Содержание

- [Архитектура](#архитектура)
- [Установка](#установка)
- [Конфигурация](#конфигурация)
- [Быстрый старт](#быстрый-старт)
- [CLI Команды](#cli-команды)
- [Анализ качества](#анализ-качества)
- [Keyword-based маршрутизация](#keyword-based-маршрутизация)
- [Генерация regex-масок через LLM](#генерация-regex-масок-через-llm)
- [Fuzzy Matching](#fuzzy-matching)
- [Валидация покрытий](#валидация-покрытий)
- [API клиенты](#api-клиенты)
- [Troubleshooting](#troubleshooting)
- [Требования](#требования)

---

## Архитектура

```
nomenclature-processor/
├── config/
│   ├── __init__.py
│   ├── settings.py              # Конфигурация API и путей
│   ├── config.yaml              # Основная конфигурация API и БД
│   └── prompts.yaml             # Реестр промптов по категориям
├── core/
│   ├── __init__.py
│   ├── models.py                # Pydantic модели данных
│   ├── database.py              # SQLite manager с upsert
│   ├── processor.py             # Основной движок обработки
│   ├── automated_processor.py   # Параметрический процессор
│   ├── parametric_client.py     # Клиент параметрического поиска
│   ├── quality_analyzer.py      # Анализ качества распознавания
│   ├── llm_mask_generator.py    # LLM генерация масок
│   ├── auto_validator.py        # Валидация масок
│   ├── mask_database.py         # БД regex-масок (SQLite)
│   ├── standard_extractor.py    # Извлечение стандарта и типа
│   ├── integration.py           # Интеграция с существующей БД и API
│   ├── coating_indexer.py       # Индексация (марка→покрытия) из ЕСН
│   └── coating_llm_client.py    # LLM-запрос правил покрытий
├── api_clients/
│   ├── __init__.py
│   ├── base.py                  # Абстрактный класс клиента
│   ├── openwebui.py             # Клиент для OpenWebUI API
│   ├── mws_gpt.py               # Клиент для MWS Cloud GPT API
│   └── gigachat.py              # Клиенты для GigaChat API
├── utils/
│   ├── __init__.py
│   ├── excel_loader.py          # Загрузка Excel через pandas/openpyxl
│   └── excel_loader_simple.py   # Загрузка Excel только через openpyxl
├── parsers/
│   ├── cascade.py               # Каскадный парсер (regex → NER → LLM)
│   ├── regex_parser.py          # Regex уровень
│   └── ner_adapter.py           # NER адаптер
├── ens/
│   ├── loader.py                # Загрузчик ЕСН с многосхемной поддержкой
│   └── indexer.py               # TF-IDF индекс для поиска
├── secrets/                     # Учетные данные (не в git)
├── prompts/templates/           # Файлы промптов (.txt)
├── prompts/debug/               # Отладочные промпты
├── logs/                        # Директория для логов
├── cache/                       # БД масок, результатов
├── models/                      # Индексы ЕСН (.pkl)
├── cli.py                       # CLI интерфейс (точка входа)
├── results.db                   # SQLite база данных (создается автоматически)
├── requirements.txt
└── README.md
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
    |- Multi-provider: OpenWebUI, MWS, GigaChat
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
    |- Поиск по параметрам в индексе ЕСН (точное совпадение)
    |- Fuzzy fallback: token-based Jaccard для текстовых параметров
      (покрытие, материал) - учитывает перестановку токенов
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
git clone <repository-url>
cd nomenclature-processor

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### Секреты

```bash
mkdir secrets
echo "your_password" > secrets/openwebui_password.txt
echo "your_api_key" > secrets/mws_key.txt
echo "your_credentials" > secrets/gigachat_credentials.txt
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
    "14Х17Н2":  ["Н.Кд", "Хим.Пас", "Н.Пас"]
    "12Х18Н10Т": ["Н.Кд", "Хим.Пас", "Н.Пас"]
    "30ХГСА":   ["Кд", "Цд", "Окс", "Фос", "Бп"]
  auto_substitution:
    - material_pattern: "^(14Х17Н2|12Х18Н10Т)$"
      wrong_coating: "Кд"
      correct_coating: "Н.Кд"
  similarity_threshold: 0.8
  strict_mode: true
  auto_substitution_enabled: true
```

### config/prompts.yaml

```yaml
prompts:
  hardware:
    name: "Крепеж и метизы"
    keywords: ["болт", "гайка", "шуруп", "винт", "заклепка",
               "шпилька", "гвоздь", "штифт", "хомут", "анкер",
               "саморез", "шайба", "крепеж", "ГОСТ"]
    service: "mws"
    model: "qwen2.5-72b-instruct"
    temperature: 0.1
    system_prompt: "Вы - эксперт по стандартам ГОСТ..."
    file: "prompts/templates/hardware.txt"
    category: "hardware"

  rolledMetal:
    name: "Прокат"
    keywords: ["труба", "швеллер", "уголок", "балка", "профиль",
               "лист", "плита", "рулон", "круг", "квадрат",
               "regex:^ст\.сорт\.нерж\.|ст\.констр\.калибр\."]
    service: "gigachat"
    model: "GigaChat-2"
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

---

## Быстрый старт

### 1. Построить индекс ЕСН

```bash
python cli.py ens build-index "data/ENS_Крепеж.xlsx" -o models/hardware/ens_hardware.pkl
```

При построении выводится статистика:
```
📊 СТАТИСТИКА ПОЛЕЙ ЕСН (динамический анализ)
============================================================
Всего записей: 243449
Уникальных полей: 64

🔴 ОБЯЗАТЕЛЬНЫЕ ПОЛЯ (>=95% заполненности):
  ✅ покрытие                         243449/243449 (100.0%)
     |- из Excel: "Покрытие"
  ✅ исполнение                       237354/243449 (97.5%)
     |- из Excel: "Вариант исполнения"

🟡 РЕКОМЕНДУЕМЫЕ ПОЛЯ (50-95%):
  номинальный_диаметр_резьбы         200696/243449 (82.4%)
     |- из Excel: "Номинальный диаметр резьбы"

⚪ НЕИСПОЛЬЗУЕМЫЕ КОЛОНКИ EXCEL (43):
    - "Длина резьбы"
    - "Способ изготовления"
  💡 Добавьте в ens_column_mapping.yaml если содержат полезные данные
```

### 2. Построить правила покрытий (опционально, для валидации)

```python
python -c "
from coating_indexer import build_coating_rules_for_standard
from coating_llm_client import CoatingLLMClient
from llm_mask_generator import LLMMaskGenerator
from config.settings import get_settings

settings = get_settings()
generator = LLMMaskGenerator(settings.api, settings)
llm_client = CoatingLLMClient(generator)

rules, llm_used = build_coating_rules_for_standard(
    standard='ОСТ 1 31509-80',
    item_type='винт',
    llm_generator=llm_client
)
print(f'LLM used: {llm_used}')
print(f'Rules: {rules}')
"
```

### 3. Сгенерировать маски

Один стандарт для теста:
```bash
python cli.py generate-masks -d cache/masks.db -i models/hardware/ens_hardware.pkl --llm --standard "ГОСТ 7798-70"
```

Первые 5 стандартов (для отладки):
```bash
python cli.py generate-masks -d cache/masks.db -i models/hardware/ens_hardware.pkl --llm --limit 5
```

Все стандарты:
```bash
python cli.py generate-masks -d cache/masks.db -i models/hardware/ens_hardware.pkl --llm
```

### 4. Обработать файл

```bash
# Все записи
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/hardware/ens_hardware.pkl -o output/results.json

# Только успешно распознанные (для загрузки в систему)
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/hardware/ens_hardware.pkl -o output/results.json --success-only

# С debug-информацией (для анализа)
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/hardware/ens_hardware.pkl -o output/results.json --include-details
```

### 5. Анализ качества распознавания

```bash
# JSON в stdout + сохранение в файл
python cli.py analyze-quality data/nomenclature.xlsx \
  -d cache/masks.db \
  -i models/hardware/ens_hardware.pkl \
  -o output/quality_report.json
```

---

```bash
# вызовы для отладки
python cli.py ens build-index "data/_ЕНС_Крепеж_test.xlsx" -o models/hardware2/ens_hardware.pkl    
python cli.py generate-masks -d cache/masks.db -i models/hardware2/ens_hardware.pkl --llm      
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/hardware2/ens_hardware.pkl -o output/results.json
python cli.py analyze-quality data/nomenclature.xlsx -d cache/masks.db -i models/hardware2/ens_hardware.pkl -o output/quality.xlsx -j output/quality.json

# Запуск с логированием DEBUG — будет видно, где раньше терялись params
python -u cli.py batch data/nomenclature.xlsx --db cache/masks.db --ens-index models/hardware2/ens_hardware.pkl --output output/results.json 2>&1 | grep -E "(PARAM_MATCH|Fallback|_apply_mask)"

# Диагностика отдельной строки + паттерна
python test/test_params.py --text "Болт (2)-12-44-Окс.Фос.ЭФП-ОСТ 1 31133-80" --pattern "^Болт\\s*(?:\\((?P<исполнение>\\d+)\\)\\s*)?(?P<номинальный_диаметр_резьбы>\\d+(?:[.,]\\d+)?)\\s*[-\\s]*\\s*(?P<длина>\\d+(?:[.,]\\d+)?)\\s*[-\\s]*\\s*(?P<покрытие>[\\w.]+)\\s*$" --standard "ОСТ 1 31133-80"
``` 

```bash
# prod
python cli.py ens build-index "data/_ЕНС_Крепеж_24.03.2026.xlsx" -o models/hardware/ens_hardware.pkl    
python cli.py generate-masks -d cache/masks.db -i models/hardware/ens_hardware.pkl --llm      
python cli.py batch data/nomenclature1.xlsx -d cache/masks.db -i models/hardware/ens_hardware.pkl -o output/results.json
python cli.py analyze-quality data/nomenclature1.xlsx -d cache/masks.db -i models/hardware/ens_hardware.pkl -o output/quality.xlsx -j output/quality.json
```

## CLI Команды

### Обработка

| Команда | Описание |
|---------|----------|
| `batch <excel> -d <db> -i <index> -o <json>` | Пакетная обработка |
| `batch <excel> ... --success-only` | Только успешно распознанные |
| `batch <excel> ... --include-details` | Включить debug-информацию |
| `process-parametric <text> -d <db> -i <index>` | Обработка одной строки |
| `process <excel> --auto` | LLM-обработка (Legacy Mode) |

### Генерация масок

| Команда | Описание |
|---------|----------|
| `generate-masks -d <db> -i <index> --llm` | Все стандарты |
| `generate-masks -d <db> -i <index> --llm --standard "ГОСТ 7798-70"` | Один стандарт |
| `generate-masks -d <db> -i <index> --llm --limit 5` | Первые 5 (отладка) |
| `cleanup -d <db> -t 0.5` | Удалить маски с низким score |

### Анализ качества

| Команда | Описание |
|---------|----------|
| `analyze-quality <excel> -d <db> -i <index>` | JSON-отчет в stdout |
| `analyze-quality <excel> -d <db> -i <index> -o <json>` | Сохранить JSON в файл |

### ENS индекс

| Команда | Описание |
|---------|----------|
| `ens build-index <excel> -o <pkl>` | Построить индекс |
| `ens search <query> -i <index>` | Поиск по индексу |
| `ens analyze <excel> -i <index>` | Анализ покрытия |

### Утилиты

| Команда | Описание |
|---------|----------|
| `prompts` | Список промптов с keywords, service, model |
| `models [--api <name>]` | Доступные модели у API-провайдера |
| `detect <text>` | Определить категорию по keywords |
| `export -o <json>` | Экспорт результатов |
| `stats` | Статистика results.db |
| `errors -l <n>` | Последние ошибки |

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
    1. Точное совпадение параметров
    2. Fuzzy fallback: token-based matching для покрытия/материала
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
|-- БОЛТ_ГОСТ_7798-70.txt          # промпт
|-- БОЛТ_ГОСТ_7798-70_a1.txt       # ответ попытки 1
|-- БОЛТ_ГОСТ_7798-70_a2.txt       # ответ попытки 2 (retry)
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
  material_coating_map:
    "14Х17Н2":  ["Н.Кд", "Хим.Пас", "Н.Пас"]
    "12Х18Н10Т": ["Н.Кд", "Хим.Пас", "Н.Пас"]
    "30ХГСА":   ["Кд", "Цд", "Окс", "Фос", "Бп"]

  # Авто-замена: wrong → correct (если матчит material_pattern)
  auto_substitution:
    - material_pattern: "^(14Х17Н2|12Х18Н10Т|08Х18Н10Т)$"
      wrong_coating: "Кд"
      correct_coating: "Н.Кд"

  similarity_threshold: 0.8    # порог fuzzy-match
  strict_mode: true            # true=reject, false=penalty
  auto_substitution_enabled: true
```

### Результаты для типовых кейсов

| Входная строка | ENS | Марка | Результат |
|---|---|---|---|
| `Винт 3-6-Кд` | `Винт 3-6-Н.Кд` | `14Х17Н2` | **AUTO-SUBSTITUTION** Кд→Н.Кд |
| `Винт 3-6-Кд` | `Винт 2,5-6-Хим.Пас` | `14Х17Н2` | **REJECTED** — Кд не допустимо |
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

### Добавление нового API клиента

1. Создать класс в `api_clients/`, наследующий `BaseLLMClient`
2. Реализовать методы: `complete()`, `health_check()`, `get_models()`
3. Добавить инициализацию в `core/processor.py::_init_api_clients()`
4. Добавить в CLI в `cli.py` (команда `models`)
5. Обновить `config/config.yaml` с настройками нового API

---

## Troubleshooting

### "settings недоступен, читаем prompts.yaml напрямую"
Проверьте что `settings` передаётся в `AutomatedParametricProcessor` и `LLMMaskGenerator`.

### "Fallback клиент mws не инициализирован!"
Проверьте что в `cli.py` создаются MWS/GigaChat клиенты, не только OpenWebUI.

### "Все N попыток неудачны"
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

---

## Upsert семантика

База данных SQLite гарантирует уникальность записей по составному ключу `(article, prompt_id)`:
- Если запись существует — она обновляется
- Если записи нет — она создается
- Повторный запуск безопасен и не создает дубликатов

---

## Производительность

- **Параллельная обработка:** настраиваемое количество workers
- **Кэширование:** SQLite с UPSERT для промежуточных результатов
- **Рекомендации:**
  - OpenWebUI (локальные модели): 4-8 workers
  - MWS Cloud GPT: 2-4 workers
  - GigaChat API: 2-4 workers (лимиты API)

---

## Тестирование

```bash
# Тест определения категории
python cli.py detect "Болт М12х1.25-6gx100.58 ГОСТ 7795-70"

# Проверка доступности API
python cli.py models --api gigachat

# Тест с небольшой выборкой
python cli.py process test_sample.xlsx --auto -w 2

# Проверка статистики
python cli.py stats
```

---

## Логирование

Логи сохраняются в `logs/processor.log` и выводятся в консоль:
- Уровень логирования настраивается в `config.yaml` (`logging.level`)
- Ротация логов: 5 файлов по 10MB каждый

### Уровни логирования по модулям

| Модуль | INFO | DEBUG |
|---|---|---|
| `automated_processor` | Инициализация, `Processing:`, `REJECTED:` | `[PARAM_MATCH]`, `[FUZZY]`, coating checks |
| `parametric_client` | — | `[_calculate_match_score]` fuzzy details |
| `coating_indexer` | `Built map for`, `LLM augmented` | Сканирование ENS |

При `level: "INFO"` в логе не будет пер-айтемных записей (каждый кандидат, каждый match). Только ключевые события: REJECTED, auto-substitution, итоговый score.

```yaml
logging:
  level: "INFO"        # INFO или DEBUG
  file: "logs/processor.log"
  max_size: "10MB"
  backup_count: 5
```

---

## Требования

- Python 3.9+
- SQLite 3.35+ (WAL mode)
- Зависимости: `pandas`, `openpyxl`, `scikit-learn`, `pyyaml`, `click`, `tqdm`, `numpy`, `requests`, `pydantic`