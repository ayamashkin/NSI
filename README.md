# Nomenclature Processor + Automated Parametric Search

Система обработки технической номенклатуры с каскадным анализом: regex-парсинг, авто-генерация regex-масок через LLM, параметрическое сопоставление с ЕСН и TF-IDF fallback.

## Возможности

- **Извлечение стандартов** — автоматическое распознавание ГОСТ, ОСТ, ТУ, ISO, DIN, РАМ в наименованиях
- **Keyword-based маршрутизация** — автоматический выбор LLM-сервиса, модели и промпта по ключевым словам (как для основной обработки, так и для генерации масок)
- **Авто-генерация regex-масок** — LLM создаёт regex-паттерны с named groups на основе примеров из ЕСН
- **Авто-валидация масок** — тестирование на реальных данных из ЕСН, активация при score >= 0.85
- **Параметрическое сопоставление** — извлечение параметров по маске и поиск в ЕСН
- **TF-IDF fallback** — нечёткий поиск по наименованию при отсутствии маски
- **Два режима работы:**
  1. **LLM Mode** — полный разбор через LLM с извлечением всех параметров по стандарту (legacy, точный)
  2. **Parametric Mode** — быстрый каскад: regex/БД масок -> LLM-генерация маски -> параметрический поиск

## Архитектура

```
Level 0: StandardExtractor (regex)
    ├─ Извлечение стандарта (ГОСТ, ОСТ, ТУ, ISO, DIN, РАМ)
    ├─ Определение типа изделия (болт, гайка, шайба, труба, ...)
    └─ Keyword-based routing: тип -> prompt_id -> service/model

Level 1: MaskDatabase (SQLite + WAL + connection pool)
    ├─ Поиск маски по (standard, item_type)
    ├─ Использование активных масок (auto_score >= 0.85)
    └─ Fallback к генерации, если маски нет

Level 2: LLMMaskGenerator
    ├─ Автоопределение prompt_id по keywords из item_type/name
    ├─ Загрузка конфига из prompts.yaml (service, model, temperature, system_prompt)
    ├─ Fallback: раздел mask_generation в config.yaml
    ├─ Multi-provider: OpenWebUI, MWS, GigaChat
    └─ Retry-стратегия с повышением temperature

Level 3: AutoValidator
    ├─ Тест маски на примерах из ЕСН
    ├─ Score = matched_required / total_required
    └─ Порог активации: 0.85 (настраивается)

Level 5: MaskDatabase.save_mask()
    ├─ Автоактивация при score >= threshold
    └─ UPSERT по pattern_hash

Level 6: ParametricENSClient
    ├─ Извлечение параметров через regex-маску (named groups)
    ├─ Поиск по параметрам в индексе ЕСН
    └─ Score: взвешенное совпадение required-параметров

Level 7: TF-IDF Fallback
    ├─ Char-ngram (2-4) TF-IDF векторизация
    ├─ Cosine similarity по ЕСН
    └─ Активация при score > 0.1
```

## Структура проекта

```
.
├── api_clients/
│   ├── base.py                  # Базовый класс BaseLLMClient
│   ├── openwebui.py             # OpenWebUI (JWT + API key)
│   ├── mws_gpt.py               # MWS Cloud GPT
│   └── gigachat.py              # GigaChat (OAuth2)
│
├── config/
│   ├── settings.py              # Settings: APIConfig, PromptConfig, MaskGenerationConfig
│   ├── config.yaml              # API ключи, таймауты, mask_generation
│   └── prompts.yaml             # Реестр промптов с keywords, service, model
│
├── core/
│   ├── processor.py             # NomenclatureProcessor (LLM Mode, legacy)
│   ├── automated_processor.py   # AutomatedParametricProcessor (Parametric Mode)
│   ├── parametric_client.py     # ParametricENSClient (Level 6)
│   ├── llm_mask_generator.py    # LLMMaskGenerator (Level 2)
│   ├── mask_database.py         # MaskDatabase + MaskRecord + ConnectionPool
│   ├── auto_validator.py        # AutoValidator (Level 3)
│   ├── database.py              # DatabaseManager (results.db)
│   └── integration.py           # ENSNomenclatureProcessor, build_ens_index, analyze
│
├── parsers/
│   ├── standard_extractor.py    # StandardExtractor (Level 0)
│   ├── cascade.py               # CascadeParser + RegexFastenerParser
│   └── ner_adapter.py           # (legacy)
│
├── ens/
│   ├── loader.py                # ENSLoader (Excel -> объекты)
│   └── indexer.py               # ENSIndex, HybridENSIndex (TF-IDF)
│
├── utils/
│   ├── excel_loader.py          # Загрузка nomenclature.xlsx
│   └── json_export.py           # Экспорт в JSON
│
├── default/
│   └── seed_default_masks.py    # Заполнение БД дефолтными масками
│
├── cli.py                       # CLI интерфейс (Click)
└── requirements.txt
```

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

# Новый раздел: настройки генерации масок (fallback)
mask_generation:
  default_service: "mws"                    # openwebui / mws / gigachat
  default_model: "qwen2.5-72b-instruct"     # fallback-модель
  default_temperature: 0.1
  keyword_match_from_name: true             # искать keywords в полном названии

database:
  path: "cache/results.db"

processing:
  default_workers: 4
  retry_attempts: 3
```

### config/prompts.yaml

```yaml
prompts:
  hardware:
    name: "Крепеж и метизы"
    keywords: ["болт", "гайка", "шуруп", "винт", "заклепка",
               "шпилька", "гвоздь", "штифт", "хомут", "анкер",
               "саморез", "шайба"]
    service: "mws"                          # <-- какой API использовать
    model: "qwen2.5-72b-instruct"           # <-- какая модель
    temperature: 0.1
    system_prompt: "Вы - эксперт по стандартам ГОСТ..."
    file: "prompts/templates/hardware.txt"
    category: "hardware"

  rolledMetal:
    name: "Прокат"
    keywords: ["труба", "швеллер", "уголок", "балка", "профиль",
               "лист", "плита", "рулон", "круг", "квадрат",
               "regex:^ст\\.сорт\\.нерж\\.|ст\\.констр\\.калибр\\."]
    service: "gigachat"                     # <-- другой сервис
    model: "GigaChat-2"
    temperature: 0.1
    system_prompt: "Вы - эксперт по стандартам ГОСТ и прокату..."
    file: "prompts/templates/rolledmetal.txt"
    category: "rolledmetal"
```

**Логика маршрутизации:** `keywords` определяет prompt_id -> оттуда берутся `service`, `model`, `temperature`, `system_prompt`. Если keywords не совпали — fallback на раздел `mask_generation` в `config.yaml`.

## Быстрый старт

### Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Секреты

```bash
mkdir secrets
echo "your_password" > secrets/openwebui_password.txt
echo "your_api_key" > secrets/mws_key.txt
echo "your_credentials" > secrets/gigachat_credentials.txt
```

### Параметрический режим (рекомендуется)

```bash
# 1. Построить индекс ЕСН
python cli.py ens build-index "data/ENS_Крепеж.xlsx" -o cache/ens_hardware.pkl

# 2. Заполнить БД дефолтными масками (быстро, без LLM)
python default/seed_default_masks.py cache/masks.db

# ИЛИ сгенерировать маски через LLM (медленнее, покрывает больше стандартов)
# Только 5 стандартов (быстро, для проверки)
python cli.py generate-masks -d cache/masks.db -i cache/ens_hardware.pkl --llm --limit 5
# Все стандарты (полный прогон)
python cli.py generate-masks -d cache/masks.db -i cache/ens_hardware.pkl --llm


# 3. Обработать файл
python cli.py batch data/nomenclature.xlsx \
    -d cache/masks.db \
    -i cache/ens_hardware.pkl \
    -o results.json
```

### LLM-режим (legacy, детальный разбор)

```bash
# Автоопределение промптов по keywords
python cli.py process data/nomenclature.xlsx --auto

# Явное указание промптов
python cli.py process data/nomenclature.xlsx -p hardware -p rolledMetal
```

## CLI Команды

### ENS и маски

| Команда | Описание |
|---------|----------|
| `ens build-index <excel>` | Построить TF-IDF индекс из Excel ЕСН |
| `ens search <query> -i <index>` | Поиск похожих записей в индексе |
| `ens analyze <excel> -i <index>` | Анализ покрытия файла индексом |
| `generate-masks -d <db> -i <index> --llm` | Авто-генерация масок для всех стандартов |
| `cleanup -d <db> -t 0.5` | Удаление масок с низким score |

### Обработка

| Команда | Описание |
|---------|----------|
| `batch <excel> -d <db> -i <index>` | Пакетная обработка (Parametric Mode) |
| `process-parametric <text> -d <db> -i <index>` | Обработка одной строки |
| `process <excel> --auto` | LLM-обработка (Legacy Mode) |

### Утилиты

| Команда | Описание |
|---------|----------|
| `prompts` | Список промптов с keywords, service, model |
| `models [--api <name>]` | Доступные модели у API-провайдера |
| `detect <text>` | Определить категорию по keywords |
| `stats` | Статистика results.db |
| `export -o <json>` | Экспорт результатов |
| `errors -l <n>` | Последние ошибки |

## Keyword-based маршрутизация

### Как работает

```
Номенклатура: "Болт М16х130.52.019 ГОСТ 7798-70"
    |
    v
StandardExtractor: standard="ГОСТ 7798-70", item_type="болт"
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
Извлечены params: {диаметр: "М16", длина: "130", ...}
Поиск в ЕСН -> ens_code="1000613872", score=0.95
```

### Поддерживаемые типы keywords

| Тип | Пример | Описание |
|-----|--------|----------|
| Подстрока | `болт` | Простое вхождение |
| Glob | `ст.*сорт` | Шаблон с `*` и `?` |
| Regex | `regex:^ст\.сорт\.нерж\.` | Регулярное выражение |

### При item_type="unknown"

Если `StandardExtractor` не определил тип изделия:
1. Пробуем keywords matching на полном наименовании (`name`)
2. Если не совпало — fallback на `prompt_id="hardware"`
3. Если сервис из `prompts.yaml` недоступен — fallback на `mask_generation.default_service`

## Генерация regex-масок через LLM

### Промпт для LLM (LLMMaskGenerator._build_prompt)

```
Ты - эксперт по техническим стандартам и регулярным выражениям.

Создай regex-паттерн для извлечения параметров из номенклатуры
{item_type} по стандарту {standard}.

Примеры:
1. Болт М16х130.52.019 ГОСТ 7798-70
2. Болт М10х60.22.6.019 ГОСТ 7798-70
...

Ответь ТОЛЬКО в формате JSON:
{
  "pattern": "regex with (?P<name>...) named groups",
  "params": ["param1", "param2"],
  "required": ["required1"]
}
```

### Процесс генерации

```bash
python cli.py generate-masks -d cache/masks.db -i cache/ens.pkl --llm
```

Для каждого `(standard, item_type)` из ЕСН:
1. Пропуск, если маска уже активна в БД
2. Нужно минимум 10 примеров
3. LLM генерирует regex с named groups
4. AutoValidator тестирует на примерах из ЕСН
5. Score >= 0.85 -> маска активируется
6. Score < 0.50 -> маска отклоняется

### Настройка fallback для генерации

Чтобы изменить сервис/модель, используемые когда keywords не совпали или prompts.yaml недоступен:

```yaml
# config.yaml
mask_generation:
  default_service: "mws"                     # <-- изменить сервис
  default_model: "qwen2.5-72b-instruct"      # <-- изменить модель
  default_temperature: 0.1
```

## Форматы данных

### Входной Excel (номенклатура)

| Колонка | Описание |
|---------|----------|
| `артикул` | Артикул товара |
| `Краткое наименование` | Наименование с параметрами и стандартом |
| `GUID` | Уникальный идентификатор |

### Выходной JSON (batch)

```json
[
  {
    "text": "Болт М16х130.52.019 ГОСТ 7798-70",
    "level": "parametric_match",
    "success": true,
    "params": {
      "thread_diameter": "М16",
      "length": "130",
      "accuracy_class": "52",
      "coating": "019",
      "standard": "ГОСТ 7798-70"
    },
    "ens_code": "1000613872",
    "confidence": 0.95,
    "processing_time_ms": 45.2
  }
]
```

### Маска в БД (MaskRecord)

| Поле | Описание |
|------|----------|
| `standard` | Стандарт (ГОСТ 7798-70) |
| `item_type` | Тип изделия (болт) |
| `pattern` | Regex с named groups |
| `params` | Список параметров |
| `required` | Обязательные параметры для matching |
| `auto_score` | Score валидации (0-1) |
| `is_active` | Активна ли маска |
| `source` | `llm`, `default`, `manual` |
| `usage_count` | Число использований |

## API клиенты

| Провайдер | Аутентификация | Модели |
|-----------|---------------|--------|
| **OpenWebUI** | JWT (login/password) или API key | `Qwen/Qwen3-14B-AWQ` и др. |
| **MWS Cloud GPT** | API key | `qwen2.5-72b-instruct`, `gpt-oss-120b` |
| **GigaChat** | OAuth2 (Client Credentials) | `GigaChat`, `GigaChat-2`, `GigaChat-Pro` |

## Производительность

| Уровень | Скорость | Точность | Стоимость |
|---------|----------|----------|-----------|
| Regex + MaskDatabase (active) | ~10 000/сек | 90%+ | 0 |
| LLM-генерация маски | ~1-3/сек | — | ~$0.01/запрос |
| ParametricMatch | ~500/сек | 85%+ | 0 |
| TF-IDF fallback | ~200/сек | 60-70% | 0 |

## Требования

- Python 3.9+
- SQLite 3.35+ (WAL mode)
- Зависимости: `pandas`, `openpyxl`, `scikit-learn`, `pyyaml`, `click`, `tqdm`, `numpy`
- API-ключи (опционально, для LLM-генерации масок)

## Troubleshooting

### "settings недоступен, читаем prompts.yaml напрямую"
`settings` не передаётся в `AutomatedParametricProcessor`. В `cli.py` проверьте `settings=settings` при создании процессора.

### "Retry конфигурация: 1 попыток, source: fallback"
Сервис из `prompts.yaml` недоступен. Проверьте:
1. Клиент инициализирован в `cli.py` (mws/gigachat могут быть пропущены)
2. `mask_generation.default_service` в `config.yaml`

### "Модель 'qwen2.5:7b' не найдена"
```bash
# Проверить доступные модели
python cli.py models --api openwebui

# Обновить default_model в config.yaml
api:
  openwebui:
    default_model: "Qwen/Qwen3-14B-AWQ"
```
