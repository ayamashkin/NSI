# Nomenclature Processor + Automated Parametric Search

Система обработки номенклатуры с двумя режимами работы:
1. **LLM Mode** — извлечение параметров через LLM (legacy)
2. **Parametric Mode** — каскадный парсинг regex → ENS → LLM с авто-генерацией масок

## 🏗️ Архитектура

```
Level 0: StandardExtractor (regex) — извлечение стандарта
    ↓
Level 1: MaskDatabase (SQLite) — проверка существующих масок
    ↓ если нет маски
Level 2: LLMMaskGenerator — генерация regex через LLM
    ↓
Level 3: AutoValidator — валидация на примерах из ЕСН (score ≥ 0.85)
    ↓
Level 5: Save to MaskDatabase (auto-approved)
    ↓
Level 6: ParametricMatching — извлечение параметров
    ↓ если не совпало
Level 7: TF-IDF Fallback — нечеткий поиск по ЕСН
```

## 📁 Структура проекта

```
nomenclature-processor/
├── config/
│   ├── settings.py              # Настройки Pydantic
│   ├── config.yaml              # API ключи, таймауты
│   └── prompts.yaml             # Реестр промптов и категорий
├── core/
│   ├── database.py              # SQLite для результатов (results.db)
│   ├── processor.py             # LLM процессор (legacy)
│   ├── integration.py           # Утилиты ENS
│   ├── automated_processor.py   # Параметрический процессор (Level 0-8)
│   ├── parametric_client.py     # Параметрическое сопоставление
│   ├── mask_database.py         # MaskDatabase (SQLite + pool)
│   ├── llm_mask_generator.py    # Генерация regex через LLM
│   └── auto_validator.py        # Авто-валидация масок
├── api_clients/
│   ├── base.py                  # Базовый класс
│   ├── openwebui.py             # OpenWebUI клиент (JWT/API key)
│   ├── mws_gpt.py               # MWS Cloud GPT
│   └── gigachat.py              # GigaChat (Enterprise + Cloud)
├── parsers/
│   ├── cascade.py               # RegexFastenerParser + CascadeParser
│   ├── standard_extractor.py    # Level 0: ГОСТ/ОСТ extraction
│   └── ner_adapter.py           # (legacy, не используется)
├── ens/
│   ├── loader.py                # ENSLoader (Excel → объекты)
│   └── indexer.py               # ENSIndex (TF-IDF)
├── utils/
│   ├── excel_loader.py          # Загрузка nomenclature.xlsx
│   └── json_export.py           # Экспорт в JSON
├── default/
│   └── seed_default_masks.py    # Заполнение БД дефолтными масками
├── cli.py                       # CLI интерфейс (все команды)
├── cli.py                       # CLI интерфейс (все команды)
├── cli.py                       # CLI интерфейс (все команды)
└── requirements.txt
```

## ⚡ Быстрый старт

### 1. Установка
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

### 2. Конфигурация

Создайте `config/config.yaml`:

```yaml
api:
  openwebui:
    base_url: "https://webui.game73.ru/api"
    username: "user@example.com"
    password_file: "secrets/openwebui_password.txt"
    timeout: 180
    default_model: "qwen3:30b"
  mws:
    base_url: "https://api.gpt.mws.ru/"
    api_key_file: "secrets/mws_key.txt"
    timeout: 120
  gigachat:
    base_url: "https://gigachat.devices.sberbank.ru/api/v1"
    api_key_file: "secrets/gigachat_credentials.txt"
    scope: "GIGACHAT_API_PERS"
    timeout: 120

database:
  path: "cache/results.db"

processing:
  default_workers: 4
```

### 3. Режим 1: Параметрический поиск (новый, рекомендуется)

```bash
# Шаг 1: Построить индекс ЕСН
python cli.py ens build-index "data/_ЕНС_Крепеж_24.03.2026.xlsx" -o models/hardware/ens_hardware.pkl

# Шаг 2: Заполнить БД дефолтными масками (быстро) ИЛИ сгенерировать через LLM
python default/seed_default_masks.py cache/masks.db
# ИЛИ
python cli.py generate-masks -d cache/masks.db -i models/hardware/ens_hardware.pkl --llm

# Шаг 3: Обработать файл
python cli.py batch data/nomenclature.xlsx \
    -d cache/masks.db \
    -i models/hardware/ens_hardware.pkl \
    -o results.json
```

### 4. Режим 2: LLM обработка (legacy)

```bash
python cli.py process data/nomenclature.xlsx --auto
```

## 🚀 CLI Команды

### ENS и Маски (Новое)

| Команда | Описание | Пример |
|---------|----------|--------|
| `ens build-index` | Построить TF-IDF индекс из Excel ЕСН | `python cli.py ens build-index ENS.xlsx -o ens.pkl` |
| `ens search` | Поиск похожих записей в индексе | `python cli.py ens search "Болт М12" -i ens.pkl` |
| `ens analyze` | Анализ покрытия файла индексом | `python cli.py ens analyze input.xlsx -i ens.pkl` |
| `generate-masks` | Авто-генерация масок для всех стандартов | `python cli.py generate-masks -d masks.db -i ens.pkl --llm` |
| `cleanup` | Удаление масок с низким score | `python cli.py cleanup -d masks.db -t 0.5` |

### Обработка номенклатуры

| Команда | Описание | Пример |
|---------|----------|--------|
| `batch` | Пакетная обработка с масками | `python cli.py batch input.xlsx -d masks.db -i ens.pkl` |
| `process` | LLM обработка (legacy) | `python cli.py process input.xlsx --auto` |
| `stats` | Статистика масок или результатов | `python cli.py stats -d masks.db` |
| `export` | Экспорт результатов в JSON | `python cli.py export -o results.json --structure by_code` |
| `errors` | Просмотр ошибок | `python cli.py errors -l 20` |
| `detect` | Определить категорию | `python cli.py detect "Болт М12"` |
| `models` | Список моделей API | `python cli.py models --api openwebui` |
| `prompts` | Список доступных промптов | `python cli.py prompts` |

## 📊 Форматы данных

### Входной Excel (номенклатура)

| артикул | Краткое наименование | GUID |
|---------|---------------------|------|
| 001 | Болт (2)-8-32-Кд-ОСТ 1 31133-80, | guid-1 |
| 002 | Винт (3)-6-46-Кд-ОСТ 1 31502-80, | guid-2 |

### Выходной JSON (batch)

```json
[
  {
    "text": "Болт (2)-8-32-Кд-ОСТ 1 31133-80,",
    "level": "parametric_match",
    "success": true,
    "params": {
      "исполнение": "2",
      "диаметр": 8,
      "длина": 32,
      "покрытие": "Кадмирование",
      "стандарт": "ОСТ 1 31133-80"
    },
    "ens_code": "1000613872",
    "confidence": 0.95,
    "processing_time_ms": 45.2
  }
]
```

## 🔧 Примеры использования

### Генерация масок для нового стандарта

```bash
# Если в файле nomenclature встречаются неизвестные стандарты:
python cli.py generate-masks \
    -d cache/masks.db \
    -i models/hardware/ens_hardware.pkl \
    --llm \
    --min-score 0.85
```

### Проверка покрытия перед обработкой

```bash
python cli.py ens analyze data/nomenclature.xlsx \
    -i models/hardware/ens_hardware.pkl \
    -s 200
# Вывод: 85% разбирается regex, 15% требует LLM
```

### Экспорт с фильтрацией

```bash
python cli.py export \
    -o results.json \
    --structure by_code \
    --include-raw \
    --status completed
```

## 🎯 Производительность

| Режим | Скорость | Точность | Стоимость |
|-------|----------|----------|-----------|
| Regex + БД | ~10,000/сек | 90% | $0 |
| С LLM генерацией | ~50/сек | 95% | ~$0.01/запись |
| TF-IDF fallback | ~500/сек | 70% | $0 |

## 🛡️ Обработка ошибок

| Level | Условие | Действие |
|-------|---------|----------|
| 0 | Не найден стандарт | TF-IDF fallback |
| 1 | Нет маски в БД | Переход к генерации (если --llm) |
| 2 | LLM не отвечает | Retry с другой температурой |
| 3 | Score < 0.50 | Отклонение маски |
| 6 | Параметры не совпали | TF-IDF fallback |

## 📝 Требования

- Python 3.9+
- SQLite 3.35+ (WAL mode)
- Зависимости: pandas, openpyxl, scikit-learn, pyyaml, pydantic, click, tqdm
- API ключи (опционально для generate-masks)

## 🔌 API Интеграции

### OpenWebUI
- Поддержка JWT (username/password) и API key
- Автоматический retry при 401
- Модели: qwen3:30b, qwen2.5:72b и др.

### MWS Cloud GPT
- Модели: qwen2.5-72b-instruct, gpt-oss-120b
- Стандартная авторизация по ключу

### GigaChat
- Enterprise: demo.gigaenterprise.ai (логин/пароль)
- Cloud: giga.chat (OAuth2)
- Модели: GigaChat, GigaChat-Pro, GigaChat-Max

## 🆘 Troubleshooting

### "Model not found" в OpenWebUI
```bash
# Проверить доступные модели
python cli.py models --api openwebui

# Исправить config.yaml
default_model: "qwen2.5:7b"  # вместо qwen3:30b
```

### Пустые params в результатах
```bash
# БД масок пуста — заполнить
python default/seed_default_masks.py cache/masks.db
```

### Долгая генерация масок
```bash
# Использовать только hardcoded маски (без LLM)
python default/seed_default_masks.py cache/masks.db

# Или проверить доступность API
python cli.py models --api openwebui
```
