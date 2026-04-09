# Nomenclature Processor

Система автоматической обработки номенклатуры с использованием LLM (Large Language Models) для извлечения технических параметров изделий.

## 🎯 Назначение

Система предназначена для:
- Автоматической классификации номенклатуры по категориям (крепеж, ЭРИ, материалы, покупные изделия)
- Извлечения структурированных технических параметров из неструктурированных наименований
- Пакетной обработки больших объемов данных (десятки тысяч позиций)
- Интеграции с локальными LLM (OpenWebUI), облачными API (MWS Cloud GPT, GigaChat)

## 🏗️ Архитектура

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
│   └── registry.py              # Реестр промптов
├── api_clients/
│   ├── __init__.py
│   ├── base.py                  # Абстрактный класс клиента
│   ├── openwebui.py             # Клиент для OpenWebUI API
│   ├── mws_gpt.py               # Клиент для MWS Cloud GPT API
│   └── gigachat.py              # Клиенты для GigaChat API (Enterprise + Cloud)
├── utils/
│   ├── __init__.py
│   ├── excel_loader.py          # Загрузка Excel через pandas/openpyxl
│   └── excel_loader_simple.py   # Загрузка Excel только через openpyxl
├── secrets/                     # Учетные данные (не в git)
├── prompts/templates/           # Файлы промптов (.txt)
├── logs/                        # Директория для логов
├── cli.py                       # CLI интерфейс (точка входа)
├── results.db                   # SQLite база данных (создается автоматически)
├── requirements.txt
└── README.md
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
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Конфигурация API

Создайте директорию `secrets/` и файлы с учетными данными:

**OpenWebUI:**
```bash
echo "your_openwebui_key" > secrets/openwebui_key.txt
```

**MWS Cloud GPT:**
```bash
echo "your_mws_api_key" > secrets/mws_key.txt
```

**GigaChat Cloud (giga.chat):**
```bash
echo "your_authorization_key" > secrets/gigachat_credentials.txt
```

### 3. Настройка config.yaml

Основная конфигурация в `config/config.yaml`:

```yaml
api:
  openwebui:
    base_url: "https://webui.game73.ru/api"
    api_key_file: "secrets/openwebui_key.txt"
    timeout: 120
    default_model: "qwen3-30b"

  mws:
    base_url: "https://api.gpt.mws.ru/"
    api_key_file: "secrets/mws_key.txt"
    timeout: 120
    default_model: "qwen2.5-72b-instruct"

  gigachat:
    base_url: "https://gigachat.devices.sberbank.ru/api/v1"
    api_key_file: "secrets/gigachat_credentials.txt"     # Authorization Key
    scope: "GIGACHAT_API_PERS"  # GIGACHAT_API_PERS, GIGACHAT_API_B2B или GIGACHAT_API_CORP
    timeout: 120
    default_model: "GigaChat"
```

### 4. Подготовка промптов

Поместите файлы промптов в `prompts/templates/` и настройте `config/prompts.yaml`:

```yaml
prompts:
  hardware:
    name: "Крепеж - полный разбор ГОСТ"
    file: "prompts/templates/hardware.txt"
    category: "hardware"
    keywords: ["болт", "гайка", "шуруп", "винт", "шайба", "заклепка"]
    service: "mws"  # или "openwebui","gigachat"
    model: "gigachat-pro"
    temperature: 0.1
```

### 5. Подготовка данных

Excel файл должен содержать колонки:
- `артикул` — уникальный код изделия
- `Краткое наименование` — наименование для анализа
- `GUID` — внутренний идентификатор

## 🚀 Использование

### CLI команды

| Команда | Описание | Пример                                          |
|---------|----------|-------------------------------------------------|
| `prompts` | Список доступных промптов | `python cli.py prompts`                         |
| `process` | Обработка Excel файла | `python cli.py process data.xlsx -p hardware` |
| `export` | Экспорт результатов в JSON | `python cli.py export -o results.json`          |
| `stats` | Статистика обработки | `python cli.py stats`                           |
| `errors` | Просмотр ошибок | `python cli.py errors -l 20`                    |
| `detect` | Определить категорию | `python cli.py detect "Болт М12х50"`            |
| `models` | Список доступных моделей | `python cli.py models --api wms`                |

### Примеры использования

#### Список промптов
```bash
python cli.py prompts
```

#### Обработка Excel
```bash
# Автоопределение промптов по ключевым словам
python cli.py process data/nomenclature.xlsx --auto

# Конкретный промпт
python cli.py process data.xlsx -p hardware

# Несколько промптов
python cli.py process data.xlsx -p hardware -p eri_v1

# С указанием API (проверяет соответствие сервиса в промпте)
python cli.py process data.xlsx -p hardware --api gigachat 

# Параллельная обработка
python cli.py process data.xlsx --auto -w 4

# Перезапись существующих результатов
python cli.py process data.xlsx -p hardware -f
```

#### Просмотр ошибок
```bash
# Последние 10 ошибок
python cli.py errors

# Последние 20 ошибок
python cli.py errors -l 20

# Ошибки конкретного промпта
python cli.py errors -p hardware
```

#### Экспорт результатов
```bash
# Плоская структура
python cli.py export -o results.json --structure flat

# Группировка по артикулам
python cli.py export -o results.json --structure by_code

# Группировка по категориям
python cli.py export -o results.json --structure by_category

# Экспорт только результатов по конкретному промпту
python cli.py export -o krepezh_results.json --prompt krepezh_v1

# Экспорт только успешных результатов по промпту
python cli.py export -o completed.json --prompt krepezh_v1 --status completed

# Экспорт с группировкой по промптам
python cli.py export -o by_prompt.json --structure by_prompt

# Экспорт с текстом промпта
python cli.py export -o krepezh_results.json --prompt hardware_washer --include-prompt

# Экспорт с текстом промпта и raw_response
python cli.py export -o krepezh_results.json --prompt hardware_washer --include-prompt --include-raw
```

#### Список моделей API
```bash
# Все сервисы
python cli.py models

# Конкретный сервис
python cli.py models --api openwebui
python cli.py models --api mws
python cli.py models --api gigachat
```

## 📊 Форматы данных

### Входной Excel

| артикул | Краткое наименование | GUID |
|---------|---------------------|------|
| 001 | Болт М12х50 ГОСТ 7798-70 | guid-1 |
| 002 | Резистор С2-29В 100 Ом | guid-2 |
| 003 | Лист стальной 2 мм | guid-3 |

### Выходной JSON (flat)

```json
[
  {
    "article": "001",
    "name": "Болт М12х50 ГОСТ 7798-70",
    "guid": "guid-1",
    "prompt_id": "hardware",
    "category": "hardware",
    "status": "completed",
    "display_name": "Болт М12х50 ГОСТ 7798-70",
    "params": [
      {
        "name": "Номинальный диаметр резьбы",
        "value": "12",
        "default": "",
        "um": "мм"
      },
      {
        "name": "Длина",
        "value": "50",
        "default": "",
        "um": "мм"
      }
    ],
    "processed_at": "2024-01-15T10:30:00",
    "model_used": "gigachat-pro",
    "api_source": "gigachat"
  }
]
```

## 🔧 Конфигурация промптов

### Правила автоопределения категории

Система поддерживает три типа ключевых слов:

1. **Обычные строки** — подстроковое совпадение
   ```yaml
   keywords: ["болт", "гайка"]
   ```

2. **Glob-шаблоны** — с wildcards `*` и `?`
   ```yaml
   keywords: ["винт*м", "шайба???"]
   ```

3. **Регулярные выражения** — префикс `regex:` или `re:`
   ```yaml
   keywords: ["regex:болт.*гост", "re:гайка.*м\\d+"]
   ```

## 🔌 API Интеграции

### OpenWebUI
- Локальные модели через OpenWebUI API
- Поддержка моделей Qwen, Llama и др.
- Автоматический парсинг JSON из markdown code blocks

### MWS Cloud GPT
- Облачный API от MWS
- Модели: qwen2.5-72b-instruct, gpt-oss-120b и др.
- Стандартная авторизация по API ключу


### GigaChat Cloud (giga.chat)
- Облачный API Сбера для физических лиц и B2B
- **Авторизация:** OAuth2 с Authorization Key (credentials)
- **Scope:** GIGACHAT_API_PERS, GIGACHAT_API_B2B или GIGACHAT_API_CORP
- Модели: GigaChat, GigaChat-Pro, GigaChat-Max

## 🔄 Upsert семантика

База данных SQLite гарантирует уникальность записей по составному ключу `(article, prompt_id)`:
- Если запись существует — она обновляется
- Если записи нет — она создается
- Повторный запуск безопасен и не создает дубликатов

## 🛡️ Обработка ошибок

| Статус | Описание | Действие при повторном запуске |
|--------|----------|-------------------------------|
| **COMPLETED** | Успешная обработка | Берется из кэша (если `cache_completed_only: true`) |
| **IGNORED** | Номенклатура не соответствует категории | Перепроверяется (если `retry_ignored: true`) |
| **ERROR** | Ошибка API или парсинга | Перепроверяется (если `retry_errors: true`) |

## 📈 Производительность

- **Параллельная обработка:** настраиваемое количество workers
- **Кэширование:** SQLite с UPSERT для промежуточных результатов
- **Рекомендации:**
  - OpenWebUI (локальные модели): 4-8 workers
  - MWS Cloud GPT: 2-4 workers
  - GigaChat API: 2-4 workers (лимиты API)

## 🧪 Тестирование

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

## 📝 Логирование

Логи сохраняются в `logs/processor.log` и выводятся в консоль:
- Уровень логирования настраивается в `config.yaml` (`logging.level`)
- Ротация логов: 5 файлов по 10MB каждый

## 🤝 Разработка

### Добавление нового API клиента

1. Создать класс в `api_clients/`, наследующий `BaseLLMClient`
2. Реализовать методы: `complete()`, `health_check()`, `get_models()`
3. Добавить инициализацию в `core/processor.py::_init_api_clients()`
4. Добавить в CLI в `cli.py` (команда `models`)
5. Обновить `config/config.yaml` с настройками нового API

### Добавление новой категории

1. Создать файл промпта в `prompts/templates/`
2. Добавить запись в `config/prompts.yaml`
3. Указать ключевые слова для автоопределения
4. Выбрать подходящий сервис API и модель

## 📋 Требования

- Python 3.9+
- Зависимости: `requests`, `pyyaml`, `pydantic`, `click`, `tqdm`, `openpyxl`, `pandas` (опционально)
- Доступ к хотя бы одному API (OpenWebUI, MWS, GigaChat)
- Excel файл с номенклатурой

## 🆘 Поддержка

При возникновении проблем:
1. Проверьте доступность API: `python cli.py models --api <name>`
2. Проверьте файлы учетных данных в `secrets/`
3. Проверьте формат Excel файла (колонки: артикул, Краткое наименование, GUID)
4. Просмотрите логи: `logs/processor.log`
5. Проверьте ошибки: `python cli.py errors`
