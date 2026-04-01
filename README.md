# Nomenclature Processor

Система автоматической обработки номенклатуры с использованием LLM (Large Language Models) для извлечения технических параметров изделий.

## 🎯 Назначение

Система предназначена для:
- Автоматической классификации номенклатуры по категориям (крепеж, ЭРИ, материалы, покупные изделия)
- Извлечения структурированных технических параметров из неструктурированных наименований
- Пакетной обработки больших объемов данных (десятки тысяч позиций)
- Интеграции с локальными LLM (OpenWebUI) и облачными API (MWS Cloud GPT)

## 🏗️ Архитектура

```
nomenclature-processor/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Конфигурация API и путей
│   └── prompts.yaml         # Реестр промптов по категориям
├── core/
│   ├── __init__.py
│   ├── models.py            # Pydantic модели данных
│   ├── database.py          # SQLite manager с upsert
│   └── processor.py         # Основной движок обработки
├── api_clients/
│   ├── __init__.py
│   ├── base.py              # Абстрактный класс клиента
│   ├── openwebui.py         # Клиент для OpenWebUI
│   └── mws_gpt.py           # Клиент для MWS Cloud GPT
├── prompts/
│   ├── __init__.py
│   ├── registry.py          # Реестр промптов
│   └── templates/           # Файлы промптов (.txt)
│       ├── hardware.txt
│       ├── rolledmetal.txt
│       └── ....txt
├── utils/
│   ├── __init__.py
│   ├── excel_loader.py      # Загрузка Excel
│   └── json_export.py       # Экспорт результатов
├── cli.py                   # CLI интерфейс
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
.venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Конфигурация

Создайте файл `.env` в корне проекта:

```env
# OpenWebUI (локальные модели)
OPENWEBUI_URL=http://localhost:3000
OPENWEBUI_API_KEY=your_openwebui_key

# MWS Cloud GPT (облачный API)
MWS_URL=https://mws.ru/api/v1
MWS_API_KEY=your_mws_api_key

# База данных
DATABASE_PATH=results.db
```

### 3. Подготовка промптов

Поместите файлы промптов в директорию `prompts/templates/`:
- `hardware.txt` - для крепежных изделий и метизов
- `rolledmetal.txt` - для проката


Настройте `config/prompts.yaml` для регистрации промптов.

### 4. Подготовка данных

Excel файл должен содержать колонки:
- `артикул` - уникальный код изделия
- `Краткое наименование` - наименование для анализа
- `GUID` - внутренний идентификатор

## 🚀 Использование

### CLI команды

#### Основные команды

| Команда | Описание                         | Пример |
|---------|----------------------------------|--------|
| `prompts` | Список доступных промптов        | `python cli.py prompts` |
| `process` | Обработка Excel файла            | `python cli.py process data.xlsx -p hardware` |
| `export` | Экспорт результатов в JSON       | `python cli.py export -o results.json` |
| `stats` | Статистика обработки             | `python cli.py stats` |
| `errors` | Просмотр ошибок                  | `python cli.py errors -l 20` |
| `detect` | Определить категорию             | `python cli.py detect "Болт М12х50"` |
| `models` | Вывести список доступных моделей | `python cli.py models` |

#### Подробное описание

##### `prompts` — Список промптов
```bash
python cli.py prompts
```
Выводит список всех настроенных промптов с указанием:
- Название и категория
- Используемый сервис API (openwebui/mws)
- Модель LLM
- Ключевые слова для автоопределения

##### `process` — Обработка Excel
```bash
# Автоопределение промптов
python cli.py process data/nomenclature.xlsx --auto

# Конкретный промпт
python cli.py process data.xlsx -p hardware

# Несколько промптов
python cli.py process data.xlsx -p hardware -p rolledmetal

# С указанием API (проверяет соответствие сервиса в промпте)
python cli.py process data.xlsx -p hardware --api mws

# Параллельная обработка (4 workers)
python cli.py process data.xlsx --auto -w 4

# Перезапись существующих результатов
python cli.py process data.xlsx -p hardware -f
```

**Опции:**
- `-p, --prompt` — ID промпта (можно несколько)
- `-a, --auto` — Автоопределение подходящих промптов по ключевым словам
- `--api` — Принудительный выбор API (openwebui/mws)
- `-w, --workers` — Количество параллельных workers
- `-f, --force` — Перезаписать существующие результаты

##### `errors` — Просмотр ошибок
```bash
# Последние 10 ошибок
python cli.py errors

# Последние 20 ошибок
python cli.py errors -l 20

# Ошибки конкретного промпта
python cli.py errors -p hardware
```

Выводит детали ошибок:
- Артикул и наименование
- Промпт и сервис API
- Текст ошибки
- Первые 300 символов ответа API (если есть)

##### `export` — Экспорт результатов
```bash
# Плоская структура (список)
python cli.py export -o results.json --structure flat

# Группировка по артикулам
python cli.py export -o results_by_code.json --structure by_code

# Группировка по категориям
python cli.py export -o results.json --structure by_category

# Группировка по промптам
python cli.py export -o results.json --structure by_prompt
```

##### `stats` — Статистика
```bash
python cli.py stats
```
Показывает:
- Общее количество записей
- Распределение по статусам (completed/ignored/error)
- Распределение по категориям
- Распределение по сервисам API

##### `detect` — Определение категории
```bash
python cli.py detect "Болт М12х50 ГОСТ 7798-70"
```
Проверяет, какие промпты подходят для указанного наименования по ключевым словам.

#### Обработка ошибок

При несоответствии сервиса API:
```bash
$ python cli.py process data.xlsx -p hardware --api openwebui
❌ Несоответствие сервиса API:
   Промпт 'hardware' использует сервис 'mws', но выбран 'openwebui'

💡 Варианты:
   1. Используйте --api mws
   2. Или не указывайте --api для использования всех промптов
```

После обработки с ошибками:
```bash
$ python cli.py process data.xlsx -p hardware
...
❌ Ошибок: 5
💡 Просмотр ошибок: python cli.py errors
```
#### Список доступных моделей

Запрос списка моделей
```bash
# Все сервисы
python cli.py models

# Конкретный сервис
python cli.py models --api openwebui
python cli.py models --api mws
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
    "model_used": "qwen-30b",
    "api_source": "OpenWebUIClient"
  }
]
```

### Выходной JSON (by_code)

```json
{
  "001": {
    "article": "001",
    "name": "Болт М12х50 ГОСТ 7798-70",
    "guid": "guid-1",
    "prompts": {
      "krepezh_v1": {
        "status": "completed",
        "category": "krepezh",
        "display_name": "Болт М12х50 ГОСТ 7798-70",
        "params": [...],
        "processed_at": "2024-01-15T10:30:00"
      }
    }
  }
}
```

## 🔧 Конфигурация промптов

Файл `config/prompts.yaml`:

```yaml
prompts:
  krepezh_v1:
    name: "Крепеж - полный разбор ГОСТ"
    file: "prompts/templates/krepezh_v1.txt"
    category: "hardware"
    keywords: ["болт", "гайка", "шуруп", "винт", "шайба", "заклепка"]
    model: "qwen-30b"
    temperature: 0.1

  eri_v1:
    name: "Электрорадиоизделия"
    file: "prompts/templates/eri_v1.txt"
    category: "eri"
    keywords: ["резистор", "конденсатор", "транзистор", "диод", "микросхема"]
    model: "qwen-30b"
    temperature: 0.1
```

### Правила автоопределения категории

Система автоматически определяет категорию по ключевым словам в наименовании:
- **Крепеж**: болт, гайка, шуруп, винт, шайба, заклепка, шпилька, гвоздь
- **ЭРИ**: резистор, конденсатор, транзистор, диод, микросхема, чип
- **Материалы**: сталь, алюминий, медь, латунь, пластик, резина, лента
- **Покупные**: подшипник, сальник, ремень, цепь, шланг, клапан

## 🔄 Upsert семантика

База данных SQLite гарантирует уникальность записей по составному ключу `(article, prompt_id)`:
- Если запись существует - она обновляется
- Если записи нет - она создается
- Это позволяет безопасно перезапускать обработку

## 🛡️ Обработка ошибок

Система обрабатывает следующие сценарии:
- **IGNORED**: Номенклатура не соответствует категории промпта
- **COMPLETED**: Успешная обработка
- **ERROR**: Ошибка API или парсинга ответа

## 📈 Производительность

- **Параллельная обработка**: Настраиваемое количество workers (параметр `-w`)
- **Кэширование**: SQLite база для хранения промежуточных результатов
- **Прогресс-бар**: Визуализация процесса обработки
- **Рекомендации**: 
  - Для OpenWebUI (локальные модели): 4-8 workers
  - Для MWS Cloud GPT: 2-4 workers (лимиты API)

## 🔌 API Интеграции

### OpenWebUI
- Поддержка локальных моделей (Qwen, Llama, etc.)
- Конфигурация через переменные окружения
- Автоматический парсинг JSON из markdown

### MWS Cloud GPT
- Облачный API от MWS
- Модели GPT-4 и аналоги
- Требуется API ключ от MWS

## 🧪 Тестирование

```bash
# Тест определения категории
python cli.py detect "Болт М12х1.25-6gx100.58 ГОСТ 7795-70"

# Тест с небольшой выборкой
python cli.py process test_sample.xlsx --auto -w 2

# Проверка статистики после теста
python cli.py stats
```

## 📝 Логирование

Логи выводятся в консоль с уровнем INFO:
- Загрузка данных
- Прогресс обработки
- Ошибки API
- Результаты сохранения

## 🤝 Разработка

### Добавление новой категории

1. Создать файл промпта в `prompts/templates/`
2. Добавить запись в `config/prompts.yaml`
3. Указать ключевые слова для автоопределения

### Добавление нового API клиента

1. Создать класс в `api_clients/`, наследующий `BaseLLMClient`
2. Реализовать методы `complete()` и `health_check()`
3. Добавить выбор в CLI

## 📋 Требования

- Python 3.9+
- Доступ к API (OpenWebUI или MWS)
- Excel файл с номенклатурой

## 📄 Лицензия

MIT License

## 🆘 Поддержка

При возникновении проблем:
1. Проверьте доступность API: `python cli.py stats`
2. Проверьте конфигурацию в `.env`
3. Проверьте формат Excel файла
4. Обратитесь к документации API провайдера

---

**Версия**: 1.0.0  
**Дата обновления**: 2024
