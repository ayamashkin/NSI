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
│   ├── settings.py          # Конфигурация API и путей
│   └── prompts.yaml         # Реестр промптов по категориям
├── core/
│   ├── models.py            # Pydantic модели данных
│   ├── database.py          # SQLite manager с upsert
│   └── processor.py         # Основной движок обработки
├── api_clients/
│   ├── base.py              # Абстрактный класс клиента
│   ├── openwebui.py         # Клиент для OpenWebUI
│   └── mws_gpt.py           # Клиент для MWS Cloud GPT
├── prompts/
│   ├── registry.py          # Реестр промптов
│   └── templates/           # Файлы промптов (.txt)
├── utils/
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
- `krepezh_v1.txt` - для крепежных изделий
- `eri_v1.txt` - для электрорадиоизделий
- `materials_v1.txt` - для материалов
- `purchased_v1.txt` - для покупных изделий

Настройте `config/prompts.yaml` для регистрации промптов.

### 4. Подготовка данных

Excel файл должен содержать колонки:
- `артикул` - уникальный код изделия
- `Краткое наименование` - наименование для анализа
- `GUID` - внутренний идентификатор

## 🚀 Использование

### CLI команды

#### Просмотр доступных промптов
```bash
python cli.py prompts
```

#### Обработка с автоопределением категорий
```bash
python cli.py process nomenclature.xlsx --auto --api openwebui -w 8
```

#### Обработка конкретными промптами
```bash
python cli.py process nomenclature.xlsx -p krepezh_v1 -p krepezh_v2 --api openwebui
```

#### Использование MWS Cloud GPT
```bash
python cli.py process nomenclature.xlsx --auto --api mws -w 4
```

#### Форсированная перезапись результатов
```bash
python cli.py process nomenclature.xlsx --auto --api openwebui -f
```

#### Проверка категории для наименования
```bash
python cli.py detect "Болт М12х50 ГОСТ 7798-70"
```

#### Экспорт результатов
```bash
# Плоская структура
python cli.py export -o results.json --structure flat

# Группировка по артикулам
python cli.py export -o results_by_code.json --structure by_code
```

#### Просмотр статистики
```bash
python cli.py stats
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
    "prompt_id": "krepezh_v1",
    "category": "krepezh",
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
    category: "krepezh"
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
