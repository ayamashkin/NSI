# Nomenclature Processor + Automated Parametric Search

Система обработки технической номенклатуры с каскадным анализом: regex-парсинг, авто-генерация regex-масок через LLM, параметрическое сопоставление с ЕСН (включая fuzzy matching для текстовых параметров) и TF-IDF fallback.

## Архитектура

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
    |- JSON preprocessing: исправление \\s, \\d, \\w от LLM
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

Level 7: TF-IDF Fallback
    |- Char-ngram (2-4) TF-IDF векторизация
    |- Cosine similarity по ЕСН
    |- Всегда success=False (параметры не извлечены)
    |- ens_code сохраняется только в details как candidate
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Секреты

```bash
mkdir secrets
echo "your_password" > secrets/openwebui_password.txt
echo "your_api_key" > secrets/mws_key.txt
echo "your_credentials" > secrets/gigachat_credentials.txt
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
  - 'наименование_типа'  # дублирует тип_изделия

auto_mapping_patterns:
  'диаметр': 'диаметр'
  'длина': 'длина'
  'исполнение': 'исполнение'
  'покрытие': 'покрытие'
  'класс прочности': 'класс_прочности'
  'марка материала': 'марка_материала'
```

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

### 2. Сгенерировать маски (один стандарт для теста)

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

### 3. Обработать файл

```bash
# Все записи
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/hardware/ens_hardware.pkl -o results.json

# Только успешно распознанные (для загрузки в систему)
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/hardware/ens_hardware.pkl -o results.json --success-only

# С debug-информацией (для анализа)
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/hardware/ens_hardware.pkl -o results.json --include-details
```


```bash
# вызовы для отладки
python cli.py ens build-index "data/_ЕНС_Крепеж_test.xlsx" -o models/hardware2/ens_hardware.pkl    
python cli.py generate-masks -d cache/masks.db -i models/hardware2/ens_hardware.pkl --llm      
python cli.py batch data/nomenclature.xlsx -d cache/masks.db -i models/hardware2/ens_hardware.pkl -o results.json
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

### Отладка

Промпты сохраняются в `prompts/debug/`:
```
prompts/debug/
|-- БОЛТ_ГОСТ_7798-70.txt          # промпт
|-- БОЛТ_ГОСТ_7798-70_a1.txt       # ответ попытки 1
|-- БОЛТ_ГОСТ_7798-70_a2.txt       # ответ попытки 2 (retry)
|-- БОЛТ_ГОСТ_7798-70_failed_a1.txt # при ошибке парсинга
```

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

## API клиенты

| Провайдер | Аутентификация | Модели |
|-----------|---------------|--------|
| **OpenWebUI** | JWT (login/password) или API key | `Qwen/Qwen3-14B-AWQ` и др. |
| **MWS Cloud GPT** | API key | `qwen2.5-72b-instruct`, `gpt-oss-120b` |
| **GigaChat** | OAuth2 (Client Credentials) | `GigaChat`, `GigaChat-2`, `GigaChat-Pro` |

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

## Требования

- Python 3.9+
- SQLite 3.35+ (WAL mode)
- Зависимости: `pandas`, `openpyxl`, `scikit-learn`, `pyyaml`, `click`, `tqdm`, `numpy`
- API-ключи (опционально, для LLM-генерации масок)
