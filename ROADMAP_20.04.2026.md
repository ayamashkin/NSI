# ROADMAP: Автоматизированный Параметрический Поиск ENS

## Архитектура Системы

```
┌─────────────────────────────────────────────────────────────┐
│                     ENS PARSING PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│ Level 0: Regex Extractor (standard from text)               │
│     ↓                                                       │
│ Level 1: MaskDatabase (check existing validated masks)      │
│     ↓ если нет                                              │
│ Level 2: AutoMaskGenerator (LLM local/cloud)                │
│     ↓                                                       │
│ Level 3: AutoValidator (test on ENS samples, score ≥ 0.85)  │
│     ↓ если score < 0.85                                     │
│ Level 4: Retry with enhanced prompt OR fallback to TF-IDF   │
│     ↓ если score ≥ 0.85                                     │
│ Level 5: Save to MaskDatabase (auto-approved)               │
│     ↓                                                       │
│ Level 6: ParametricMatching (extract params, compare)       │
│     ↓ если не совпало                                       │
│ Level 7: ENSIndex (TF-IDF fallback)                         │
│     ↓ если не нашли                                         │
│ Level 8: LLM Direct (few-shot with ENS examples)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: MVP (2-3 недели)

### Цель
Базовый параметрический поиск с авто-валидацией для топ-20 стандартов.

### Компоненты

#### 1.1 MaskDatabase
**Требования:**
- SQLite с полями: standard, type, pattern, params, auto_score, test_examples, is_active
- CRUD операции
- Получение лучшей маски по (standard, type)
- Миграции и версионирование

**Принципы:**
- Pattern — уникальный ключ (hash от pattern + standard)
- Auto_score ≥ 0.85 → is_active = True (auto-approve)
- Хранить историю использования (usage_count, last_used)

#### 1.2 StandardExtractor
**Требования:**
- Regex для извлечения стандарта из текста
- Поддержка: ГОСТ, ОСТ, ТУ, ISO, DIN
- Возврат: (standard_type, standard_number, start_pos, end_pos)

**Принципы:**
- Не использовать LLM для определения стандарта (нерелевантно)
- Детерминировано, быстро (< 1ms)

#### 1.3 AutoMaskValidator
**Требования:**
- Сбор N примеров из ENS для данного (standard, type)
- Прогон сгенерированной маски на примерах
- Расчет: score = успешные извлечения / всего примеров
- Минимум: 10 примеров для валидации

**Принципы:**
- Успешное извлечение = все required params найдены
- Тестовые примеры сохраняются в БД (для анализа)
- Score < 0.85 → маска не активируется

#### 1.4 DEFAULT_MASKS
**Требования:**
- Hardcoded маски для топ-20 стандартов (покрывают 80% случаев)
- Auto_score = 1.0 (предполагается идеальной)
- Source = "default"

**Список стандартов:**
- ГОСТ 7798-70, 7795-70 (болты)
- ГОСТ 5915-70, 5927-70 (гайки)
- ОСТ 1 31502-80, 31509-80 (винты)
- ОСТ 1 33035-80 (гайки)
- и др.

### Критерии приемки Phase 1
- [ ] MaskDatabase создана и протестирована
- [ ] DEFAULT_MASKS покрывают 80% записей в тестовом ENS
- [ ] AutoValidator корректно считает score (тесты на известных данных)
- [ ] Интеграция с существующим ENSNomenclatureProcessor

---

## Phase 2: LLM Генерация Масок (2-3 недели)

### Цель
Автоматическая генерация масок для новых/неизвестных стандартов.

### Компоненты

#### 2.1 LLMMaskGenerator
**Требования:**
- Поддержка локальных моделей (Qwen3:7b через Ollama/OpenWebUI)
- Поддержка облачных (GPT-4, Claude) для fallback
- Промпт-инженеринг с few-shot примерами
- Таймаут: 30 сек на генерацию
- Retry logic (3 попытки с разными температурами)

**Принципы:**
- Локальная модель — по умолчанию (дешево, быстро)
- Облачная — если локальная не справилась (3 попытки → fail)
- Промпт должен включать: стандарт, тип, 3 примера из ENS

**Промпт шаблон:**
```
Стандарт: {standard}
Тип: {type}
Примеры из справочника:
1. {example_1}
2. {example_2}
3. {example_3}

Создай regex-паттерн для извлечения параметров.
Ответ в JSON: {{"pattern": "...", "params": [...], "required": [...]}}
```

#### 2.2 Retry Strategy
**Логика:**
```
Attempt 1: temperature=0.1 (детерминировано)
Attempt 2: temperature=0.3 (немного креативности)
Attempt 3: enhanced prompt (больше контекста)
Attempt 4: cloud LLM (GPT-4)
Attempt 5: fallback to TF-IDF (поиск без маски)
```

#### 2.3 MaskQualityGate
**Требования:**
- Score < 0.50 → отклонить, не сохранять
- 0.50 ≤ Score < 0.85 → сохранить как "draft", не использовать
- Score ≥ 0.85 → activate, использовать
- Логирование всех попыток

### Критерии приемки Phase 2
- [ ] LLM генерирует валидный JSON в 95% случаев
- [ ] Средний score сгенерированных масок ≥ 0.80
- [ ] Время генерации < 10 сек (локально)
- [ ] Fallback работает (TF-IDF если LLM не справился)

---

## Phase 3: Оптимизация и Мониторинг (2 недели)

### Цель
Улучшение качества и наблюдаемость системы.

### Компоненты

#### 3.1 Metrics Dashboard
**Метрики:**
- Hit rate по уровням (Level 1: X%, Level 2: Y%, ...)
- Средний score по стандартам
- Время обработки на уровень
- Количество активных масок
- Distribution score (histogram)

**Инструменты:**
- Prometheus/Grafana или просто логи + SQLite
- Ежедневный отчет в Telegram/Slack

#### 3.2 Mask Refinement
**Логика:**
- Если маска используется > 100 раз и средний match score < 0.9 → перегенерировать
- A/B тестирование: старая маска vs новая
- Автоматическое удаление масок с score < 0.5 (cleanup)

#### 3.3 Caching Layer
**Требования:**
- LRU cache для частых (standard, type)
- TTL: 1 час для маски, 24 часа для статистики
- Размер: 1000 масок в памяти

### Критерии приемки Phase 3
- [ ] Hit rate Level 1-2 ≥ 85%
- [ ] Среднее время match < 50ms
- [ ] Мониторинг показывает distribution score
- [ ] Auto-cleanup удаляет < 1% масок в месяц

---

## Phase 4: Интеграция и Production (2 недели)

### Цель
Полная замена/дополнение существующего ENSNomenclatureProcessor.

### Интеграция

#### 4.1 Migration Path
```python
# Старый код (TF-IDF only):
index = ENSIndex.load('ens.pkl')
results = index.search(query)

# Новый код (Parametric + TF-IDF):
processor = ENSNomenclatureProcessor(
    ens_path='ens.pkl',
    mask_db='masks.db',
    use_parametric=True,  # новый флаг
    use_llm_generation=True
)
result = processor.process(query)
```

#### 4.2 Feature Flags
- `USE_PARAMETRIC=true/false` — включить параметрический поиск
- `AUTO_GENERATE_MASKS=true/false` — разрешить LLM генерацию
- `MIN_MASK_SCORE=0.85` — порог активации маски
- `MAX_LLM_RETRIES=3` — количество попыток

#### 4.3 Backward Compatibility
- TF-IDF всегда доступен как fallback
- Существующие результаты не ломаются
- Можно откатить на старую версию (флаг)

### Production Checklist
- [ ] Docker container с Qwen3:7b (если локально)
- [ ] Backup стратегия для masks.db
- [ ] Rate limiting для LLM API
- [ ] Graceful degradation (работает без LLM)
- [ ] Логирование ошибок в Sentry

---

## Технические Требования

### Performance
| Метрика | Требование |
|---------|-----------|
| Regex extraction | < 1ms |
| Mask lookup (SQLite) | < 5ms |
| Auto-validation (10 samples) | < 50ms |
| LLM generation (local) | < 10s |
| Full match (cache hit) | < 10ms |
| Full match (cache miss) | < 100ms |

### Quality Gates
| Gate | Condition |
|------|-----------|
| Mask activation | auto_score ≥ 0.85 |
| Retry trigger | score < 0.50 OR no match |
| Fallback to TF-IDF | no mask after 3 retries |
| Mask deprecation | usage > 100 AND avg_score < 0.7 |

### Data Requirements
- ENS для валидации: минимум 10 примеров на (standard, type)
- Маски хранить: в SQLite + backup JSON
- Логи: retention 30 дней

---

## Риски и Митигация

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| LLM генерирует кривой regex | Средняя | Auto-validator отсеивает (score < 0.85) |
| Долгая генерация маски | Средняя | Таймаут 30с + fallback to TF-IDF |
| SQLite bottleneck | Низкая | Connection pooling, WAL mode |
| LLM недоступен | Низкая | Fallback to DEFAULT_MASKS + TF-IDF |
| Memory leak (кэш) | Низкая | LRU cache с TTL, мониторинг |

---

## Success Metrics (через 3 месяца)

- **Coverage**: 90% стандартов имеют активную маску
- **Accuracy**: 95% параметров извлекаются корректно
- **Speed**: Среднее время обработки < 50ms
- **Cost**: < $100/мес на LLM API (облако)
- **Reliability**: Uptime 99.5%

---

## Итого Timeline

| Phase | Длительность | Результат |
|-------|--------------|-----------|
| Phase 1 (MVP) | 2-3 недели | DEFAULT_MASKS + AutoValidator |
| Phase 2 (LLM) | 2-3 недели | Генерация масок для новых стандартов |
| Phase 3 (Opt) | 2 недели | Мониторинг, оптимизация |
| Phase 4 (Prod) | 2 недели | Интеграция, feature flags |
| **Итого** | **8-10 недель** | Production-ready система |
