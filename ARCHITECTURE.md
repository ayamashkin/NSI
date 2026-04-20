
# Архитектура: Автоматизированный Параметрический Поиск

## Схема Потока Данных (Упрощённая)

```
ВХОД: "Болт М12х50 ГОСТ 7798-70"
  ↓
[Level 0] StandardExtractor (regex)
  ↓ standard="ГОСТ 7798-70", type="болт"
[Level 1] MaskDatabase Lookup
  ↓
  ├─ Найдена? → [Level 6] ParametricMatch → РЕЗУЛЬТАТ
  │
  └─ Не найдена? → [Level 2] LLM Generation
                      ↓
                   [Level 3] AutoValidator
                      ↓
              Score >= 0.85? → [Level 5] Save → [Level 6] Match
              Score < 0.85?  → [Level 4] Retry/Fallback
                                    ↓
                               [Level 7] TF-IDF Fallback
```

## Компоненты

### 1. MaskDatabase (SQLite)
Таблица masks:
- id, standard, item_type, pattern, params, required
- auto_score (0.0-1.0), is_active, source
- usage_count, created_at

### 2. AutoValidator
1. Берет 10+ примеров из ENS для (standard, type)
2. Применяет pattern на каждом
3. Считает: score = успешные / всего
4. Если >= 0.85 → activate mask

### 3. Retry Strategy
- Попытка 1: Qwen3:7b, temp=0.1
- Попытка 2: Qwen3:7b, temp=0.3  
- Попытка 3: GPT-4 (cloud)
- Fallback: TF-IDF

## Пороги
- Mask activation: score >= 0.85
- Retry trigger: score < 0.50
- Cache TTL: 1 час

## Timeline
- Phase 1 (MVP): 2-3 недели
- Phase 2 (LLM): 2-3 недели
- Phase 3 (Opt): 2 недели
- Phase 4 (Prod): 2 недели
- Итого: 8-10 недель
