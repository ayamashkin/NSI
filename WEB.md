# ENS Verification Web Interface

Web-интерфейс для пакетной обработки номенклатуры и ручной верификации сопоставлений с ЕНС.

## Архитектура

```
Frontend (React + TypeScript + Tailwind)  :5173 (dev) / :8000 (prod)
       |
       | REST API /api/*
       |
Backend (FastAPI + Python)                :8000
       |
       | Использует
       |
Существующая логика (core/*.py)
```

## Стек

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, Lucide icons
- **Backend**: FastAPI, Uvicorn, Pydantic, pandas
- **Storage**: SQLite (result.db) — через `core/result_database.py`
- **Processing**: `core/batch_service.py` — выделенная логика из `cli.py`

## Запуск

### Быстрый старт

```bash
# 1. Backend (port 8000)
cd /mnt/agents/output
python web_start.py

# 2. Frontend dev server (port 3000, HMR)
cd /mnt/agents/output/app
npm run dev
```

Открыть `http://localhost:3000` — frontend через Vite dev server с proxy на backend.

### Production (backend обслуживает статику)

```bash
cd /mnt/agents/output/app && npm run build
cd /mnt/agents/output && python web_start.py
```

Открыть `http://localhost:8000` — backend раздаёт собранный frontend из `app/dist/`.

### Через uvicorn напрямую

```bash
cd /mnt/agents/output
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

## Файлы

| Файл | Описание |
|---|---|
| `api_server.py` | FastAPI: endpoints, Pydantic models, background tasks |
| `web_start.py` | Скрипт запуска uvicorn для backend |
| `core/batch_service.py` | Чистые функции batch processing (выделено из cli.py) |
| `core/result_database.py` | SQLite: upsert_result, get_result, search |
| `app/src/api/client.ts` | TypeScript клиент для всех endpoints |
| `app/src/pages/UploadPage.tsx` | Страница загрузки Excel + настройки |
| `app/src/pages/ResultsPage.tsx` | Страница результатов: таблица + фильтры |
| `app/src/components/VerifyModal.tsx` | Модальное окно верификации (топ-5 кандидатов) |
| `app/src/App.tsx` | Роутинг: `/` → UploadPage, `/results/:jobId` → ResultsPage |

## Экраны

### 1. Загрузка (UploadPage)

- Drag-and-drop Excel-файл (.xlsx, .xls, .xlsm)
- Настройки: домен (combobox из `config/*.yaml`), потоки (1-16), пути к БД масок и результатов
- Кнопка «Обработать» → загрузка файла + запуск фоновой обработки → редирект на результаты

### 2. Результаты (ResultsPage)

- Прогресс-бар с polling (2 сек) во время обработки
- Таблица: #, наименование, код ЕНС, confidence (цветной badge), тип, стандарт, тип сопоставления
- Фильтры: стандарт, тип изделия, confidence min/max, «только успешные», кнопка «Применить»
- Кнопка «Верифицировать» на каждой строке с confidence < 1.0
- Экспорт: Excel / JSON

### 3. Верификация (VerifyModal)

- Исходное наименование + текущий ЕНС (если есть)
- Топ-5 кандидатов с: кодом, именем, score, сравнением параметров (= / ~ / !=)
- Клик по кандидату — выбор
- Ручной ввод кода ЕНС (input)
- Кнопка «Подтвердить» → `POST /api/verify` → `verified=true`, `match_type='manual_verification'`

## API Endpoints

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/domains` | Список доменов из `config/*.yaml` |
| `POST` | `/api/upload` | Загрузка Excel → `{job_id, rows, name_column}` |
| `POST` | `/api/process/{job_id}` | Запуск batch processing (body: ProcessConfig) |
| `GET` | `/api/jobs/{job_id}` | Статус задания + прогресс |
| `POST` | `/api/jobs/{job_id}/results` | Результаты с фильтрами (body: FilterRequest) |
| `GET` | `/api/jobs/{job_id}/results/{idx}/candidates` | Топ-5 кандидатов для строки |
| `POST` | `/api/jobs/{job_id}/results/{idx}/verify` | Ручная верификация (body: VerifyRequest) |
| `GET` | `/api/jobs/{job_id}/export/{format}` | Экспорт: `excel` или `json` |
| `GET` | `/api/result-db/search` | Поиск в result.db по параметрам |

## Pydantic Models

```python
ProcessConfig:
  domain: str = 'hardware'
  workers: int = 4           # 1-16
  db_path: str = 'cache/masks.db'
  result_db_path: str = 'cache/result.db'

FilterRequest:
  standard: Optional[str]
  item_type: Optional[str]
  confidence_min: Optional[float]
  confidence_max: Optional[float]
  success_only: bool = False
  limit: int = 50            # 1-1000
  offset: int = 0

VerifyRequest:
  ens_code: str
  ens_name: Optional[str]
  confidence: float = 1.0    # 0.0-1.0
```

## Добавление новых endpoints

1. **Backend**: добавить функцию в `api_server.py`
2. **Pydantic model**: если нужен input — добавить `class ...` на строку ~100
3. **Frontend**: добавить метод в `app/src/api/client.ts`
4. **Компонент**: использовать метод через `import { api } from '@/api/client'`

## Proxy (dev)

`app/vite.config.ts`:
```typescript
server: {
  proxy: {
    '/api': { target: 'http://localhost:8000', changeOrigin: true },
  },
},
```

## Что вынесено из cli.py

- `process_batch()` — пакетная обработка с callback-прогрессом
- `process_excel()` — чтение Excel + поиск колонки + batch
- `results_to_excel_rows()` — преобразование в строки для экспорта
- `BatchService` — инициализация процессора, workers, сохранение в БД

`cli.py batch` теперь может использовать `BatchService` напрямую (backward compatible).
