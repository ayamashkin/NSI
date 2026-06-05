#!/usr/bin/env python3
# =============================================================================
# ФАЙЛ: api_server.py
# ПОСЛЕДНИЕ 5 ИЗМЕНЕНИЙ (МСК, UTC+3), от новых к старым:
# 2026-06-04 15:00:00 — FEAT: FastAPI backend + static files (frontend build).
# 2026-06-04 15:00:00 — FEAT: ResultDatabaseManager (core/result_database.py).
# 2026-06-04 15:00:00 — FEAT: BatchService (core/batch_service.py) — вынос из cli.py.
# =============================================================================

"""
FastAPI Backend — web API для пакетной обработки и верификации номенклатуры.
Использует существующую Python-логику: batch_service.py, result_database.py.
"""

import os
import sys
import json
import asyncio
import logging
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ENS Verification API",
    description="Web API для пакетной обработки и верификации номенклатуры с ЕНС",
    version="1.0.0",
)

# CORS для React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (for async processing)
jobs: Dict[str, dict] = {}

# Global batch service instance (lazy init)
_batch_service = None

def get_batch_service(
    db_path: str = 'cache/masks.db',
    ens_index_path: Optional[str] = None,
    result_db_path: str = 'cache/result.db',
    domain: str = 'hardware',
    workers: int = 4,
):
    """Get or create BatchService instance."""
    global _batch_service
    from core.batch_service import BatchService
    _batch_service = BatchService(
        db_path=db_path,
        ens_index_path=ens_index_path,
        result_db_path=result_db_path,
        domain=domain,
        workers=workers,
        no_cache=False,
        include_details=True,
    )
    return _batch_service


# =============================================================================
# Pydantic Models
# =============================================================================

class ProcessConfig(BaseModel):
    domain: str = Field(default='hardware', description='Домен ENS')
    workers: int = Field(default=4, ge=1, le=16, description='Количество потоков')
    db_path: str = Field(default='cache/masks.db', description='Путь к БД масок')
    result_db_path: str = Field(default='cache/result.db', description='Путь к result.db')


class ResultItem(BaseModel):
    id: int
    text: str
    ens_code: Optional[str] = None
    ens_name: Optional[str] = None
    success: bool
    confidence: float
    match_type_ru: Optional[str] = None
    item_type: Optional[str] = None
    standard: Optional[str] = None
    params: dict = Field(default_factory=dict)
    ens_params: dict = Field(default_factory=dict)
    ens_params_mask: dict = Field(default_factory=dict)
    mask_pattern: Optional[str] = None
    details: dict = Field(default_factory=dict)


class CandidateItem(BaseModel):
    ens_code: Optional[str] = None
    name: Optional[str] = None
    score: float = 0.0
    params_comparison: dict = Field(default_factory=dict)


class VerifyRequest(BaseModel):
    ens_code: str = Field(..., description='Код ЕНС для верификации')
    ens_name: Optional[str] = Field(default=None, description='Наименование ЕНС')
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FilterRequest(BaseModel):
    standard: Optional[str] = None
    item_type: Optional[str] = None
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    success_only: bool = False
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/domains")
def list_domains():
    """Список доступных доменов."""
    from core.batch_service import BatchService
    return {"domains": BatchService.get_available_domains()}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(..., description="Excel-файл с номенклатурой"),
):
    """Загрузить Excel-файл. Вернуть job_id для обработки."""
    if not file.filename.endswith(('.xlsx', '.xls', '.xlsm')):
        raise HTTPException(400, detail="Только Excel-файлы (.xlsx, .xls, .xlsm)")

    job_id = str(uuid.uuid4())
    upload_dir = Path(tempfile.gettempdir()) / "ens_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{job_id}_{file.filename}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Validate: can read and find name column
    try:
        df = pd.read_excel(file_path)
        from core.batch_service import _find_name_column
        name_col = _find_name_column(df)
        if name_col is None:
            raise HTTPException(400, detail=f"Колонка с наименованием не найдена. Колонки: {list(df.columns)}")
        jobs[job_id] = {
            "id": job_id,
            "status": "uploaded",
            "file_path": str(file_path),
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "name_column": name_col,
            "results": None,
            "stats": None,
            "error": None,
        }
        return {
            "job_id": job_id,
            "status": "uploaded",
            "rows": len(df),
            "name_column": name_col,
            "columns": list(df.columns),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Ошибка чтения Excel: {e}")


@app.post("/api/process/{job_id}")
async def process_batch(
    job_id: str,
    config: ProcessConfig,
    background_tasks: BackgroundTasks,
):
    """Запустить пакетную обработку для загруженного файла."""
    if job_id not in jobs:
        raise HTTPException(404, detail="Job не найден. Загрузите файл сначала.")

    job = jobs[job_id]
    if job["status"] == "processing":
        return {"job_id": job_id, "status": "processing", "message": "Обработка уже идет"}

    job["status"] = "processing"
    job["config"] = config.model_dump()

    # Run in background
    background_tasks.add_task(_run_batch_processing, job_id, config)

    return {"job_id": job_id, "status": "processing", "total_rows": job["rows"]}


def _run_batch_processing(job_id: str, config: ProcessConfig):
    """Background task: run batch processing."""
    job = jobs[job_id]
    try:
        from core.batch_service import BatchService
        service = get_batch_service(
            db_path=config.db_path,
            domain=config.domain,
            workers=config.workers,
            result_db_path=config.result_db_path,
        )
        results, stats, df, name_col = service.process_excel(
            job["file_path"],
            progress_callback=lambda current, total, s: _update_progress(job_id, current, total, s),
        )
        job["status"] = "completed"
        job["results"] = [r.to_dict() for r in results]
        job["stats"] = stats
        job["excel_rows"] = service.results_to_excel_rows(results, df, name_col)
        logger.info("[API] Job %s completed: %s", job_id, stats)
    except Exception as e:
        logger.error("[API] Job %s failed: %s", job_id, e)
        job["status"] = "failed"
        job["error"] = str(e)


def _update_progress(job_id: str, current: int, total: int, stats: dict):
    """Update job progress."""
    if job_id in jobs:
        jobs[job_id]["progress"] = {
            "current": current,
            "total": total,
            "percent": round(current / total * 100, 1) if total > 0 else 0,
            "stats": stats,
        }


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Получить статус задания."""
    if job_id not in jobs:
        raise HTTPException(404, detail="Job не найден")
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "filename": job.get("filename"),
        "rows": job.get("rows"),
        "progress": job.get("progress"),
        "stats": job.get("stats"),
        "error": job.get("error"),
    }


@app.post("/api/jobs/{job_id}/results")
def get_results(job_id: str, filters: FilterRequest):
    """Получить результаты обработки с фильтрами."""
    if job_id not in jobs:
        raise HTTPException(404, detail="Job не найден")

    job = jobs[job_id]
    if job.get("results") is None:
        return {"results": [], "total": 0, "stats": job.get("stats")}

    all_results = job["results"]
    filtered = all_results

    # Apply filters
    if filters.standard:
        filtered = [r for r in filtered if (r.get('standard') or '').upper() == filters.standard.upper()]
    if filters.item_type:
        filtered = [r for r in filtered if (r.get('item_type') or '').upper() == filters.item_type.upper()]
    if filters.confidence_min is not None:
        filtered = [r for r in filtered if (r.get('confidence') or 0) >= filters.confidence_min]
    if filters.confidence_max is not None:
        filtered = [r for r in filtered if (r.get('confidence') or 0) <= filters.confidence_max]
    if filters.success_only:
        filtered = [r for r in filtered if r.get('success')]

    total = len(filtered)
    paginated = filtered[filters.offset:filters.offset + filters.limit]

    # Add id for frontend
    for i, r in enumerate(paginated):
        r['id'] = filters.offset + i

    return {
        "results": paginated,
        "total": total,
        "stats": job.get("stats"),
    }


@app.get("/api/jobs/{job_id}/results/{result_idx}/candidates")
def get_candidates(job_id: str, result_idx: int):
    """Получить топ-5 кандидатов для записи результата."""
    if job_id not in jobs:
        raise HTTPException(404, detail="Job не найден")

    job = jobs[job_id]
    results = job.get("results")
    if not results or result_idx >= len(results):
        raise HTTPException(404, detail="Результат не найден")

    result = results[result_idx]
    details = result.get('details', {})

    # Get candidates from debug_candidates or top_candidates
    candidates = details.get('top_candidates', []) or details.get('debug_candidates', [])

    # If no candidates in details, return empty
    if not candidates:
        # Try to get candidates from result_db
        try:
            from core.result_database import ResultDatabaseManager
            manager = ResultDatabaseManager(db_path=job.get('config', {}).get('result_db_path', 'cache/result.db'))
            db_result = manager.get_result(name=result.get('text'))
            if db_result and db_result.get('details'):
                db_details = db_result['details']
                if isinstance(db_details, str):
                    db_details = json.loads(db_details)
                candidates = db_details.get('debug_candidates', []) or db_details.get('top_candidates', [])
        except Exception as e:
            logger.warning("[API] Failed to load candidates from DB: %s", e)

    return {
        "text": result.get('text'),
        "ens_code": result.get('ens_code'),
        "ens_name": result.get('ens_name'),
        "confidence": result.get('confidence'),
        "success": result.get('success'),
        "candidates": candidates[:5],
    }


@app.post("/api/jobs/{job_id}/results/{result_idx}/verify")
def verify_result(job_id: str, result_idx: int, request: VerifyRequest):
    """Верифицировать результат — записать выбранный ЕНС в result.db."""
    if job_id not in jobs:
        raise HTTPException(404, detail="Job не найден")

    job = jobs[job_id]
    results = job.get("results")
    if not results or result_idx >= len(results):
        raise HTTPException(404, detail="Результат не найден")

    result = results[result_idx]
    result_db_path = job.get('config', {}).get('result_db_path', 'cache/result.db')

    try:
        from core.result_database import ResultDatabaseManager
        manager = ResultDatabaseManager(db_path=result_db_path)

        # Update result with verified ENS code
        changed, reason = manager.upsert_result(
            name=result.get('text'),
            ens_code=request.ens_code,
            ens_name=request.ens_name,
            success=True,
            confidence=request.confidence,
            match_type='manual_verification',
            match_type_ru='Ручная верификация',
            item_type=result.get('item_type'),
            standard=result.get('standard'),
            params=result.get('params'),
            ens_params=result.get('ens_params'),
            ens_params_mask=result.get('ens_params_mask', {}),
            verified=True,
        )

        # Update in-memory result
        result['ens_code'] = request.ens_code
        result['ens_name'] = request.ens_name
        result['success'] = True
        result['confidence'] = request.confidence
        result['match_type'] = 'manual_verification'
        result['match_type_ru'] = 'Ручная верификация'

        return {
            "success": True,
            "changed": changed,
            "reason": reason,
            "ens_code": request.ens_code,
        }
    except Exception as e:
        logger.error("[API] Verification failed: %s", e)
        raise HTTPException(500, detail=f"Ошибка верификации: {e}")


@app.get("/api/jobs/{job_id}/export/{format}")
def export_results(job_id: str, format: str):
    """Экспорт результатов в Excel или JSON."""
    if job_id not in jobs:
        raise HTTPException(404, detail="Job не найден")

    job = jobs[job_id]
    if job.get("status") != "completed":
        raise HTTPException(400, detail="Обработка еще не завершена")

    excel_rows = job.get("excel_rows", [])
    if not excel_rows:
        raise HTTPException(400, detail="Нет данных для экспорта")

    if format.lower() == 'excel':
        df = pd.DataFrame(excel_rows)
        export_dir = Path(tempfile.gettempdir()) / "ens_exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / f"{job_id}_results.xlsx"

        # Format multiline cells
        with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            ws = writer.sheets['Results']
            # Set column widths
            for idx_col, col_name in enumerate(df.columns):
                max_len = max(
                    df[col_name].astype(str).str.len().max() if not df.empty else 10,
                    len(str(col_name))
                )
                ws.column_dimensions[chr(65 + idx_col) if idx_col < 26 else 'A'].width = min(max_len + 2, 50)

        return FileResponse(
            export_path,
            filename=f"{job.get('filename', 'results').replace('.xlsx', '')}_results.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif format.lower() == 'json':
        return JSONResponse(
            content=job.get("results", []),
            headers={"Content-Disposition": f"attachment; filename={job_id}_results.json"}
        )

    else:
        raise HTTPException(400, detail="Формат должен быть 'excel' или 'json'")


@app.get("/api/result-db/search")
def search_result_db(
    q: Optional[str] = Query(None, description="Поиск по наименованию"),
    standard: Optional[str] = Query(None),
    item_type: Optional[str] = Query(None),
    confidence_min: Optional[float] = Query(None),
    confidence_max: Optional[float] = Query(None),
    success_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    result_db_path: str = Query('cache/result.db'),
):
    """Поиск в result.db (глобальный, не привязан к job)."""
    try:
        from core.result_database import ResultDatabaseManager
        manager = ResultDatabaseManager(db_path=result_db_path)
        results = manager.search(
            query=q,
            standard=standard,
            item_type=item_type,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
            success_only=success_only,
            limit=limit,
            offset=offset,
        )
        return {"results": results, "total": len(results)}
    except Exception as e:
        logger.error("[API] DB search failed: %s", e)
        raise HTTPException(500, detail=f"Ошибка поиска: {e}")


# =============================================================================
# Static files (frontend build)
# =============================================================================

_frontend_dist = Path(__file__).parent / "app" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
