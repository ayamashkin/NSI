"""
Core Models Module
Pydantic модели данных для системы обработки номенклатуры.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Статусы обработки."""
    COMPLETED = "completed"
    IGNORED = "ignored"
    ERROR = "error"
    PENDING = "pending"


class Category(str, Enum):
    """Категории номенклатуры."""
    HARDWARE = "hardware"
    KREPEZH = "krepezh"
    ERI = "eri"
    MATERIALS = "materials"
    PURCHASED = "purchased"
    UNKNOWN = "unknown"


class Parameter(BaseModel):
    """Модель параметра изделия."""
    name: str
    value: Optional[str] = ""
    default: Optional[str] = ""
    um: Optional[str] = ""  # Единица измерения


class ProcessingResult(BaseModel):
    """Результат обработки номенклатуры."""
    article: str
    name: str
    guid: str
    prompt_id: str
    category: str
    status: ProcessingStatus
    display_name: Optional[str] = None
    params: List[Parameter] = Field(default_factory=list)
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: Optional[str] = None
    api_source: Optional[str] = None


class NomenclatureItem(BaseModel):
    """Элемент номенклатуры из Excel."""
    article: str = Field(..., alias="артикул")
    name: str = Field(..., alias="Краткое наименование")
    guid: str = Field(..., alias="GUID")

    class Config:
        populate_by_name = True


class PromptConfigModel(BaseModel):
    """Модель конфигурации промпта (для валидации)."""
    id: str
    name: str
    file: str
    category: str
    keywords: List[str]
    service: str
    model: str
    temperature: float = 0.1


class APIConfigModel(BaseModel):
    """Модель конфигурации API."""
    base_url: str
    api_key_file: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 120
    default_model: Optional[str] = None
