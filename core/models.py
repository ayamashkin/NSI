from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    COMPLETED = "completed"
    IGNORED = "ignored"
    ERROR = "error"
    PENDING = "pending"


class Category(str, Enum):
    KREPEZH = "krepezh"
    ERI = "eri"
    MATERIALS = "materials"
    PURCHASED = "purchased"
    UNKNOWN = "unknown"


class Parameter(BaseModel):
    name: str
    value: Optional[str] = ""
    default: Optional[str] = ""
    um: Optional[str] = ""


class ProcessingResult(BaseModel):
    article: str
    name: str
    guid: str
    prompt_id: str
    category: Category
    status: ProcessingStatus
    display_name: Optional[str] = None
    params: List[Parameter] = Field(default_factory=list)
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: Optional[str] = None
    api_source: Optional[str] = None


class NomenclatureItem(BaseModel):
    article: str = Field(..., alias="артикул")
    name: str = Field(..., alias="Краткое наименование")
    guid: str = Field(..., alias="GUID")

    class Config:
        populate_by_name = True


class PromptConfig(BaseModel):
    id: str
    name: str
    file_path: str
    category: Category
    keywords: List[str]
    model: str
    temperature: float = 0.1