# =============================================================================
# FILE: core/domain_config.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 CHANGES (UTC+3):
# 2026-05-27 18:15:00 — Добавлен prompt_template для доменных промптов
# 2026-05-27 14:05:00 — Создан DomainConfig для доменной архитектуры ENS
# 2026-05-27 14:05:00 — Добавлена поддержка skip_fields, meta_fields, retain_fields
# 2026-05-27 14:05:00 — Добавлены field_aliases и нормализация имён полей
# 2026-05-27 14:05:00 — Реализован загрузчик YAML из config/domains/
# =============================================================================
"""
Domain Configuration Module
Загружает предметную конфигурацию (домен) из YAML.
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DomainConfig:
    """Конфигурация предметной области."""
    domain: str = ""
    description: str = ""
    index_path: str = ""
    prompt_template: Optional[str] = None
    skip_fields: Set[str] = field(default_factory=set)
    meta_fields: Set[str] = field(default_factory=set)
    retain_fields: Set[str] = field(default_factory=set)
    field_aliases: Dict[str, str] = field(default_factory=dict)
    twin_threshold: float = 1.0
    visible_threshold: float = 0.05
    max_field_name_len: int = 30

    @classmethod
    def load(cls, domain: str, base_path: str = "config/domains") -> "DomainConfig":
        """Загрузить конфигурацию домена из YAML."""
        path = Path(base_path) / f"{domain}.yaml"
        if not path.exists():
            logger.warning("[DomainConfig] Domain file not found: %s", path)
            return cls(domain=domain)

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        idx = data.get("index", {})

        # field_aliases: ключ в YAML может содержать точки/скобки — читаем как строки
        aliases_raw = idx.get("field_aliases", {})
        aliases: Dict[str, str] = {}
        for k, v in aliases_raw.items():
            aliases[str(k).strip()] = str(v).strip()

        return cls(
            domain=domain,
            description=data.get("description", ""),
            index_path=data.get("index_path", f"cache/ens_{domain}.pkl"),
            prompt_template=data.get("prompt_template"),
            skip_fields=set(str(x) for x in idx.get("skip_fields", [])),
            meta_fields=set(str(x) for x in idx.get("meta_fields", [])),
            retain_fields=set(str(x) for x in idx.get("retain_fields", [])),
            field_aliases=aliases,
            twin_threshold=float(idx.get("twin_threshold", 1.0)),
            visible_threshold=float(idx.get("visible_threshold", 0.05)),
            max_field_name_len=int(idx.get("max_field_name_len", 30)),
        )

    def canonicalize_field_name(self, original: str) -> str:
        """Преобразовать оригинальное имя поля ENS в canonical name."""
        if original in self.field_aliases:
            return self.field_aliases[original]

        # Fallback: очистка
        name = re.sub(r"\s*\([^)]*\)\s*", " ", original).strip()
        name = name.lower().replace(" ", "_").replace(",", "_")
        name = re.sub(r"_+", "_", name).strip("_")

        if len(name) > self.max_field_name_len:
            # Удаляем повторяющиеся слова и артикли
            name = re.sub(r"\b(ди|de|la|le|et|du)\b", "", name)
            name = re.sub(r"_+", "_", name).strip("_")
            name = name[:self.max_field_name_len]

        return name

    def is_skip_field(self, field: str) -> bool:
        return field in self.skip_fields

    def is_meta_field(self, field: str) -> bool:
        return field in self.meta_fields

    def is_retain_field(self, field: str) -> bool:
        return field in self.retain_fields