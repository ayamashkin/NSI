# =============================================================================
# FILE: core/domain_config.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 CHANGES (UTC+3):
# 2026-05-29 13:30:00 — FEAT: added twin_groups field for regex↔DB field mappings (e.g., свойства→группа_прочности)
# 2026-05-28 22:32:00 — Added loose_match_fields for substring-match validation (e.g. coating)
# 2026-05-28 12:45:00 — Added min_examples field (default 5) for index builder threshold
# 2026-05-27 21:15:00 — Added meta_regex_groups for configurable regex meta-groups
# 2026-05-27 18:15:00 — Added prompt_template for domain prompts
# =============================================================================
"""
Domain Configuration Module
Loads subject configuration (domain) from YAML.
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
    """Subject area configuration."""
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
    meta_regex_groups: List[str] = field(default_factory=lambda: ["тип_изделия", "нтд_1"])
    min_examples: int = 5  # minimum examples per (standard, type) to include in index
    loose_match_fields: Set[str] = field(default_factory=set)  # fields where substring match is OK
    twin_groups: List[List[str]] = field(default_factory=list)  # [regex_field, db_field] mappings

    @classmethod
    def load(cls, domain: str, base_path: str = "config/domains") -> "DomainConfig":
        """Load domain configuration from YAML."""
        path = Path(base_path) / f"{domain}.yaml"
        if not path.exists():
            logger.warning("[DomainConfig] Domain file not found: %s", path)
            return cls(domain=domain)

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        idx = data.get("index", {})

        # field_aliases: key in YAML may contain dots/brackets — read as strings
        aliases_raw = idx.get("field_aliases", {})
        aliases: Dict[str, str] = {}
        for k, v in aliases_raw.items():
            aliases[str(k).strip()] = str(v).strip()

        # meta_regex_groups from index or root
        meta_groups = data.get("meta_regex_groups", idx.get("meta_regex_groups", ["тип_изделия", "нтд_1"]))
        if isinstance(meta_groups, str):
            meta_groups = [g.strip() for g in meta_groups.split(",")]

        # loose_match_fields from index or root
        loose_fields = set(str(x) for x in idx.get("loose_match_fields", data.get("loose_match_fields", [])))

        # twin_groups from index
        twin_groups_raw = idx.get("twin_groups", [])
        twin_groups: List[List[str]] = []
        if isinstance(twin_groups_raw, list):
            for item in twin_groups_raw:
                if isinstance(item, list) and len(item) >= 2:
                    twin_groups.append([str(x) for x in item])

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
            meta_regex_groups=meta_groups,
            min_examples=int(idx.get("min_examples", 5)),
            loose_match_fields=loose_fields,
            twin_groups=twin_groups,
        )

    def canonicalize_field_name(self, original: str) -> str:
        """Convert original ENS field name to canonical name."""
        if original in self.field_aliases:
            return self.field_aliases[original]

        # Fallback: cleanup
        name = re.sub(r"\s*\([^)]*\)\s*", " ", original).strip()
        name = name.lower().replace(" ", "_").replace(",", "_")
        name = re.sub(r"_+", "_", name).strip("_")

        if len(name) > self.max_field_name_len:
            # Remove duplicate words and articles
            name = re.sub(r"(ди|de|la|le|et|du)", "", name)
            name = re.sub(r"_+", "_", name).strip("_")
            name = name[:self.max_field_name_len]

        return name

    def is_skip_field(self, field: str) -> bool:
        return field in self.skip_fields

    def is_meta_field(self, field: str) -> bool:
        return field in self.meta_fields

    def is_retain_field(self, field: str) -> bool:
        return field in self.retain_fields

    def is_loose_match_field(self, field: str) -> bool:
        return field in self.loose_match_fields