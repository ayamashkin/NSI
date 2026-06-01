#!/usr/bin/env python3
# =============================================================================
# ENS Index Builder & Loader
# Structured index builder and query loader for ENS (ЕНС) reference data.
#
# LAST_FIXES:
#  2026-06-01 11:00:00 UTC+3 — FIX: ENSIndexLoader reads visible_fields from index stats; adds name/ens_code to root for compatibility
#  2026-05-28 20:40:00 UTC+3 — FIX: get_mask() now normalizes standard via canonicalize_standard()
#  2026-05-28 20:30:00 UTC+3 — FIX: get_mask() fallback search by standard error fixed
#  2026-05-28 20:15:00 UTC+3 — FIX: get_mask() now uses canonicalize_standard() for standard normalization
#  2026-05-28 20:00:00 UTC+3 — FIX: get_mask() fallback search by standard error fixed
# =============================================================================

import pickle
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from utils.standard_utils import canonicalize_standard
from core.domain_config import DomainConfig

logger = logging.getLogger(__name__)


class ENSIndexBuilder:
    """Build structured ENS domain index from Excel."""

    def __init__(self, domain_config: DomainConfig):
        self.config = domain_config
        self.skip_fields = set(domain_config.skip_fields or [])
        self.meta_fields = set(domain_config.meta_fields or [])
        self.retain_fields = set(domain_config.retain_fields or [])
        self.loose_match_fields = set(domain_config.loose_match_fields or [])
        self.twin_groups = domain_config.twin_groups or []
        self.field_aliases = domain_config.field_aliases or {}
        self.twin_threshold = getattr(domain_config, 'twin_threshold', 1.0)
        self.visible_threshold = getattr(domain_config, 'visible_threshold', 0.05)
        self.max_field_name_len = getattr(domain_config, 'max_field_name_len', 30)
        self.min_examples = getattr(domain_config, 'min_examples', 5)

    def _canonicalize_field_name(self, field: str) -> str:
        if field in self.field_aliases:
            return self.field_aliases[field]
        field = field.lower().strip()
        field = field.replace(' ', '_').replace('-', '_')
        field = field.replace('(', '').replace(')', '')
        field = field.replace('.', '_')
        field = field.replace('__', '_')
        if len(field) > self.max_field_name_len:
            field = field[:self.max_field_name_len]
        return field

    def _get_example_name(self, row: pd.Series) -> str:
        return str(row.get('наименование') or row.get('полное_наименование') or '')

    def _build_index(self, df: pd.DataFrame) -> Dict[str, Dict[str, Dict]]:
        df = df.copy()
        df.columns = [self._canonicalize_field_name(str(c)) for c in df.columns]
        df['стандарт_канон'] = df['стандарт'].apply(canonicalize_standard)
        df['тип_канон'] = df['тип_изделия'].str.upper().str.strip()
        grouped = df.groupby(['стандарт_канон', 'тип_канон'])
        index = {}
        for (std, itype), group in grouped:
            if len(group) < self.min_examples:
                continue
            visible_fields = set()
            meta_fields = set(self.meta_fields)
            skip_fields = set(self.skip_fields)
            for _, row in group.iterrows():
                name = self._get_example_name(row)
                for field in row.index:
                    if field.startswith('_'):
                        continue
                    if field in skip_fields:
                        continue
                    if field in meta_fields:
                        continue
                    value = row.get(field)
                    if value is not None and pd.notna(value) and str(value) in name:
                        visible_fields.add(field)
            twin_groups = []
            for group_pair in self.twin_groups:
                if len(group_pair) == 2:
                    f1, f2 = group_pair
                    if f1 in visible_fields and f2 in visible_fields:
                        twin_groups.append([f1, f2])
            examples = []
            for _, row in group.iterrows():
                example = {}
                for field in visible_fields:
                    value = row.get(field)
                    if value is not None and pd.notna(value):
                        example[field] = str(value).strip()
                example['_meta'] = {
                    'name': row.get('наименование') or row.get('полное_наименование') or '',
                    'ens_code': row.get('код') or '',
                }
                examples.append(example)
            entry = {
                'examples': examples,
                'twin_groups': twin_groups,
                'stats': {
                    'visible_fields': sorted(visible_fields),
                    'metadata_fields': sorted(meta_fields),
                    'total_examples': len(group),
                    'twin_groups': twin_groups,
                },
            }
            if std not in index:
                index[std] = {}
            index[std][itype] = entry
        return index

    def build(self, excel_file: str, output_path: str) -> str:
        df = pd.read_excel(excel_file)
        index = self._build_index(df)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(index, f)
        logger.info("[ENSIndexBuilder] Built index: %s (standards=%d)", output_path, len(index))
        return output_path


class ENSIndexLoader:
    """Load and query a structured ENS index."""

    def __init__(self, index_path: str):
        self.index_path = index_path
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        self.index = {}
        self.index_meta = {}  # (std, type) -> entry metadata
        if isinstance(data, dict) and not data.get('items'):
            for std, types in data.items():
                for itype, entry in types.items():
                    examples = entry.get('examples', [])
                    self.index[(std, itype.upper())] = examples
                    self.index_meta[(std, itype.upper())] = {
                        'visible_fields': entry.get('stats', {}).get('visible_fields', []),
                        'metadata_fields': entry.get('stats', {}).get('metadata_fields', []),
                        'twin_groups': entry.get('twin_groups', []),
                    }
        else:
            items = data.get('items', [])
            for item in items:
                std = canonicalize_standard(item.get('стандарт') or item.get('нтд') or 'UNKNOWN')
                itype = item.get('тип_изделия') or item.get('наименование_типа') or item.get('тип') or 'unknown'
                key = (std, itype.upper())
                if key not in self.index:
                    self.index[key] = []
                self.index[key].append(item)
        self.visible_fields = set()
        self.loose_match_fields = set()
        self.twin_groups = []
        for (std, itype), examples in self.index.items():
            meta_info = self.index_meta.get((std, itype), {})
            stored_visible = meta_info.get('visible_fields', [])
            if stored_visible:
                self.visible_fields.update(stored_visible)
                self.twin_groups.extend(meta_info.get('twin_groups', []))
                continue
            for ex in examples:
                name = str(ex.get('_meta', {}).get('name', '') or ex.get('наименование') or ex.get('полное_наименование') or '')
                for field, value in ex.items():
                    if field.startswith('_'):
                        continue
                    if value is not None and value != '' and str(value) in name:
                        self.visible_fields.add(field)
                twin = ex.get('twin_groups')
                if twin:
                    self.twin_groups.extend(twin)
        self.twin_map = {}
        for group in self.twin_groups:
            if len(group) == 2:
                self.twin_map[group[0]] = group[1]
                self.twin_map[group[1]] = group[0]

    def get_examples(self, standard: str, item_type: str) -> List[Dict]:
        key = (standard, item_type.upper())
        examples = self.index.get(key, [])
        # FIX 2026-06-01 11:00:00 UTC+3: ensure backward compatibility by adding name/ens_code to root
        for ex in examples:
            meta = ex.get('_meta', {})
            if 'name' in meta and 'наименование' not in ex:
                ex['наименование'] = meta['name']
            if 'ens_code' in meta and 'код' not in ex:
                ex['код'] = meta['ens_code']
        return examples

    def get_candidates(self, standard: str, item_type: str):
        key = (standard, item_type.upper())
        if key not in self.index:
            return []
        examples = self.index[key]
        candidates = []
        for ex in examples:
            candidate = {}
            for field in self.visible_fields:
                value = ex.get(field)
                if value is not None and value != '':
                    candidate[field] = value
            meta = ex.get('_meta', {})
            candidate['name'] = meta.get('name', '')
            candidate['ens_code'] = meta.get('ens_code', '')
            candidate['наименование'] = meta.get('name', '')
            candidate['код'] = meta.get('ens_code', '')
            candidate['_meta'] = meta
            candidates.append(candidate)
        return candidates