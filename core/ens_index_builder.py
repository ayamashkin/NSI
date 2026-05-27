# =============================================================================
# FILE: core/ens_index_builder.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 CHANGES (UTC+3):
# 2026-05-27 17:46:00 — Исправлено: поля наименование/код/тип/НТД исключены из visible_fields stats
# 2026-05-27 17:46:00 — _meta-поля удаляются из field_meta ДО формирования stats (не попадают в regex)
# 2026-05-27 17:46:00 — Добавлен _meta_field_names для определения полей, уходящих в _meta
# 2026-05-27 14:10:00 — Создан построитель доменного индекса ENS
# 2026-05-27 14:10:00 — Реализована нормализация заголовков и удаление skip_fields
# =============================================================================
"""
ENS Index Builder Module
Строит структурированный доменный индекс из Excel-файла ЕНС.
"""
import logging
import click
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from core.domain_config import DomainConfig
from utils.standard_utils import canonicalize_standard

logger = logging.getLogger(__name__)


class ENSIndexBuilder:
    """Построитель индекса ENS для заданного домена."""

    def __init__(self, domain_config: DomainConfig):
        self.domain = domain_config
        self._is_value_in_name = self._build_is_value_in_name()

    def _build_is_value_in_name(self):
        """Статический метод для проверки вхождения значения в наименование."""
        # Импортируем из llm_mask_generator если доступен, иначе inline
        try:
            from core.llm_mask_generator import LLMMaskGenerator
            return LLMMaskGenerator._is_value_in_name
        except Exception:
            return self._default_is_value_in_name

    @staticmethod
    def _default_is_value_in_name(val: str, name: str, param_key: str = "", standard: str = "") -> bool:
        """Fallback проверка вхождения значения в наименование."""
        if not val or not name:
            return False
        val_raw = str(val).strip()
        val_str = val_raw.lower().replace(",", ".")
        name_lower = name.lower().replace(",", ".")
        if val_str in name_lower:
            return True
        if re.match(r"^\d+[.,]\d+$", val_raw):
            no_sep = re.sub(r"[.,]", "", val_str)
            if no_sep in name_lower:
                return True
        if re.search(r"[a-zA-Zа-яА-Я]", val_str):
            tokens = re.split(r"[.\-]", val_str)
            tokens = [t for t in tokens if t and re.search(r"[a-zA-Zа-яА-Я]", t)]
            for tok in tokens:
                if tok in name_lower:
                    return True
            prefix = re.match(r"^([a-zA-Zа-яА-Я]+)", val_str)
            if prefix and prefix.group(1) in name_lower:
                return True
        if "." in val_str and val_str.endswith(".0"):
            int_part = val_str[:-2]
            if int_part and int_part in name_lower:
                return True
        if re.match(r"^\d+[a-zA-Zа-яА-Я]+$", val_str):
            if val_str in name_lower:
                return True
        m_match = re.match(r"^[мm](\d+(?:[.,]\d+)?)$", val_raw, re.IGNORECASE)
        if m_match:
            num = m_match.group(1)
            if num.lower() in name_lower:
                return True
        return False

    def build(self, excel_path: str, output_path: str) -> str:
        """Построить индекс из Excel и сохранить в .pkl."""
        logger.info("[ENSIndexBuilder] Loading %s for domain=%s", excel_path, self.domain.domain)
        df = pd.read_excel(excel_path)
        logger.info("[ENSIndexBuilder] Loaded %d rows, %d columns", len(df), len(df.columns))

        # Найти ключевые колонки
        name_col = self._find_column(df, ["наименование", "полное наименование", "name"])
        type_col = self._find_column(df, ["наименование типа", "тип изделия", "тип"])
        std_col = self._find_column(df, ["нтд", "стандарт", "нтд_1", "standard"])
        code_col = self._find_column(df, ["код", "mdm_key", "id"])
        full_name_col = self._find_column(df, ["полное наименование", "full name"])

        if not name_col:
            raise ValueError("Column with name not found in Excel")
        if not type_col:
            raise ValueError("Column with type not found in Excel")
        if not std_col:
            raise ValueError("Column with standard not found in Excel")

        # Группировка по (стандарт, тип)
        groups: Dict[Tuple[str, str], List[Dict]] = {}
        for _, row in df.iterrows():
            std = canonicalize_standard(str(row.get(std_col, "")))
            itype = str(row.get(type_col, "")).strip()
            if not std or not itype:
                continue
            key = (std, itype)
            if key not in groups:
                groups[key] = []
            record = {}
            for col in df.columns:
                val = row.get(col)
                if pd.notna(val):
                    record[str(col)] = str(val).strip()
                else:
                    record[str(col)] = None
            groups[key].append(record)

        logger.info("[ENSIndexBuilder] Grouped into %d (standard, type) pairs", len(groups))

        # Построение структурированного индекса
        index: Dict[str, Dict[str, Any]] = {}
        for (std, itype), examples in groups.items():
            if len(examples) < 5:
                continue  # пропускаем мелкие группы
            built = self._build_standard_type(std, itype, examples, name_col, code_col, full_name_col, type_col)
            if built:
                if std not in index:
                    index[std] = {}
                index[std][itype] = built
                self._print_group_stats(std, itype, built)

        # Сохранение
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(index, f)

        total_examples = sum(
            len(st[itype]["examples"])
            for st in index.values()
            for itype in st
        )
        logger.info(
            "[ENSIndexBuilder] Saved index to %s: %d standards, %d types, %d examples",
            output_path, len(index), sum(len(v) for v in index.values()), total_examples
        )
        return output_path

    def _print_group_stats(self, standard: str, item_type: str, built: Dict) -> None:
        """Вывести статистику по сформированной группе (стандарт, тип).

        Показывает ТОЛЬКО поля, реально присутствующие в индексе (field_meta).
        Все удалённые (пустые, константные, невидимые) поля исключены.
        """
        stats = built.get("stats", {})
        field_meta = built.get("field_meta", {})
        twin_groups = built.get("twin_groups", [])
        examples = built.get("examples", [])
        total = stats.get("total", len(examples))
        visible_fields = stats.get("visible_fields", [])
        metadata_fields = stats.get("metadata_fields", [])

        # Фильтруем twin_groups: показываем только осмысленные (не giant clusters)
        meaningful_twins = [
            g for g in twin_groups
            if len(g) <= 5
        ]

        click.echo(f"\n{'=' * 60}")
        click.echo(f"📊 {standard} / {item_type} — {total} примеров")
        click.echo(f"{'=' * 60}")
        click.echo(
            f" Полей в индексе: {len(field_meta)} (visible: {len(visible_fields)}, metadata: {len(metadata_fields)})")

        if visible_fields:
            click.echo(f"\n 📋 Видимые параметры (участвуют в regex):")
            for f in visible_fields:
                meta = field_meta.get(f, {})
                vc = meta.get("visible_count", 0)
                ratio = vc / total * 100 if total > 0 else 0
                bar_len = int(ratio / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                click.echo(f" {f:30s} {bar} {vc:3d}/{total} ({ratio:5.1f}%) [{meta.get('original_name', f)[:40]}]")

        if metadata_fields:
            click.echo(f"\n 🔒 Метаданные (retain + meta_fields):")
            for f in metadata_fields:
                meta = field_meta.get(f, {})
                vc = meta.get("visible_count", 0)
                click.echo(f" {f:30s} visible={vc}/{total} [{meta.get('original_name', f)[:40]}]")

        if meaningful_twins:
            click.echo(f"\n 👯 Близнецы (twin_groups):")
            for group in meaningful_twins:
                click.echo(f" {' = '.join(group)} → canonical: {group[0]}")
        elif twin_groups:
            giant = [g for g in twin_groups if len(g) > 5]
            click.echo(f"\n ⚠️ Обнаружены giant twin_clusters: {len(giant)} (полей > 5, скорее всего пустые поля)")

        click.echo(f"{'=' * 60}")

    @staticmethod
    def _norm_field_name(name: str) -> str:
        """Нормализация имени поля для сравнения (skip_fields matching)."""
        import re
        return re.sub(r"[^\wа-яА-Я]", "", str(name).lower().strip())

    def _find_column(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        for col in df.columns:
            col_lower = str(col).lower().strip()
            for kw in keywords:
                if kw in col_lower:
                    return col
        return None

    def _build_standard_type(
        self,
        standard: str,
        item_type: str,
        examples: List[Dict],
        name_col: str,
        code_col: Optional[str],
        full_name_col: Optional[str],
        type_col: str,
    ) -> Optional[Dict]:
        """Построить запись индекса для пары (стандарт, тип)."""
        if not examples:
            return None

        # 0. Нормализованный skip_fields для fuzzy matching
        skip_normalized = {self._norm_field_name(f) for f in self.domain.skip_fields}

        # 1. Нормализация заголовков
        canonical_map: Dict[str, str] = {}  # original -> canonical
        all_fields = set()
        for ex in examples:
            all_fields.update(ex.keys())

        # Удаляем skip_fields (с fuzzy matching по нормализованным именам)
        all_fields = {f for f in all_fields if self._norm_field_name(f) not in skip_normalized}

        for field in all_fields:
            canonical = self.domain.canonicalize_field_name(field)
            # Разрешаем дубли canonical имен
            if canonical in canonical_map.values():
                base = canonical
                i = 1
                while canonical in canonical_map.values():
                    canonical = f"{base}_{i}"
                    i += 1
            canonical_map[field] = canonical

        # Перестроить examples с canonical именами
        normalized_examples: List[Dict] = []
        for ex in examples:
            new_ex: Dict[str, Any] = {}
            for orig, val in ex.items():
                if self._norm_field_name(orig) in skip_normalized:
                    continue
                if orig in canonical_map:
                    new_ex[canonical_map[orig]] = val
                else:
                    new_ex[orig] = val
            normalized_examples.append(new_ex)

        # 2. Удалить всегда пустые колонки (>99% пустых)
        non_empty_counts: Dict[str, int] = {}
        for ex in normalized_examples:
            for k, v in ex.items():
                if v is not None and str(v).strip() not in ("", " ", "0", "0.0", "None"):
                    non_empty_counts[k] = non_empty_counts.get(k, 0) + 1

        total_examples = len(normalized_examples)
        always_empty = {
            k for k in all_fields
            if non_empty_counts.get(canonical_map.get(k, k), 0) == 0
        }
        # Также удаляем поля с заполненностью < 1% (редкие)
        rarely_filled = {
            k for k in all_fields
            if 0 < non_empty_counts.get(canonical_map.get(k, k), 0) / total_examples < 0.01
        }
        fields_to_drop = always_empty | rarely_filled

        filtered_examples: List[Dict] = []
        for ex in normalized_examples:
            filtered_ex = {k: v for k, v in ex.items() if k not in fields_to_drop}
            filtered_examples.append(filtered_ex)

        # 3. Удалить константные колонки
        field_values: Dict[str, Set[str]] = {}
        for ex in filtered_examples:
            for k, v in ex.items():
                if v is not None:
                    field_values.setdefault(k, set()).add(str(v).strip())

        constant_fields = {k for k, vals in field_values.items() if len(vals) <= 1}
        # Если две колонки идентичны — удалить вторую
        identical_pairs = []
        keys = list(field_values.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj = keys[i], keys[j]
                if field_values[ki] == field_values[kj] and len(field_values[ki]) > 0:
                    identical_pairs.append((ki, kj))
        for a, b in identical_pairs:
            if b not in constant_fields:
                constant_fields.add(b)

        # Не удаляем retain_fields и meta_fields из константных
        safe_constant = {
            k for k in constant_fields
            if k not in self.domain.retain_fields and k not in self.domain.meta_fields
        }

        filtered_examples2: List[Dict] = []
        for ex in filtered_examples:
            filtered_ex = {k: v for k, v in ex.items() if k not in safe_constant}
            filtered_examples2.append(filtered_ex)

        # 4. Вычислить visible_count для каждого поля
        visible_counts: Dict[str, int] = {}
        total_count = len(filtered_examples2)
        for ex in filtered_examples2:
            name = ex.get(self.domain.canonicalize_field_name(name_col), "")
            if not name:
                for k in ex:
                    if "наименование" in k.lower() and "полное" not in k.lower():
                        name = str(ex.get(k, ""))
                        break
            if not name and full_name_col:
                name = str(ex.get(self.domain.canonicalize_field_name(full_name_col), ""))
            if not name:
                continue
            for k, v in ex.items():
                if v is None:
                    continue
                if self._is_value_in_name(str(v), name, param_key=k, standard=standard):
                    visible_counts[k] = visible_counts.get(k, 0) + 1

        # 5. Удалить невидимые (visible_count == 0 и не в retain_fields/meta_fields)
        invisible_fields = set()
        for k in list(field_values.keys()):
            if k in safe_constant:
                continue
            vc = visible_counts.get(k, 0)
            if vc == 0 and k not in self.domain.retain_fields and k not in self.domain.meta_fields:
                invisible_fields.add(k)

        final_examples: List[Dict] = []
        for ex in filtered_examples2:
            final_ex = {k: v for k, v in ex.items() if k not in invisible_fields}
            final_examples.append(final_ex)

        # 6. Определить близнецов (Union-Find) — только для полей с visible_count > 0
        twin_groups = self._detect_twin_groups(final_examples, visible_counts)

        # 7. Разрешить близнецов
        resolved_examples: List[Dict] = []
        twin_canonical_map: Dict[str, str] = {}
        for group in twin_groups:
            canonical = group[0]
            for twin in group[1:]:
                twin_canonical_map[twin] = canonical

        for ex in final_examples:
            resolved = {}
            for k, v in ex.items():
                if k in twin_canonical_map:
                    ck = twin_canonical_map[k]
                    if ck in resolved:
                        continue
                    resolved[ck] = v
                else:
                    resolved[k] = v
            resolved_examples.append(resolved)

        # === ИСПРАВЛЕНИЕ: определяем поля, которые уйдут в _meta ===
        # Эти поля не должны участвовать в visible_fields / field_meta для regex
        meta_field_names: Set[str] = set()
        # meta_fields из домена
        for mf in self.domain.meta_fields:
            meta_field_names.add(self.domain.canonicalize_field_name(mf))
        # Ключевые служебные колонки
        meta_field_names.add(self.domain.canonicalize_field_name(name_col))
        if code_col:
            meta_field_names.add(self.domain.canonicalize_field_name(code_col))
        if full_name_col:
            meta_field_names.add(self.domain.canonicalize_field_name(full_name_col))
        meta_field_names.add(self.domain.canonicalize_field_name(type_col))
        # Также любые canonical имена, содержащие "наименование" (кроме тех что уже учтены)
        for can in list(canonical_map.values()):
            if "наименование" in can.lower() or can.lower() in ("стандарт", "нтд", "нтд_1", "standard"):
                meta_field_names.add(can)

        # 8. Сформировать _meta + field_meta (только для значимых полей)
        # Собираем множество canonical имён ВСЕХ удалённых полей
        dropped_canonical: Set[str] = set()
        for orig in all_fields:
            can = canonical_map.get(orig, orig)
            if (orig in fields_to_drop or
                can in safe_constant or
                can in invisible_fields):
                dropped_canonical.add(can)

        field_meta: Dict[str, Dict] = {}
        for orig, can in canonical_map.items():
            if can in dropped_canonical:
                continue
            # ПРОПУСКАЕМ поля, которые уйдут в _meta — они не участвуют в regex
            if can in meta_field_names:
                continue
            # visible_count для этого поля
            vc = visible_counts.get(can, 0)
            is_meta = can in self.domain.meta_fields or can in self.domain.retain_fields
            # Пропускаем поля, которые:
            # - не видны (vc == 0) И
            # - не являются метаданными (retain/meta)
            if vc == 0 and not is_meta:
                continue
            field_meta[can] = {
                "original_name": orig,
                "visible_count": vc,
                "total_count": total_count,
                "is_metadata": is_meta,
            }

        visible_fields = sorted([k for k in field_meta if not field_meta[k]["is_metadata"]])
        metadata_fields = sorted([k for k in field_meta if field_meta[k]["is_metadata"]])

        structured_examples: List[Dict] = []
        for ex in resolved_examples:
            meta: Dict[str, Any] = {
                "standard": standard,
                "item_type": item_type,
            }
            if code_col and code_col in ex:
                meta["ens_code"] = str(ex.pop(code_col, ""))
            for k in list(ex.keys()):
                if "наименование" in k.lower() and "полное" not in k.lower():
                    meta["name"] = ex.pop(k)
                    break
            if full_name_col:
                can_full = self.domain.canonicalize_field_name(full_name_col)
                if can_full in ex:
                    meta["full_name"] = ex.pop(can_full)
            can_type = self.domain.canonicalize_field_name(type_col)
            if can_type in ex:
                meta["item_type"] = ex.pop(can_type)
            for mf in self.domain.meta_fields:
                can_mf = self.domain.canonicalize_field_name(mf)
                if can_mf in ex:
                    meta[can_mf] = ex.pop(can_mf)

            structured_examples.append({
                "_meta": meta,
                **ex,
            })

        stats = {
            "total": total_count,
            "visible_fields": visible_fields,
            "metadata_fields": metadata_fields,
        }

        return {
            "examples": structured_examples,
            "twin_groups": twin_groups,
            "field_meta": field_meta,
            "stats": stats,
        }

    def _detect_twin_groups(self, examples: List[Dict], visible_counts: Dict[str, int]) -> List[List[str]]:
        """Union-Find по visible values (threshold=1.0).

        ИСПРАВЛЕНИЕ: пропускаем поля с visible_count == 0 — у них нет значений
        для сравнения (все None/пустые), и они создают ложные giant clusters.
        """
        # Фильтруем только поля, которые реально видны хотя бы в одном примере
        meaningful_keys = {
            k for k, vc in visible_counts.items()
            if vc > 0 or k in self.domain.retain_fields or k in self.domain.meta_fields
        }

        pair_stats: Dict[Tuple[str, str], List[int]] = {}
        for ex in examples:
            keys = sorted([k for k in ex.keys() if not k.startswith("_") and k in meaningful_keys])
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    # Пропускаем пары, где оба значения пустые
                    va = str(ex.get(a, "")).strip()
                    vb = str(ex.get(b, "")).strip()
                    if not va and not vb:
                        continue
                    pair = tuple(sorted((a, b)))
                    if pair not in pair_stats:
                        pair_stats[pair] = [0, 0]
                    pair_stats[pair][0] += 1
                    if va == vb:
                        pair_stats[pair][1] += 1

        twin_edges = []
        for (a, b), (total, matches) in pair_stats.items():
            if total > 0 and matches / total >= self.domain.twin_threshold:
                twin_edges.append((a, b))

        if not twin_edges:
            return []

        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for a, b in twin_edges:
            union(a, b)

        groups_map: Dict[str, List[str]] = {}
        for node in parent:
            root = find(node)
            groups_map.setdefault(root, []).append(node)

        groups = []
        for members in groups_map.values():
            if len(members) >= 2:
                freq = {m: visible_counts.get(m, 0) for m in members}
                members_sorted = sorted(members, key=lambda m: -freq[m])
                groups.append(members_sorted)

        logger.info("[ENSIndexBuilder] Detected %d twin groups for %s", len(groups), self.domain.domain)
        return groups