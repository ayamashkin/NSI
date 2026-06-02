# =============================================================================
# ФАЙЛ: core/ens_index_builder.py
# ПОСЛЕДНИЕ 5 ИЗМЕНЕНИЙ (МСК, UTC+3):
# 2026-06-02 12:00:00 — FIX: code_col canonicalize — ищем code_canonical в ex, не оригинальное имя
# 2026-06-02 11:45:00 — FEAT: _meta["code"] вместо _meta["ens_code"] — канонический ключ
# 2026-05-28 14:00:00 — FIX: удалён no_sep decimal heuristic (ложное срабатывание "1,5" в "15")
# 2026-05-28 14:00:00 — FIX: все наименование-поля в _meta, не только первое
# 2026-05-28 12:45:00 — FIX: _norm_field_name убирает underscores (skip_fields работают)
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
        # REMOVED: no_sep heuristic caused false positives (e.g., "1,5" matched "15")
        # if re.match(r"^\d+[.,]\d+$", val_raw):
        #     no_sep = re.sub(r"[.,]", "", val_str)
        #     if no_sep in name_lower:
        #         return True
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

        index: Dict[str, Dict[str, Any]] = {}
        min_examples = self.domain.min_examples
        for (std, itype), examples in groups.items():
            if len(examples) < min_examples:
                logger.debug("[ENSIndexBuilder] Skipping %s / %s: %d examples < min %d",
                             std, itype, len(examples), min_examples)
                continue
            built = self._build_standard_type(std, itype, examples, name_col, code_col, full_name_col, type_col)
            if built:
                if std not in index:
                    index[std] = {}
                index[std][itype] = built
                self._print_group_stats(std, itype, built)

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
        """Вывести статистику по сформированной группе (стандарт, тип)."""
        stats = built.get("stats", {})
        field_meta = built.get("field_meta", {})
        twin_groups = built.get("twin_groups", [])
        examples = built.get("examples", [])
        total = stats.get("total", len(examples))
        visible_fields = stats.get("visible_fields", [])
        metadata_fields = stats.get("metadata_fields", [])
        drop_reasons = built.get("drop_reasons", {})

        meaningful_twins = [g for g in twin_groups if len(g) <= 5]

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

        if drop_reasons:
            click.echo(f"\n 🗑️ Удалённые поля ({len(drop_reasons)}):")
            for f in sorted(drop_reasons.keys()):
                reason = drop_reasons[f]
                click.echo(f"   {f:30s} → {reason}")

        click.echo(f"{'=' * 60}")

    @staticmethod
    def _norm_field_name(name: str) -> str:
        """Нормализация имени поля для сравнения (skip_fields matching).

        ИСПРАВЛЕНИЕ: удаляем _ тоже, иначе skip_fields с подчёркиваниями
        не совпадают с оригинальными именами колонок Excel.
        """
        return re.sub(r"[_\W]", "", str(name).lower().strip())

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

        # 0. Нормализованный skip_fields
        skip_normalized = {self._norm_field_name(f) for f in self.domain.skip_fields}

        # 1. Нормализация заголовков
        canonical_map: Dict[str, str] = {}  # original -> canonical
        all_fields = set()
        for ex in examples:
            all_fields.update(ex.keys())

        all_fields = {f for f in all_fields if self._norm_field_name(f) not in skip_normalized}

        for field in all_fields:
            canonical = self.domain.canonicalize_field_name(field)
            if canonical in canonical_map.values():
                base = canonical
                i = 1
                while canonical in canonical_map.values():
                    canonical = f"{base}_{i}"
                    i += 1
            canonical_map[field] = canonical

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

        # Собираем причины удаления полей
        drop_reasons: Dict[str, str] = {}

        # 2. Удалить всегда пустые / редкие колонки (<1%)
        non_empty_counts: Dict[str, int] = {}
        for ex in normalized_examples:
            for k, v in ex.items():
                if v is not None and str(v).strip() not in ("", " ", "0", "0.0", "None"):
                    non_empty_counts[k] = non_empty_counts.get(k, 0) + 1

        total_examples = len(normalized_examples)
        canonical_keys = set(canonical_map.values())
        always_empty = {k for k in canonical_keys if non_empty_counts.get(k, 0) == 0}
        rarely_filled = {k for k in canonical_keys if 0 < non_empty_counts.get(k, 0) / total_examples < 0.01}
        fields_to_drop = always_empty | rarely_filled

        for k in always_empty:
            drop_reasons[k] = "always_empty"
        for k in rarely_filled:
            drop_reasons[k] = f"rarely_filled ({non_empty_counts.get(k,0)}/{total_examples} < 1%)"

        filtered_examples: List[Dict] = []
        for ex in normalized_examples:
            filtered_ex = {k: v for k, v in ex.items() if k not in fields_to_drop}
            filtered_examples.append(filtered_ex)

        # 3. ПРЕДВАРИТЕЛЬНЫЙ visible_count — ДО удаления констант
        #    Это нужно, чтобы защитить константные поля, которые реально видны в наименованиях
        visible_counts_pre: Dict[str, int] = {}
        for ex in filtered_examples:
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
                    visible_counts_pre[k] = visible_counts_pre.get(k, 0) + 1

        # 4. Удалить константные колонки — ТОЛЬКО если они НЕ видны в наименованиях
        field_values: Dict[str, Set[str]] = {}
        for ex in filtered_examples:
            for k, v in ex.items():
                if v is not None:
                    field_values.setdefault(k, set()).add(str(v).strip())

        constant_fields = {k for k, vals in field_values.items() if len(vals) <= 1}

        # Идентичные пары — удаляем вторую только если она тоже не видна
        identical_pairs = []
        keys = list(field_values.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj = keys[i], keys[j]
                if field_values[ki] == field_values[kj] and len(field_values[ki]) > 0:
                    # Удаляем b только если оно не видно (или оба не видны)
                    # Если b видно — оставляем, т.к. оно нужно для regex
                    if visible_counts_pre.get(kj, 0) == 0:
                        identical_pairs.append((ki, kj))
                    elif visible_counts_pre.get(ki, 0) == 0:
                        identical_pairs.append((kj, ki))

        for a, b in identical_pairs:
            if b not in constant_fields:
                constant_fields.add(b)

        # КОНЦЕПЦИЯ: константное поле удаляем ТОЛЬКО если:
        #   - оно НЕ в retain_fields / meta_fields
        #   - оно НЕ видно в наименованиях (visible_counts_pre == 0)
        safe_constant = {
            k for k in constant_fields
            if k not in self.domain.retain_fields
            and k not in self.domain.meta_fields
            and visible_counts_pre.get(k, 0) == 0
        }

        for k in safe_constant:
            orig = ""
            for o, c in canonical_map.items():
                if c == k:
                    orig = o
                    break
            drop_reasons[k] = f"constant (value={list(field_values.get(k, {'?'}))[0]}) but NOT visible in names"

        # Константные поля, которые ОСТАВЛЕНЫ (visible или retain), логируем отдельно
        kept_constant = constant_fields - safe_constant
        for k in kept_constant:
            vc = visible_counts_pre.get(k, 0)
            reason = f"constant KEPT — visible in {vc} names"
            if k in self.domain.retain_fields:
                reason += " [retain_fields]"
            if k in self.domain.meta_fields:
                reason += " [meta_fields]"
            logger.debug("[ENSIndexBuilder] %s / %s: field '%s' %s", standard, item_type, k, reason)

        filtered_examples2: List[Dict] = []
        for ex in filtered_examples:
            filtered_ex = {k: v for k, v in ex.items() if k not in safe_constant}
            filtered_examples2.append(filtered_ex)

        # 5. ПЕРЕСЧЁТ visible_count после удаления константных
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

        # 6. Удалить невидимые (visible_count == 0 и не в retain_fields/meta_fields)
        invisible_fields = set()
        for k in list(field_values.keys()):
            if k in safe_constant:
                continue
            vc = visible_counts.get(k, 0)
            if vc == 0 and k not in self.domain.retain_fields and k not in self.domain.meta_fields:
                invisible_fields.add(k)

        for k in invisible_fields:
            drop_reasons[k] = drop_reasons.get(k, "") + "invisible (value never found in names); "

        final_examples: List[Dict] = []
        for ex in filtered_examples2:
            final_ex = {k: v for k, v in ex.items() if k not in invisible_fields}
            final_examples.append(final_ex)

        # 7. Определить близнецов
        twin_groups = self._detect_twin_groups(final_examples, visible_counts)

        # 8. Разрешить близнецов
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

        # 9. Определить meta поля
        meta_field_names: Set[str] = set()
        for mf in self.domain.meta_fields:
            meta_field_names.add(self.domain.canonicalize_field_name(mf))
        meta_field_names.add(self.domain.canonicalize_field_name(name_col))
        if code_col:
            meta_field_names.add(self.domain.canonicalize_field_name(code_col))
        if full_name_col:
            meta_field_names.add(self.domain.canonicalize_field_name(full_name_col))
        meta_field_names.add(self.domain.canonicalize_field_name(type_col))
        for can in list(canonical_map.values()):
            if "наименование" in can.lower() or can.lower() in ("стандарт", "нтд", "нтд_1", "standard"):
                meta_field_names.add(can)

        # 10. field_meta (только значимые поля)
        dropped_canonical: Set[str] = set()
        for orig in all_fields:
            can = canonical_map.get(orig, orig)
            if (can in fields_to_drop or can in safe_constant or can in invisible_fields):
                dropped_canonical.add(can)

        field_meta: Dict[str, Dict] = {}
        for orig, can in canonical_map.items():
            if can in dropped_canonical:
                continue
            if can in meta_field_names:
                continue
            vc = visible_counts.get(can, 0)
            is_meta = can in self.domain.meta_fields or can in self.domain.retain_fields
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

        # FEAT 2026-06-02: use canonical key for code field in _meta
        code_canonical = self.domain.canonicalize_field_name(code_col) if code_col else None

        structured_examples: List[Dict] = []
        for ex in resolved_examples:
            meta: Dict[str, Any] = {
                "standard": standard,
                "item_type": item_type,
            }
            # FIX 2026-06-02: use canonical name, not original Excel column name
            if code_canonical and code_canonical in ex:
                meta["code"] = str(ex.pop(code_canonical, ""))
            # Remove ALL наименование fields to _meta, not just first
            name_keys = [k for k in list(ex.keys()) if "наименование" in k.lower() and "полное" not in k.lower()]
            for k in name_keys:
                if "name" not in meta:
                    meta["name"] = ex.pop(k)
                else:
                    ex.pop(k, None)
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
            "drop_reasons": drop_reasons,
        }

    def _detect_twin_groups(self, examples: List[Dict], visible_counts: Dict[str, int]) -> List[List[str]]:
        """Union-Find по visible values (threshold=1.0)."""
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