# =============================================================================
# FILE: core/llm_mask_generator.py
# REPO: https://github.com/ayamashkin/NSI
# LAST 5 CHANGES (UTC+3):
# 2026-05-27 14:15:00 — Рефакторинг под доменную архитектуру ENS
# 2026-05-27 14:15:00 — Удалены захардкоженные SERVICE_FIELDS и FIELD_NAME_MAP
# 2026-05-27 14:15:00 — Добавлена загрузка структурированного индекса ens_{domain}.pkl
# 2026-05-27 14:15:00 — _detect_twin_groups и _resolve_twins удалены (читаем из индекса)
# 2026-05-27 14:15:00 — _canonicalize_field_name и _auto_detect_visible удалены
# =============================================================================
"""
LLM Mask Generator Module (Domain-based)
Generates regex masks using LLM with pre-built ENS domain index.
"""
import json
import logging
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.standard_utils import canonicalize_standard

logger = logging.getLogger(__name__)


@dataclass
class MaskGenerationResult:
    pattern: str = ""
    params: List[str] = field(default_factory=list)
    required: List[str] = field(default_factory=list)
    standard: str = ""
    item_type: str = ""
    raw_response: str = ""
    service: str = ""
    model: str = ""
    temperature: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


class LLMMaskGenerator:
    """Генератор масок через LLM с доменным ENS-индексом."""

    # Параметры, которые никогда не должны попадать в regex (даже если видны)
    SKIP_PARAMS = {
        "марка_материала", "толщина_покрытия", "наличие_бп",
        "автор_последнего_изменения", "дата_последнего_изменения",
    }

    def __init__(
        self,
        clients: Dict[str, Any],
        settings: Any = None,
        max_retries: int = 3,
        domain: str = "hardware",
        ens_index_path: Optional[str] = None,
    ):
        self.clients = clients
        self.settings = settings
        self.max_retries = max_retries
        self.domain = domain
        self.validator = None
        self._domain_index: Optional[Dict] = None
        self._ens_index_path = ens_index_path or f"cache/ens_{domain}.pkl"
        logger.info("[LLMMaskGenerator] Initialized domain=%s index=%s", domain, self._ens_index_path)

    def _load_domain_index(self) -> Dict:
        """Загрузить структурированный доменный индекс."""
        if self._domain_index is not None:
            return self._domain_index
        path = Path(self._ens_index_path)
        if not path.exists():
            logger.warning("[LLMMaskGenerator] Domain index not found: %s", path)
            self._domain_index = {}
            return self._domain_index
        try:
            with open(path, "rb") as f:
                self._domain_index = pickle.load(f)
            count = sum(len(v) for v in self._domain_index.values())
            logger.info("[LLMMaskGenerator] Loaded domain index: %d standards, %d types",
                        len(self._domain_index), count)
        except Exception as e:
            logger.error("[LLMMaskGenerator] Failed to load domain index: %s", e)
            self._domain_index = {}
        return self._domain_index

    def _get_index_entry(self, standard: str, item_type: str) -> Optional[Dict]:
        """Получить запись индекса для (standard, item_type)."""
        index = self._load_domain_index()
        std = canonicalize_standard(standard)
        itype = item_type.strip()
        if std in index and itype in index[std]:
            return index[std][itype]
        # fuzzy fallback
        for s in index:
            if std in s or s in std:
                for t in index[s]:
                    if itype.lower() == t.lower():
                        return index[s][t]
        return None

    def _get_ens_examples(self, standard: str, item_type: str, max_examples: int = 20) -> List[Dict]:
        """Получить примеры из доменного индекса."""
        entry = self._get_index_entry(standard, item_type)
        if not entry:
            # Fallback: legacy validator
            validator = self._get_validator()
            if validator:
                try:
                    return validator._get_ens_examples(standard, item_type)[:max_examples]
                except Exception as e:
                    logger.warning("[LLMMaskGenerator] Fallback examples failed: %s", e)
            return []
        examples = entry.get("examples", [])
        return examples[:max_examples]

    def _get_twin_groups(self, standard: str, item_type: str) -> List[List[str]]:
        """Получить twin_groups из индекса."""
        entry = self._get_index_entry(standard, item_type)
        if entry:
            return entry.get("twin_groups", [])
        return []

    def _get_field_meta(self, standard: str, item_type: str) -> Dict[str, Dict]:
        """Получить field_meta из индекса."""
        entry = self._get_index_entry(standard, item_type)
        if entry:
            return entry.get("field_meta", {})
        return {}

    def _get_visible_params_from_index(self, standard: str, item_type: str) -> Tuple[set, set]:
        """Получить required/optional из индекса (visible_fields)."""
        entry = self._get_index_entry(standard, item_type)
        if not entry:
            return {"тип_изделия", "нтд_1"}, set()
        stats = entry.get("stats", {})
        visible = set(stats.get("visible_fields", []))
        # metadata fields не участвуют в regex
        metadata = set(stats.get("metadata_fields", []))
        visible = visible - metadata - self.SKIP_PARAMS
        # Разделяем на required (видны в >=85%) и optional
        total = stats.get("total", 0)
        if total == 0:
            return visible, set()
        field_meta = entry.get("field_meta", {})
        required = set()
        optional = set()
        for f in visible:
            vc = field_meta.get(f, {}).get("visible_count", 0)
            ratio = vc / total
            if ratio >= 0.85:
                required.add(f)
            else:
                optional.add(f)
        required.add("тип_изделия")
        required.add("нтд_1")
        return required, optional

    def _get_validator(self):
        """Lazy init legacy validator."""
        if self.validator is None:
            try:
                from core.auto_validator import AutoValidator
                ens_path = None
                if self.settings and hasattr(self.settings, "database"):
                    ens_path = getattr(self.settings.database, "ens_index_path", None)
                if not ens_path:
                    ens_path = self._ens_index_path
                self.validator = AutoValidator(ens_index_path=ens_path, activation_threshold=0.85)
            except Exception as e:
                logger.warning("[LLMMaskGenerator] Failed to init validator: %s", e)
        return self.validator

    @staticmethod
    def _is_value_in_name(val: str, name: str, param_key: str = "", standard: str = "") -> bool:
        """Проверить, что значение параметра присутствует в строке номенклатуры."""
        if not val or not name:
            return False
        if param_key in {"марка_материала", "толщина_покрытия", "наличие_бп",
                         "автор_последнего_изменения", "дата_последнего_изменения"}:
            return False
        name_clean = name
        if standard:
            name_clean = re.sub(
                r'ОСТ\s*\d+\s*\d+-\d+|ГОСТ\s*\d+-\d+',
                '', name_clean, flags=re.IGNORECASE
            )
        val_raw = str(val).strip()
        val_str = val_raw.lower().replace(",", ".")
        name_lower = name_clean.lower().replace(",", ".")
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

    def _filter_unambiguous(
        self,
        examples: List[Dict],
        twin_groups: List[List[str]],
        standard: str = "",
    ) -> Tuple[List[Tuple[Dict, Dict[str, str]]], List[Tuple[Dict, Dict[str, str]]]]:
        """Разделить примеры на однозначные и неоднозначные."""
        unambiguous = []
        ambiguous = []
        for ex in examples:
            # visible = все поля кроме _meta и SKIP_PARAMS, которые есть в name
            name = ex.get("_meta", {}).get("name", "")
            if not name:
                name = ex.get("_meta", {}).get("full_name", "")
            vis: Dict[str, str] = {}
            for k, v in ex.items():
                if k.startswith("_"):
                    continue
                if v is None or str(v).strip() == "":
                    continue
                if k in self.SKIP_PARAMS:
                    continue
                if self._is_value_in_name(str(v), name, param_key=k, standard=standard):
                    vis[k] = str(v)
            # resolve twins
            twin_map = {}
            for group in twin_groups:
                can = group[0]
                for twin in group[1:]:
                    twin_map[twin] = can
            resolved = {}
            for k, v in vis.items():
                if k in twin_map:
                    ck = twin_map[k]
                    if ck in resolved:
                        continue
                    resolved[ck] = v
                else:
                    resolved[k] = v
            values = list(resolved.values())
            if len(values) != len(set(values)):
                ambiguous.append((ex, resolved))
            else:
                unambiguous.append((ex, resolved))
        logger.info("[LLMMaskGenerator] Unambiguous: %d, Ambiguous: %d", len(unambiguous), len(ambiguous))
        return unambiguous, ambiguous

    def _get_global_visible(
        self,
        unambiguous: List[Tuple[Dict, Dict[str, str]]],
        threshold: float = 0.85,
    ) -> Tuple[set, set]:
        if not unambiguous:
            return {"тип_изделия", "нтд_1"}, set()
        total = len(unambiguous)
        param_counts: Dict[str, int] = {}
        for ex, vis in unambiguous:
            for key in vis:
                param_counts[key] = param_counts.get(key, 0) + 1
        required = set()
        optional = set()
        for key, count in param_counts.items():
            ratio = count / total
            if ratio >= threshold:
                required.add(key)
            elif ratio >= 0.20:
                optional.add(key)
        required.add("тип_изделия")
        required.add("нтд_1")
        return required, optional

    def _format_stats(
        self,
        examples: List[Dict],
        standard: str = "",
    ) -> str:
        if not examples:
            return "(нет данных)"
        twin_groups = self._get_twin_groups(standard, examples[0].get("_meta", {}).get("item_type", ""))
        unambiguous, _ = self._filter_unambiguous(examples, twin_groups, standard=standard)
        required, optional = self._get_global_visible(unambiguous)
        global_visible = required | optional
        param_counts: Dict[str, int] = {}
        for ex, vis in unambiguous:
            for key in vis:
                if key in global_visible:
                    param_counts[key] = param_counts.get(key, 0) + 1
        total = len(examples)
        lines = []
        for key, count in sorted(param_counts.items(), key=lambda x: -x[1]):
            lines.append(f" {key}: {count} из {total} ({count/total*100:.0f}%)")
        return "\n".join(lines) if lines else "(нет параметров)"

    def _select_representative_examples(self, examples: List[Dict], max_count: int = 10) -> List[Dict]:
        if len(examples) <= max_count:
            return examples

        def _visible_params(ex: Dict) -> set:
            name = ex.get("_meta", {}).get("name", "")
            if not name:
                name = ex.get("_meta", {}).get("full_name", "")
            vis = set()
            for k, v in ex.items():
                if k.startswith("_"):
                    continue
                if v is None or str(v).strip() == "":
                    continue
                if self._is_value_in_name(str(v), name, param_key=k):
                    vis.add(k)
            return vis

        scored = [(ex, _visible_params(ex)) for ex in examples]
        scored.sort(key=lambda x: -len(x[1]))
        selected = []
        covered = set()
        for ex, vis in scored:
            if len(selected) >= max_count:
                break
            new_params = vis - covered
            if new_params or len(selected) < max_count:
                selected.append(ex)
                covered |= vis
        if len(selected) < max_count:
            used = set(id(e) for e in selected)
            for ex, _ in scored:
                if id(ex) not in used:
                    selected.append(ex)
                    if len(selected) >= max_count:
                        break
        return selected

    def _format_examples(
        self,
        examples: List[Dict],
        standard: str,
        item_type: str,
    ) -> str:
        if not examples:
            return "(примеры отсутствуют)"

        display_examples = self._select_representative_examples(examples, max_count=10)
        twin_groups = self._get_twin_groups(standard, item_type)
        unambiguous, _ = self._filter_unambiguous(examples, twin_groups, standard=standard)
        required, optional = self._get_global_visible(unambiguous)
        global_visible = required | optional

        lines = []
        lines.append(f"Структура: <{item_type}> [исполнение] <параметры> <покрытие> {standard}")
        lines.append("")
        lines.append(f"Параметры, участвующие в наименовании: {', '.join(sorted(global_visible))}")
        lines.append("")

        for i, ex in enumerate(display_examples, 1):
            meta = ex.get("_meta", {})
            name = meta.get("name", meta.get("full_name", ""))
            if not name:
                continue

            # visible для этого примера
            vis: Dict[str, str] = {}
            for k, v in ex.items():
                if k.startswith("_"):
                    continue
                if v is None or str(v).strip() == "":
                    continue
                if self._is_value_in_name(str(v), name, param_key=k, standard=standard):
                    vis[k] = str(v)

            # resolve twins
            twin_map = {}
            for group in twin_groups:
                can = group[0]
                for twin in group[1:]:
                    twin_map[twin] = can
            resolved = {}
            for k, v in vis.items():
                if k in twin_map:
                    ck = twin_map[k]
                    if ck in resolved:
                        continue
                    resolved[ck] = v
                else:
                    resolved[k] = v

            # неоднозначные ключи
            val_to_keys: Dict[str, List[str]] = {}
            for k, v in resolved.items():
                val_to_keys.setdefault(v, []).append(k)
            ambiguous_keys = set()
            for v, keys in val_to_keys.items():
                if len(keys) >= 2:
                    for k in keys:
                        ambiguous_keys.add(k)

            visible_list = []
            missing_list = []
            ambiguous_list = []

            for key in sorted(global_visible):
                if key == "тип_изделия":
                    val = meta.get("item_type", "")
                    if val and name.lower().startswith(str(val).strip().lower()):
                        visible_list.append((key, str(val).strip(), 0))
                    else:
                        missing_list.append(key)
                    continue
                if key not in resolved:
                    missing_list.append(key)
                    continue
                val_str = resolved[key]
                is_in_name = self._is_value_in_name(val_str, name, param_key=key, standard=standard)
                if key in ambiguous_keys:
                    ambiguous_list.append((key, val_str))
                elif is_in_name:
                    pos = name.lower().find(val_str.lower())
                    if pos < 0:
                        m = re.search(r"[a-zA-Zа-яА-Я0-9]+", val_str)
                        if m:
                            pos = name.lower().find(m.group().lower())
                        if pos < 0:
                            pos = 999
                    visible_list.append((key, val_str, pos))
                else:
                    missing_list.append(key)

            visible_list.sort(key=lambda x: x[2])

            lines.append(f'{i}. Исходное: "{name}"')
            if visible_list:
                vis_str = " ".join([f"(?P<{k}>{v})" for k, v, _ in visible_list])
                lines.append(f" Видимые: {vis_str}")
            if ambiguous_list:
                amb_str = " ".join([f"(?P<{k}>{v})" for k, v in ambiguous_list])
                lines.append(f" Неоднозначные: {amb_str}")
            if missing_list:
                lines.append(f" Отсутствуют: {', '.join(missing_list)}")
            lines.append("")

        return "\n".join(lines)

    def _get_prompt_template(self) -> str:
        if self.settings and hasattr(self.settings, "mask_generation"):
            mg = self.settings.mask_generation
            template_path = getattr(mg, "prompt_template", "")
            if template_path:
                p = Path(template_path)
                if p.exists():
                    return p.read_text(encoding="utf-8")
            for path in [
                "prompts/templates/mask_generation.txt",
                "prompts/mask_generation.txt",
                "config/mask_generation.txt",
            ]:
                p = Path(path)
                if p.exists():
                    return p.read_text(encoding="utf-8")
        return self._default_template()

    def _default_template(self) -> str:
        return r"""Ты — эксперт по техническим стандартам ГОСТ/ОСТ/ТУ и регулярным выражениям Python 3 (re модуль).

=== ЗАДАЧА ===

Создай regex-паттерн с named groups (?P...) для извлечения параметров из строки номенклатуры.

=== ПРИМЕРЫ ИЗ ЕНС ===

{examples_text}

=== СТАТИСТИКА ПАРАМЕТРОВ ===

{stats_text}

=== ЖЁСТКИЕ ЗАПРЕТЫ (нарушение = брак) ===

1. **ТОЛЬКО ВИДИМЫЕ ПАРАМЕТРЫ**. Named group создаётся ТОЛЬКО если значение реально присутствует в исходной строке номенклатуры.
2. **ИМЯ ГРУППЫ ТИПА ИЗДЕЛИЯ — СТРОГО `тип_изделия`**. Не `наименование_типа`, не `тип`, не `вид`. Только `тип_изделия`.
3. **НЕТ ГРУППЕ `исполнение`**, если в примерах нет вариантов с `(N)` или `N-` после типа изделия. Если исполнение есть — группа опциональная `(?:\(?(?P<исполнение>\d+)\)?)?`.
4. **НЕТ ГРУППЕ `технические_характеристики` как отдельной**. Если в строке только покрытие (Бп, Кд, Хим.Пас) — используй только `покрытие`.
5. **НЕТ `standard` в `required`**. `standard` — метаданные, не извлекается из строки.
6. **НЕТ `толщина_покрытия` в regex**, если в строке нет явного числа толщины.

=== ПРАВИЛА РАЗДЕЛИТЕЛЕЙ ===

7. Разделители между параметрами: `[-\s]+` (тире, пробел, таб). Не используй `\s*` — оно не матчит тире!
8. Десятичная точка/запятая: `(?:[.,]\d+)?` только внутри числовой группы. Пример: `\d+(?:[.,]\d+)?` для 12.5.

=== КИРИЛЛИЦА В REGEX ===

9. **Шаг резьбы**: используй `[xXхХ×]` (включая кириллические х/Х и × U+00D7). Пример: `M12х1.5` должно матчиться.
10. **Покрытие**: `[\w.]+` матчит кириллицу (включая "ОСТ"!). Поэтому покрытие должно строго предшествовать `нтд_1` в паттерне, или использовать негативный lookahead `(?!\w)`.
11. **Класс допуска/прочности**: используй `[a-zA-Zа-яА-Я]+` вместо `[a-z]+` — в номенклатуре могут быть кириллические буквы (6г, 6Н).

=== СТРУКТУРА ПАТТЕРНА ===

12. Порядок групп: `тип_изделия` → `исполнение` (опц.) → числовые параметры → `покрытие` → `нтд_1`.
13. Полная строка: `^...$` обязательно.
14. Имена групп ≤30 символов, только [a-zA-Zа-яА-Я0-9_].
15. **ЗАПРЕЩЕНЫ nested named groups**: `(?P(?P...))` — НЕВАЛИДНЫ в Python re. Используй `(?:...)` для вложенных групп.
16. **ЗАПРЕЩЕНЫ unbalanced parentheses**: каждая `(` должна иметь парную `)`.

=== ПРАВИЛО ТОЧКИ ===

17. Точка `.` в номенклатуре:
   - ДЕСЯТИЧНАЯ: если дробная часть имеет смысл (12.5 мм).
   - РАЗДЕЛИТЕЛЬ: если после точки ровно 2 цифры-кода (100.58 → длина=100, группа=58).
   - **ПРАВИЛО**: при сомнении разделяй: `(?P<длина>\d+)\.(?P<группа>\d+)` вместо `(?P<длина>\d+(?:[.,]\d+)?)`.

=== КРИТИЧНО: ФОРМАТ НТД_1 ===

18. **Группа `нтд_1` ДОЛЖНА матчить ПОЛНОЕ название стандарта** из заголовка этого промпта.
   - Для ОСТ: `(?P<нтд_1>ОСТ\s*1\s*\d+-\d+)` или `(?P<нтд_1>ОСТ\s*1\s*33049-80)`
   - Для ГОСТ: `(?P<нтд_1>ГОСТ\s*\d+-\d+)` или `(?P<нтд_1>ГОСТ\s*7795-70)`
   - **ЗАПРЕЩЕНО** использовать `\d+` вместо `ОСТ`/`ГОСТ` — это сломает парсинг.

=== ПРАВИЛО ОПЦИОНАЛЬНЫХ ПАРАМЕТРОВ ===

В примерах ниже параметры помечены как:
- Видимые: (?P<name>value) — присутствуют в этом примере
- Отсутствуют: name1, name2 — есть в других примерах, но не в этом

Если параметр "Отсутствует" в части примеров — он ОПЦИОНАЛЬНЫЙ.
Используй (?P<...>)? для опциональных параметров.

=== ПРИМЕРЫ ПРАВИЛЬНЫХ ПАТТЕРНОВ ===

**ОСТ 1 33049-80 / Гайка** (без шага, без исполнения):
```
^(?P<тип_изделия>Гайка)[-\s]+(?P<номинальный_диаметр_резьбы>\d+)(?:[xXхХ×](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?[-\s]+(?P<покрытие>[\w.]+)[-\s]+(?P<нтд_1>ОСТ\s*1\s*33049-80)$
```

**ГОСТ 7795-70 / Болт** (с классом допуска, длиной, покрытием):
```
^(?P<тип_изделия>Болт)[-\s]+M(?P<номинальный_диаметр_резьбы>\d+)(?:[xXхХ×](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?[-\s]+(?P<класс_поле_допуска>\d+[a-zA-Zа-яА-Я]+)[xXхХ×](?P<длина>\d+(?:[.,]\d+)?)[-\s]+(?P<покрытие>[\w.]+)?[-\s]*(?P<нтд_1>ГОСТ\s*7795-70)$
```

**ОСТ 1 31133-80 / Болт** (с исполнением, покрытием):
```
^(?P<тип_изделия>Болт)[-\s]+(?:\(?(?P<исполнение>\d+)\)?)?[-\s]+(?P<наружный_диаметр>\d+)[-\s]+(?P<длина>\d+(?:[.,]\d+)?)[-\s]+(?P<покрытие>[\w.]+)[-\s]+(?P<нтд_1>ОСТ\s*1\s*31133-80)$
```

=== ФОРМАТ ОТВЕТА ===

```json
{
  "pattern": "^...$",
  "params": ["тип_изделия", "номинальный_диаметр_резьбы", "покрытие", "нтд_1", "шаг_резьбы"],
  "required": ["тип_изделия", "номинальный_диаметр_резьбы", "покрытие", "нтд_1"]
}
```

- `params`: ВСЕ named group имена (только видимые в строке).
- `required`: обязательные параметры (все кроме опциональных: исполнение, шаг_резьбы).

=== ПРОВЕРКА ПЕРЕД ОТВЕТОМ ===

Перед выводом JSON проверь:
- [ ] Все группы из `params` реально видны в примерах?
- [ ] `тип_изделия` — первый и обязательный?
- [ ] `standard` НЕТ в `required`?
- [ ] `исполнение` есть только если в примерах есть `(N)`?
- [ ] Шаг резьбы использует `[xXхХ×]`?
- [ ] Покрытие не перехватывает `ОСТ`/`ГОСТ`?
- [ ] `нтд_1` содержит полное название стандарта (`ОСТ 1 XXXXX-80` или `ГОСТ XXXX-XX`), а НЕ `\d+`?
- [ ] `^` и `$` присутствуют?
- [ ] НЕТ nested named groups `(?P(?P...))`?
- [ ] НЕТ unbalanced parentheses?
- [ ] Pattern компилируется в Python re без ошибок?

=== СТРОГОЕ СООТВЕТСТВИЕ ===

Выведи ОДИН JSON-объект. Без комментариев, markdown, объяснений — только JSON."""

    def _build_prompt(self, standard: str, item_type: str, examples: List[Dict],
                      name: str = "", standard_info: Any = None) -> str:
        template = self._get_prompt_template()
        examples_text = self._format_examples(examples, standard, item_type)
        stats_text = self._format_stats(examples, standard)
        service, model, temperature = self._resolve_service()
        replacements = {
            "{examples_text}": examples_text,
            "{stats_text}": stats_text,
            "{item_type}": item_type,
            "{standard}": standard,
            "{provider}": service or "LLM",
            "{model}": model or "unknown",
            "{temperature}": str(temperature),
            "{timestamp}": datetime.now().isoformat(),
        }
        for placeholder, value in replacements.items():
            if placeholder in template:
                template = template.replace(placeholder, value)
        if "{params_list}" in template:
            visible = self._extract_visible_params(examples)
            template = template.replace("{params_list}", json.dumps(visible, ensure_ascii=False))
        if "{required_list}" in template:
            visible = self._extract_visible_params(examples)
            optional = {"исполнение", "шаг_резьбы", "толщина_покрытия", "variant"}
            required = [p for p in visible if p not in optional]
            template = template.replace("{required_list}", json.dumps(required, ensure_ascii=False))
        has_task = "=== ЗАДАЧА ===" in template or "ЗАДАЧА:" in template.lower()
        has_format = "=== ФОРМАТ ОТВЕТА ===" in template or "```json" in template
        task_section = ""
        if not has_task:
            task_section = f"""
=== ЗАДАЧА ===

Создай regex-паттерн для стандарта {standard}, тип изделия {item_type}.
Используй ВИДИМЫЕ параметры из примеров выше."""
        format_section = ""
        if not has_format:
            format_section = """

=== ФОРМАТ ОТВЕТА ===

```json
{
  "pattern": "^...$",
  "params": ["тип_изделия", ...],
  "required": ["тип_изделия", ...]
}
```

Только JSON, без комментариев."""
        header = f"""# Тип изделия: {item_type}
# Стандарт: {standard}
# Провайдер: {service or 'LLM'}
# Модель: {model or 'unknown'}
# Температура: {temperature}
# Время: {datetime.now().isoformat()}
# =================================================="""
        prompt = header + "\n" + template + task_section + format_section
        return prompt

    def _extract_visible_params(self, examples: List[Dict]) -> List[str]:
        if not examples:
            return ["тип_изделия", "номинальный_диаметр_резьбы", "покрытие", "нтд_1"]
        twin_groups = self._get_twin_groups(
            examples[0].get("_meta", {}).get("standard", ""),
            examples[0].get("_meta", {}).get("item_type", "")
        )
        unambiguous, _ = self._filter_unambiguous(examples, twin_groups)
        required, optional = self._get_global_visible(unambiguous)
        return list(required | optional)

    def _get_debug_dir(self) -> Optional[Path]:
        if not self.settings:
            return None
        mg = getattr(self.settings, "mask_generation", None)
        if not mg:
            return None
        if not getattr(mg, "save_debug_prompts", False):
            return None
        debug_dir = getattr(mg, "debug_prompts_dir", "prompts/debug")
        if not debug_dir:
            return None
        return Path(debug_dir)

    def _save_debug_prompt(self, standard: str, item_type: str, prompt: str) -> None:
        base_dir = self._get_debug_dir()
        if not base_dir:
            return
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{item_type}_{standard}.txt"
            path = base_dir / fname
            path.write_text(prompt, encoding='utf-8')
            logger.debug("[LLMMaskGenerator] Prompt saved to %s", path)
        except Exception as e:
            logger.debug("[LLMMaskGenerator] Failed to save prompt: %s", e)

    def _save_debug_response(self, standard: str, item_type: str, response: str,
                             service: str, attempt: int) -> None:
        base_dir = self._get_debug_dir()
        if not base_dir:
            return
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{item_type}_{standard}_a{attempt}.txt"
            path = base_dir / fname
            svc, model, temp = self._resolve_service()
            raw_content = response
            for prefix in ["```json", "```python", "```"]:
                if raw_content.startswith(prefix):
                    raw_content = raw_content[len(prefix):].strip()
                    break
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3].strip()
            lines = [
                f"# Тип изделия: {item_type}",
                f"# Стандарт: {standard}",
                f"# Провайдер: {svc or 'LLM'}",
                f"# Модель: {model or 'unknown'}",
                f"# Температура: {temp}",
                f"# Время: {datetime.now().isoformat()}",
                "# ==================================================",
                "",
                raw_content,
            ]
            path.write_text("\n".join(lines), encoding='utf-8')
            logger.debug("[LLMMaskGenerator] Response saved to %s", path)
        except Exception as e:
            logger.debug("[LLMMaskGenerator] Failed to save response: %s", e)

    def generate_mask(
        self,
        standard: str,
        item_type: str,
        examples: Optional[List[Dict]] = None,
        name: str = "",
        standard_info: Any = None,
    ) -> Tuple[Optional[MaskGenerationResult], Optional[Dict]]:
        canon_std = canonicalize_standard(standard)
        if examples is None:
            examples = self._get_ens_examples(canon_std, item_type, max_examples=20)
        prompt = self._build_prompt(canon_std, item_type, examples, name, standard_info)
        self._save_debug_prompt(canon_std, item_type, prompt)
        service, model, temperature = self._resolve_service()
        logger.info("[LLMMaskGenerator] Generating mask for %s/%s via %s (examples=%d)",
                    canon_std, item_type, service, len(examples))
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            for svc_name, client in self.clients.items():
                try:
                    result = self._call_llm(client, prompt, model, temperature)
                    if result:
                        self._save_debug_response(canon_std, item_type, result["text"], svc_name, attempt)
                        mask = self._parse_mask_response(
                            result["text"], canon_std, item_type,
                            service=svc_name,
                            model=result.get("model", model),
                            temperature=temperature,
                            tokens_prompt=result.get("tokens_prompt", 0),
                            tokens_completion=result.get("tokens_completion", 0)
                        )
                        if mask:
                            try:
                                re.compile(mask.pattern, re.IGNORECASE)
                            except re.error as re_err:
                                logger.warning("[LLMMaskGenerator] Generated mask fails to compile: %s — %s",
                                               mask.pattern[:80], re_err)
                                continue
                            meta = {
                                "provider": mask.service or svc_name,
                                "model": mask.model or model,
                                "temperature": mask.temperature or temperature,
                                "tokens_prompt": mask.tokens_prompt,
                                "tokens_completion": mask.tokens_completion,
                                "attempts": attempt,
                            }
                            logger.info("[LLMMaskGenerator] Generated mask via %s (attempt %d)", svc_name, attempt)
                            return mask, meta
                except Exception as e:
                    last_error = e
                    logger.debug("[LLMMaskGenerator] %s attempt %d failed: %s", svc_name, attempt, e)
        logger.error("[LLMMaskGenerator] Failed after %d attempts: %s", self.max_retries, last_error)
        return None, None

    def _resolve_service(self) -> Tuple[str, str, float]:
        service = ""
        model = ""
        temperature = 0.1
        if self.settings and hasattr(self.settings, "mask_generation"):
            mg = self.settings.mask_generation
            service = getattr(mg, "default_service", "")
            model = getattr(mg, "default_model", "")
            temperature = getattr(mg, "default_temperature", 0.1)
        if not service and self.settings and hasattr(self.settings, "default_service"):
            service = self.settings.default_service
        return service, model, temperature

    def _call_llm(self, client: Any, prompt: str, model: str, temperature: float) -> Optional[Dict]:
        client_type = type(client).__name__
        logger.debug("[LLMMaskGenerator] Calling %s with model=%s temp=%s", client_type, model, temperature)
        try:
            if hasattr(client, "chat") or hasattr(client, "generate"):
                method = getattr(client, "chat", None) or getattr(client, "generate", None)
                messages = [{"role": "user", "content": prompt}]
                try:
                    response = method(messages=messages, model=model, temperature=temperature)
                except TypeError as te:
                    logger.debug("[LLMMaskGenerator] messages failed, trying prompt: %s", te)
                    response = method(prompt=prompt, model=model, temperature=temperature)
                text = None
                if isinstance(response, str):
                    text = response
                elif isinstance(response, dict):
                    text = response.get("text", "") or response.get("raw", "") or response.get("content", "")
                    if not text:
                        choices = response.get("choices", [])
                        if choices and isinstance(choices, list):
                            choice = choices[0]
                            if isinstance(choice, dict):
                                msg = choice.get("message", {})
                                text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                            else:
                                text = str(choice)
                    if not text:
                        text = response.get("text", "") or response.get("content", "")
                elif hasattr(response, "text"):
                    text = response.text
                elif hasattr(response, "content"):
                    text = response.content
                else:
                    text = str(response)
                if text and len(text) > 10:
                    tokens_prompt = getattr(client, "last_tokens_prompt", 0) or getattr(client, "_last_prompt_tokens", 0)
                    tokens_completion = getattr(client, "last_tokens_completion", 0) or getattr(client, "_last_completion_tokens", 0)
                    return {
                        "text": text,
                        "model": model,
                        "tokens_prompt": tokens_prompt,
                        "tokens_completion": tokens_completion,
                    }
        except Exception as e:
            logger.warning("[LLMMaskGenerator] %s call failed: %s", client_type, e)
        try:
            if hasattr(client, "complete"):
                response = client.complete(prompt, model=model, temperature=temperature)
                if response and len(str(response)) > 10:
                    return {"text": str(response), "model": model, "tokens_prompt": 0, "tokens_completion": 0}
        except Exception as e:
            logger.debug("[LLMMaskGenerator] complete() failed: %s", e)
        logger.error("[LLMMaskGenerator] All LLM call methods failed for %s", client_type)
        return None

    @staticmethod
    def _extract_json_fields(text: str) -> Optional[Dict]:
        def _find_quoted_value(text: str, key: str) -> Optional[str]:
            pos = text.find(key)
            if pos < 0:
                return None
            quote = text.find('"', pos + len(key))
            if quote < 0:
                return None
            i = quote + 1
            while i < len(text):
                if text[i] == '\\' and i + 1 < len(text):
                    i += 2
                elif text[i] == '"':
                    break
                else:
                    i += 1
            if i >= len(text):
                return None
            return text[quote + 1:i]

        def _find_array(text: str, key: str) -> List[str]:
            pos = text.find(key)
            if pos < 0:
                return []
            bracket = text.find('[', pos + len(key))
            if bracket < 0:
                return []
            depth = 1
            j = bracket + 1
            while j < len(text) and depth > 0:
                if text[j] == '[':
                    depth += 1
                elif text[j] == ']':
                    depth -= 1
                j += 1
            if depth != 0:
                return []
            try:
                return json.loads(text[bracket:j])
            except Exception:
                return []

        raw_pattern = _find_quoted_value(text, '"pattern"')
        if not raw_pattern:
            return None
        pattern = raw_pattern.replace(r'\\\\', '\\')
        pattern = pattern.replace('\\"', '"')
        pattern = pattern.replace('\\n', '\n')
        pattern = pattern.replace('\\t', ' ')
        params = _find_array(text, '"params"')
        required = _find_array(text, '"required"')
        return {"pattern": pattern, "params": params, "required": required}


    def _parse_mask_response(
        self,
        text: str,
        standard: str,
        item_type: str,
        service: str = "",
        model: str = "",
        temperature: float = 0.0,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
    ) -> Optional[MaskGenerationResult]:
        if not text:
            return None
        data = None
        candidate = None
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                if "content" in parsed and isinstance(parsed["content"], dict):
                    data = parsed["content"]
                elif "raw" in parsed and isinstance(parsed["raw"], str):
                    text = parsed["raw"]
                else:
                    data = parsed
        except (ValueError, SyntaxError, TypeError):
            pass
        if data is None:
            try:
                import yaml
                data = yaml.safe_load(text)
                if isinstance(data, dict):
                    if "content" in data and isinstance(data["content"], dict):
                        data = data["content"]
                    elif "raw" in data and isinstance(data["raw"], str):
                        raw_text = data["raw"]
                        for prefix in ["```json", "```python", "```"]:
                            if raw_text.startswith(prefix):
                                raw_text = raw_text[len(prefix):].strip()
                                break
                        if raw_text.endswith("```"):
                            raw_text = raw_text[:-3].strip()
                        try:
                            data = json.loads(raw_text)
                        except json.JSONDecodeError:
                            try:
                                data = yaml.safe_load(raw_text)
                            except Exception:
                                data = None
            except Exception:
                pass
        if data is None:
            for prefix in ["```json", "```python", "```"]:
                start = text.find(prefix)
                if start >= 0:
                    start += len(prefix)
                    end = text.find("```", start)
                    if end >= 0:
                        candidate = text[start:end].strip()
                        break
            if candidate:
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        import yaml
                        data = yaml.safe_load(candidate)
                    except Exception:
                        pass
        if data is None:
            for start_match in re.finditer(r"(?m)^[ \t]*\{", text):
                pos = start_match.start()
                brace_count = 0
                in_string = False
                escape = False
                for i, ch in enumerate(text[pos:], start=pos):
                    if escape:
                        escape = False
                        continue
                    if ch == '\\' and not escape:
                        escape = True
                        continue
                    if ch == '"' and not escape:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if ch == '{':
                            brace_count += 1
                        elif ch == '}':
                            brace_count -= 1
                        if brace_count == 0:
                            candidate = text[pos:i + 1]
                            try:
                                data = json.loads(candidate)
                            except json.JSONDecodeError:
                                try:
                                    import yaml
                                    data = yaml.safe_load(candidate)
                                except Exception:
                                    pass
                            break
                if data is not None:
                    break
        if data is None:
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if json_match:
                candidate = json_match.group()
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        import yaml
                        data = yaml.safe_load(candidate)
                    except Exception:
                        pass
        if data is None:
            try:
                data = self._extract_json_fields(text)
            except Exception:
                data = None
        if data is None or not isinstance(data, dict):
            return None
        pattern = data.get("pattern", "")
        params = data.get("params", [])
        required = data.get("required", [])
        if not pattern or not pattern.startswith("^") or not pattern.endswith("$"):
            return None
        pattern = self._fix_pattern(pattern, standard, item_type)
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error:
            return None
        result = MaskGenerationResult(
            pattern=pattern, params=params, required=required,
            standard=standard, item_type=item_type, raw_response=text,
            service=service, model=model, temperature=temperature,
            tokens_prompt=tokens_prompt, tokens_completion=tokens_completion,
        )
        return self._sanitize_mask_result(result)

    def _fix_pattern(self, pattern: str, standard: str, item_type: str) -> str:
        if "ОСТ" in standard and r"(?P<нтд_1>\d+" in pattern:
            pattern = re.sub(r"\(?P<нтд_1>\d+[^\)]*\)", f"(?P<нтд_1>{re.escape(standard)})", pattern)
        if "ГОСТ" in standard and r"(?P<нтд_1>\d+" in pattern:
            pattern = re.sub(r"\(?P<нтд_1>\d+[^\)]*\)", f"(?P<нтд_1>{re.escape(standard)})", pattern)
        if r"\\|" in pattern:
            pattern = pattern.replace(r"\\|", "|")
        if "наименование_типа" in pattern and "тип_изделия" not in pattern:
            pattern = pattern.replace("наименование_типа", "тип_изделия")
        nested_fix = re.sub(
            r'\(?P<([^>]+)>\((\(?P<[^>]+>[^)]+\)\))',
            lambda m: f'(?P<{m.group(1)}>(?:{re.sub(r"\(?P<[^>]+>", "(?:", m.group(2))}))',
            pattern
        )
        if nested_fix != pattern:
            pattern = nested_fix
        max_iter = 5
        for _ in range(max_iter):
            new_pattern = re.sub(
                r'\(?P<([^>]+)>\(([^()]*\(?P<[^)]+\)[^()]*)\)',
                lambda m: f'(?P<{m.group(1)}>(?:{re.sub(r"\(?P<[^>]+>", "(?:", m.group(2))}))',
                pattern
            )
            if new_pattern == pattern:
                break
            pattern = new_pattern
        return pattern

    def _sanitize_mask_result(self, result: MaskGenerationResult) -> MaskGenerationResult:
        pattern = result.pattern
        params = list(result.params)
        required = list(result.required)
        for sp in self.SKIP_PARAMS:
            if sp in params:
                params.remove(sp)
            if sp in required:
                required.remove(sp)
            pattern = re.sub(rf'\(?P<{re.escape(sp)}>[^)]+\)\??', '', pattern)
        if "тип_изделия" in params and "наименование_типа" in params:
            params.remove("наименование_типа")
            if "наименование_типа" in required:
                required.remove("наименование_типа")
            pattern = re.sub(r"\(?P<наименование_типа>[^)]+\)(?:\s*[-\s]*)?", "", pattern)
        typo_fixes = {
            "тип_2изделия": "тип_изделия",
            "тип_изделия2": "тип_изделия",
            "тип_изделеия": "тип_изделия",
        }
        for bad, good in typo_fixes.items():
            if bad in params:
                params = [good if p == bad else p for p in params]
                required = [good if p == bad else p for p in required]
                pattern = pattern.replace(f"(?P<{bad}>", f"(?P<{good}>")
        required = [p for p in required if p in params]
        try:
            pattern = re.sub(r"\(?P<[^>]+>\)\?", "", pattern)
        except re.error:
            pass
        result.pattern = pattern
        result.params = params
        result.required = required
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as re_err:
            logger.warning("[LLMMaskGenerator] Sanitized pattern still invalid: %s — %s", pattern[:80], re_err)
        return result