# =============================================================================
# FILE: core/llm_mask_generator.py
# REPO: https://github.com/ayamashkin/NSI
# =============================================================================
# FIX 2026-05-27 16:04 UTC+3:
# 1. Убран захардкоженный CHECK_KEYS. Теперь видимые параметры определяются
#    автоматически из ВСЕХ полей примера ЕНС (кроме служебных).
# 2. Добавлен SERVICE_FIELDS — поля, которые никогда не участвуют в наименовании.
# 3. Добавлен FIELD_NAME_MAP — маппинг заголовков ENS → canonical имена.
# 4. _auto_detect_visible: для каждого примера проверяет все поля, значения
#    которых присутствуют в наименовании.
# 5. _detect_twin_groups: автоматическое определение близнецов (параметры,
#    значения которых всегда совпадают в примерах, где оба видны).
# 6. _resolve_twins: замена близнецов на canonical name (первый по порядку).
# 7. _filter_unambiguous: разделение на однозначные/неоднозначные примеры.
# 8. _get_global_visible: параметры, видные в >=85% однозначных примеров.
# 9. Обновлены _format_examples, _format_stats, _extract_visible_params.
# =============================================================================
"""
LLM Mask Generator Module
Generates regex masks using LLM with ENS examples context.
"""

import json
import logging
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
    """Результат генерации маски."""
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
    """Генератор масок через LLM с ENS-примерами."""

    # Поля ЕНС, которые никогда не участвуют в наименовании изделия
    SERVICE_FIELDS = {
        "Код", "Наименование", "Пометка удаления", "Базовая единица измерения",
        "Вести учет по характеристикам", "Вид специальной приемки",
        "Габаритные размеры, масса", "Гражданская продукция", "Заблокировано",
        "Марка материала", "Наименование типа", "НТД", "НТД на материал",
        "Обозначение, тип, артикул", "Полное наименование", "Свойства",
        "Соответствие ТР ТС", "Специальная приемка", "Технические характеристики",
        "Тип позиции", "Торговая марка", "ТР ТС", "Организация Корпорации",
        "Автор", "Автор последнего изменения", "Дата последнего изменения",
        "Дата создания", "Комментарий эксперта", "Наличие БП", "MDM Key",
        "Ссылка", "Классификатор ЕНС", "Классификатор ЕНС: Код",
        "ОКВЭД2", "ОКВЭД2: Код", "ОКПД2", "ОКПД2: Код",
        "Единицы измерения", "Категория качества", "Торговая марка",
    }

    # Маппинг заголовков ENS → canonical имена параметров
    FIELD_NAME_MAP = {
        "Наружный диаметр (диаметр вписанного круга), сторона квадрата (стороны поперечного сечения)": "наружный_диаметр",
        "Номинальный диаметр резьбы": "номинальный_диаметр_резьбы",
        "Номинальный диаметр": "номинальный_диаметр_резьбы",
        "Длина резьбы": "длина_резьбы",
        "Шаг резьбы": "шаг_резьбы",
        "Шаг второй резьбы": "шаг_второй_резьбы",
        "Толщина (проката, стенки, полки)": "толщина_проката_стенки_полки",
        "Группа (класс) прочности": "группа_класс_прочности",
        "Класс (поле) допуска": "класс_поле_допуска",
        "Класс (поле) допуска ввинчиваемого конца": "класс_допуска_ввинчиваемого_конца",
        "Марка материала": "марка_материала",
        "Марка материала_1": "марка_материала_1",
        "Тип резьбы": "тип_резьбы",
        "Покрытие": "покрытие",
        "Толщина покрытия": "толщина_покрытия",
        "Исполнение": "исполнение",
        "Вариант исполнения": "исполнение",
        "Длина": "длина",
        "Длина общая (OAL)": "длина_общая",
        "Длина ввинчиваемого конца": "длина_ввинчиваемого_конца",
        "Внутренний диаметр (условный проход)": "внутренний_диаметр",
        "Обозначение толщины покрытия": "толщина_покрытия",
        "Цвет": "цвет",
        "Свойства": "свойства",
        "Твердость": "твердость",
        "Ширина": "ширина",
        "Высота": "высота",
        "Толщина проката": "толщина_проката",
        "Диаметр посадочного отверстия": "диаметр_посадочного_отверстия",
        "Диаметр наружный (режущей части)": "диаметр_наружный",
        "Диаметр цилиндра": "диаметр_цилиндра",
        "Диаметр проволоки": "диаметр_проволоки",
        "Расстояние между осями": "расстояние_между_осями",
        "Расстояние между гранями": "расстояние_между_гранями",
        "Число витков": "число_витков",
        "Число шагов резьбы на дюйм": "число_шагов_на_дюйм",
        "Тип круга (головки)": "тип_круга",
        "Тип болта": "тип_болта",
        "Тип шлица": "тип_шлица",
        "Тип зуба": "тип_зуба",
        "Климатическое исполнение": "климатическое_исполнение",
        "Предельные отклонения": "предельные_отклонения",
        "Состояние поверхности": "состояние_поверхности",
        "Состояние материала": "состояние_материала",
        "Состояние поставки металлопроката": "состояние_поставки",
        "Форма поставки": "форма_поставки",
        "Вид и сторона покрытия": "вид_покрытия",
        "Лакокрасочное и полимерное покрытие": "лакокрасочное_покрытие",
        "Давление": "давление",
        "Радиус": "радиус",
        "Уклон": "уклон",
        "Наличие фаски": "наличие_фаски",
        "Наличие отверстий": "наличие_отверстий",
        "Класс (степень) точности": "класс_точности",
        "Класс качества": "класс_качества",
        "Каталожный номер": "каталожный_номер",
        "Типоразмер": "типоразмер",
        "Особые условия": "особые_условия",
        "Конструкция": "конструкция",
        "Серия (товара, изделия, продукта)": "серия",
        "Марка (товара, изделия, продукта)": "марка",
        "Обозначение болта": "обозначение_болта",
        "Обозначение гайки": "обозначение_гайки",
        "Обозначение винта": "обозначение_винта",
        "Обозначение шайбы": "обозначение_шайбы",
        "Обозначение заклепки": "обозначение_заклепки",
        "Условное обозначение марки материала": "усл_обозначение_марки",
        "Условное обозначение номера цвета": "усл_обозначение_цвета",
        "Условное обозначение толщины": "усл_обозначение_толщины",
        "Условное обозначение длины": "усл_обозначение_длины",
        "Индекс диаметра": "индекс_диаметра",
        "Обозначение диаметра корпуса": "диаметр_корпуса",
        "Обозначение ширины фаски": "ширина_фаски",
        "Обозначение размера \"под ключ\"": "размер_под_ключ",
        "Обозначение длины резьбы": "длина_резьбы_обозначение",
        "Способ изготовления": "способ_изготовления",
        "Категория проката": "категория_проката",
        "Направление резьбы": "направление_резьбы",
        "Способ получения стали": "способ_получения_стали",
        "Назначение материала": "назначение_материала",
        "Шкала твердости": "шкала_твердости",
        "Длина хвостовика": "длина_хвостовика",
        "Комплектность": "комплектность",
        "ВидПриемки": "вид_приемки",
        "Температурный диапазон (выкипания, эксплуатации)": "температурный_диапазон",
        "Категория размещения": "категория_размещения",
        "Номинальный диаметр": "номинальный_диаметр",
    }

    SKIP_PARAMS = {
        "марка_материала", "марка_материала_1", "толщина_покрытия",
        "наличие_бп", "автор_последнего_изменения", "дата_последнего_изменения",
    }

    def __init__(
        self,
        clients: Dict[str, Any],
        settings: Any = None,
        max_retries: int = 3,
    ):
        self.clients = clients
        self.settings = settings
        self.max_retries = max_retries
        self.validator = None
        logger.info("[LLMMaskGenerator] Initialized with %d clients", len(clients))

    def _get_validator(self):
        """Lazy init validator for ENS examples."""
        if self.validator is None:
            try:
                from core.auto_validator import AutoValidator
                ens_path = None
                if self.settings and hasattr(self.settings, "database"):
                    ens_path = getattr(self.settings.database, "ens_index_path", None)
                if not ens_path:
                    ens_path = "cache/ens_hardware.pkl"
                self.validator = AutoValidator(
                    ens_index_path=ens_path,
                    activation_threshold=0.85
                )
                logger.info("[LLMMaskGenerator] Validator initialized with %s", ens_path)
            except Exception as e:
                logger.warning("[LLMMaskGenerator] Failed to init validator: %s", e)
        return self.validator

    def _get_ens_examples(self, standard: str, item_type: str, max_examples: int = 20) -> List[Dict]:
        """Получить примеры из ЕНС для подстановки в промпт."""
        validator = self._get_validator()
        if not validator:
            logger.warning("[LLMMaskGenerator] No validator, returning empty examples")
            return []
        try:
            examples = validator._get_ens_examples(standard, item_type)
            if examples:
                logger.info("[LLMMaskGenerator] Loaded %d ENS examples for %s/%s",
                            len(examples), standard, item_type)
                return examples[:max_examples]
        except Exception as e:
            logger.warning("[LLMMaskGenerator] Failed to load examples: %s", e)
        return []

    def _canonicalize_field_name(self, field: str) -> str:
        """Преобразовать имя поля ENS в короткое canonical имя для regex."""
        if field in self.FIELD_NAME_MAP:
            return self.FIELD_NAME_MAP[field]
        # Fallback: cleanup
        name = re.sub(r'\s*\([^)]*\)\s*', ' ', field).strip()
        name = name.lower().replace(" ", "_").replace(",", "_")
        name = re.sub(r'_+', '_', name).strip('_')
        # Сокращение длинных имен
        if len(name) > 30:
            # Удаляем повторяющиеся слова и артикли
            name = re.sub(r'\b(di|de|la|le|et|du)\b', '', name)
            name = re.sub(r'_+', '_', name).strip('_')
        return name[:30]

    def _auto_detect_visible(self, ex: Dict, standard: str = "") -> Dict[str, str]:
        """Автоопределение видимых параметров из ВСЕХ полей примера ЕНС.

        Для каждого поля (кроме служебных) проверяем: присутствует ли его
        значение в строке наименования. Возвращает {canonical_name: value}.
        """
        name = ex.get("наименование", ex.get("полное_наименование", ""))
        if not name:
            return {}

        visible: Dict[str, str] = {}
        for field, value in ex.items():
            if field in self.SERVICE_FIELDS:
                continue
            if value is None or str(value).strip() == "":
                continue
            val_str = str(value).strip()
            if self._is_value_in_name(val_str, name, param_key=field, standard=standard):
                canonical = self._canonicalize_field_name(field)
                # Не дублируем: если два поля маппятся на один canonical,
                # оставляем первый найденный
                if canonical not in visible:
                    visible[canonical] = val_str
        return visible

    def _detect_twin_groups(
        self,
        examples: List[Dict],
        threshold: float = 1.0,
    ) -> List[List[str]]:
        """Автоопределение групп близнецов.

        Близнецы — параметры, значения которых ВСЕГДА совпадают в примерах,
        где оба параметра видны в наименовании.

        Returns:
            Список групп: [["a", "b", "c"], ["d", "e"]]
            Первый элемент группы — canonical (приоритетный) параметр.
        """
        # Собираем visible для всех примеров
        all_visible = []
        for ex in examples:
            vis = self._auto_detect_visible(ex)
            if vis:
                all_visible.append(vis)

        if not all_visible:
            return []

        # Статистика по парам: (a, b) -> [total_cooccurrence, matches]
        pair_stats: Dict[Tuple[str, str], List[int]] = {}

        for vis in all_visible:
            keys = sorted(vis.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    pair = (a, b)
                    if pair not in pair_stats:
                        pair_stats[pair] = [0, 0]
                    pair_stats[pair][0] += 1
                    if vis[a] == vis[b]:
                        pair_stats[pair][1] += 1

        # Фильтруем по threshold
        twin_edges = []
        for (a, b), (total, matches) in pair_stats.items():
            if total > 0 and matches / total >= threshold:
                twin_edges.append((a, b))

        # Транзитивное замыкание: объединяем связанные пары в группы
        if not twin_edges:
            return []

        # Union-Find
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

        # Сортируем группы: canonical = первый по частоте видимости
        groups = []
        for members in groups_map.values():
            if len(members) >= 2:
                # Сортируем по частоте появления в all_visible
                freq = {m: sum(1 for vis in all_visible if m in vis) for m in members}
                members_sorted = sorted(members, key=lambda m: -freq[m])
                groups.append(members_sorted)

        logger.info("[LLMMaskGenerator] Detected twin groups: %s", groups)
        return groups

    def _resolve_twins(
        self,
        visible: Dict[str, str],
        twin_groups: List[List[str]],
    ) -> Dict[str, str]:
        """Заменить близнецов на canonical name.

        Логика:
        - Если в visible есть canonical — удаляем всех близнецов.
        - Если canonical отсутствует, но есть близнец — подставляем canonical
          со значением близнеца.
        - Если несколько близнецов — берём первого по порядку в группе.
        """
        resolved = dict(visible)
        for group in twin_groups:
            canonical = group[0]
            twins = group[1:]

            # Если canonical уже есть — удаляем близнецов
            if canonical in resolved:
                for t in twins:
                    if t in resolved:
                        del resolved[t]
                continue

            # Иначе ищем первого видимого близнеца
            twin_val = None
            for t in twins:
                if t in resolved:
                    twin_val = resolved[t]
                    del resolved[t]
                    break

            if twin_val is not None:
                resolved[canonical] = twin_val

        return resolved

    def _filter_unambiguous(
        self,
        examples: List[Dict],
        twin_groups: List[List[str]],
        standard: str = "",
    ) -> Tuple[List[Tuple[Dict, Dict[str, str]]], List[Tuple[Dict, Dict[str, str]]]]:
        """Разделить примеры на однозначные и неоднозначные.

        После разрешения близнецов проверяем: есть ли дублирующиеся values.
        Если да — пример неоднозначный (исключается из global_visible).
        """
        unambiguous = []
        ambiguous = []

        for ex in examples:
            vis = self._auto_detect_visible(ex, standard=standard)
            vis = self._resolve_twins(vis, twin_groups)

            values = list(vis.values())
            if len(values) != len(set(values)):
                ambiguous.append((ex, vis))
            else:
                unambiguous.append((ex, vis))

        logger.info(
            "[LLMMaskGenerator] Unambiguous: %d, Ambiguous: %d",
            len(unambiguous), len(ambiguous)
        )
        return unambiguous, ambiguous

    def _get_global_visible(
        self,
        unambiguous: List[Tuple[Dict, Dict[str, str]]],
        threshold: float = 0.85,
    ) -> Tuple[set, set]:
        """Определить глобально видимые параметры из однозначных примеров.

        Returns:
            (required_params, optional_params)
            required: видны в >= threshold примеров
            optional: видны в >= 0.20 и < threshold примеров
        """
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

        # Всегда добавляем тип и стандарт
        required.add("тип_изделия")
        required.add("нтд_1")

        logger.info(
            "[LLMMaskGenerator] Global visible: required=%s, optional=%s",
            sorted(required), sorted(optional)
        )
        return required, optional

    def _format_stats(
        self,
        examples: List[Dict],
        standard: str = "",
        twin_groups: Optional[List[List[str]]] = None,
    ) -> str:
        """Форматировать статистику глобально видимых параметров для вставки в промпт."""
        if not examples:
            return "(нет данных)"

        if twin_groups is None:
            twin_groups = self._detect_twin_groups(examples)

        unambiguous, _ = self._filter_unambiguous(examples, twin_groups, standard=standard)
        required, optional = self._get_global_visible(unambiguous)
        global_visible = required | optional

        # Подсчёт только по однозначным примерам
        param_counts: Dict[str, int] = {}
        for ex, vis in unambiguous:
            for key in vis:
                if key in global_visible:
                    param_counts[key] = param_counts.get(key, 0) + 1

        total = len(examples)
        lines = []
        for key, count in sorted(param_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {key}: {count} из {total} ({count/total*100:.0f}%)")
        return "\n".join(lines) if lines else "(нет параметров)"

    @staticmethod
    def _is_value_in_name(val: str, name: str, param_key: str = "", standard: str = "") -> bool:
        """Проверить, что значение параметра присутствует в строке номенклатуры."""
        if not val or not name:
            return False

        # Служебные параметры — skip
        skip_canonical = {
            "марка_материала", "марка_материала_1", "толщина_покрытия",
            "наличие_бп", "автор_последнего_изменения", "дата_последнего_изменения",
        }
        if param_key in skip_canonical:
            return False

        name_clean = name
        if standard:
            name_clean = re.sub(
                r'ОСТ\s*\d+\s*\d+-\d+|ГОСТ\s*\d+-\d+',
                '',
                name_clean,
                flags=re.IGNORECASE
            )

        val_raw = str(val).strip()
        val_str = val_raw.lower().replace(",", ".")
        name_lower = name_clean.lower().replace(",", ".")

        # Прямое вхождение
        if val_str in name_lower:
            return True

        # Число с десятичной точкой: 1,5 → 1.5 и 15
        if re.match(r"^\d+[.,]\d+$", val_raw):
            no_sep = re.sub(r"[.,]", "", val_str)
            if no_sep in name_lower:
                return True

        # Покрытие и текстовые значения: токенизация
        if re.search(r"[a-zA-Zа-яА-Я]", val_str):
            tokens = re.split(r"[.\-]", val_str)
            tokens = [t for t in tokens if t and re.search(r"[a-zA-Zа-яА-Я]", t)]
            for tok in tokens:
                if tok in name_lower:
                    return True
            prefix = re.match(r"^([a-zA-Zа-яА-Я]+)", val_str)
            if prefix and prefix.group(1) in name_lower:
                return True

        # .0 → целое
        if '.' in val_str and val_str.endswith('.0'):
            int_part = val_str[:-2]
            if int_part and int_part in name_lower:
                return True

        # Число+буква: 6г, 6Н
        if re.match(r"^\d+[a-zA-Zа-яА-Я]+$", val_str):
            if val_str in name_lower:
                return True

        # M12 → 12
        m_match = re.match(r"^[мm](\d+(?:[.,]\d+)?)$", val_raw, re.IGNORECASE)
        if m_match:
            num = m_match.group(1)
            if num.lower() in name_lower:
                return True

        return False

    @staticmethod
    def _find_value_positions(val: str, name: str, param_key: str = "") -> List[Tuple[int, int]]:
        """Найти все позиции значения в строке номенклатуры."""
        if not val or not name:
            return []
        val_str = val.strip().lower()
        name_lower = name.lower()
        positions = []
        start = 0
        while True:
            idx = name_lower.find(val_str, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(val_str)))
            start = idx + 1
        return positions

    def _select_representative_examples(self, examples: List[Dict], max_count: int = 10) -> List[Dict]:
        """Выбрать representative примеры, покрывающие максимум комбинаций параметров."""
        if len(examples) <= max_count:
            return examples

        # Определяем visible params для каждого примера (без twin resolution,
        # но с исключением неоднозначных ключей)
        def _visible_params(ex: Dict) -> set:
            vis = self._auto_detect_visible(ex)
            # Удаляем неоднозначные ключи (те, у которых value дублируется)
            val_to_keys: Dict[str, List[str]] = {}
            for k, v in vis.items():
                val_to_keys.setdefault(v, []).append(k)
            ambiguous_keys = set()
            for v, keys in val_to_keys.items():
                if len(keys) >= 2:
                    for k in keys:
                        ambiguous_keys.add(k)
            return {k for k in vis if k not in ambiguous_keys}

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

        logger.info("[LLMMaskGenerator] Selected %d representative examples (coverage: %s)",
                    len(selected), sorted(covered))
        return selected

    def _format_examples(
        self,
        examples: List[Dict],
        standard: str,
        item_type: str,
    ) -> str:
        """Форматировать ENS-примеры для вставки в промпт.

        FIX 2026-05-27: Автоопределение видимых параметров из всех полей.
        Близнецы разрешаются, неоднозначные примеры исключаются из global_visible.
        """
        if not examples:
            return "(примеры отсутствуют)"

        display_examples = self._select_representative_examples(examples, max_count=10)

        # Шаг 1: автоопределение близнецов
        twin_groups = self._detect_twin_groups(examples)

        # Шаг 2: разделить на однозначные и неоднозначные
        unambiguous, ambiguous = self._filter_unambiguous(examples, twin_groups, standard=standard)

        # Шаг 3: глобально видимые параметры
        required, optional = self._get_global_visible(unambiguous)
        global_visible = required | optional

        # Шаг 4: формирование примеров для промпта
        lines = []
        lines.append(f"Структура: <{item_type}> [исполнение] <параметры> <покрытие> {standard}")
        lines.append("")
        lines.append(f"Параметры, участвующие в наименовании: {', '.join(sorted(global_visible))}")
        lines.append("")

        for i, ex in enumerate(display_examples, 1):
            name = ex.get("наименование", ex.get("полное_наименование", ""))
            if not name:
                continue

            vis = self._auto_detect_visible(ex, standard=standard)
            vis = self._resolve_twins(vis, twin_groups)

            # Определяем неоднозначные ключи в этом примере
            val_to_keys: Dict[str, List[str]] = {}
            for k, v in vis.items():
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
                    # Тип изделия — всегда первое слово
                    val = ex.get("Наименование типа", ex.get("Наименование типа", ""))
                    if val and name.lower().startswith(str(val).strip().lower()):
                        visible_list.append((key, str(val).strip(), 0))
                    else:
                        missing_list.append(key)
                    continue

                if key not in vis:
                    missing_list.append(key)
                    continue

                val_str = vis[key]
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
                lines.append(f"   Видимые: {vis_str}")
            if ambiguous_list:
                amb_str = " ".join([f"(?P<{k}>{v})" for k, v in ambiguous_list])
                lines.append(f"   Неоднозначные: {amb_str}")
            if missing_list:
                lines.append(f"   Отсутствуют: {', '.join(missing_list)}")
            lines.append("")

        return "\n".join(lines)

    def _get_prompt_template(self) -> str:
        """Загрузить шаблон промпта."""
        if self.settings and hasattr(self.settings, "mask_generation"):
            mg = self.settings.mask_generation
            template_path = getattr(mg, "prompt_template", "")
            if template_path:
                p = Path(template_path)
                if p.exists():
                    content = p.read_text(encoding="utf-8")
                    logger.info("[LLMMaskGenerator] Loaded prompt template from %s", template_path)
                    return content
                else:
                    logger.warning("[LLMMaskGenerator] prompt_template path not found: %s", template_path)
            for path in [
                "prompts/templates/mask_generation.txt",
                "prompts/mask_generation.txt",
                "config/mask_generation.txt",
            ]:
                p = Path(path)
                if p.exists():
                    logger.info("[LLMMaskGenerator] Loaded prompt template from fallback %s", path)
                    return p.read_text(encoding="utf-8")
            logger.warning("[LLMMaskGenerator] No template file found, using default")
            return self._default_template()

    def _default_template(self) -> str:
        r"""Default template with v3 rules."""
        return r"""Ты — эксперт по техническим стандартам ГОСТ/ОСТ/ТУ и регулярным выражениям Python 3 (re модуль).

=== ЗАДАЧА ===

Создай regex-паттерн с named groups (?P<name>...) для извлечения параметров из строки номенклатуры.

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
15. **ЗАПРЕЩЕНЫ nested named groups**: `(?P<name>(?P<name2>...))` — НЕВАЛИДНЫ в Python re. Используй `(?:...)` для вложенных групп.
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
Используй (?P<name>...)? для опциональных параметров.

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
- [ ] НЕТ nested named groups `(?P<a>(?P<b>...))`?
- [ ] НЕТ unbalanced parentheses?
- [ ] Pattern компилируется в Python re без ошибок?

=== СТРОГОЕ СООТВЕТСТВИЕ ===

Выведи ОДИН JSON-объект. Без комментариев, markdown, объяснений — только JSON."""

    def _build_prompt(self, standard: str, item_type: str, examples: List[Dict],
                      name: str = "", standard_info: Any = None) -> str:
        """Собрать промпт с ENS-примерами."""
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
        """Извлечь список глобально видимых параметров из ENS-примеров."""
        if not examples:
            return ["тип_изделия", "номинальный_диаметр_резьбы", "покрытие", "нтд_1"]

        twin_groups = self._detect_twin_groups(examples)
        unambiguous, _ = self._filter_unambiguous(examples, twin_groups)
        required, optional = self._get_global_visible(unambiguous)
        global_visible = required | optional
        return list(global_visible)

    def _get_debug_dir(self) -> Optional[Path]:
        """Получить путь к debug-директории из конфига."""
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
        """Сохранить промпт в debug-директорию."""
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
        """Сохранить ответ LLM в debug-директорию."""
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
        """Генерация маски через LLM с ENS-примерами."""
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
                        self._save_debug_response(canon_std, item_type, result["text"],
                                                  svc_name, attempt)

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
                                logger.warning(
                                    "[LLMMaskGenerator] Generated mask fails to compile: %s — %s",
                                    mask.pattern[:80], re_err
                                )
                                continue

                            meta = {
                                "provider": mask.service or svc_name,
                                "model": mask.model or model,
                                "temperature": mask.temperature or temperature,
                                "tokens_prompt": mask.tokens_prompt,
                                "tokens_completion": mask.tokens_completion,
                                "attempts": attempt,
                            }
                            logger.info(
                                "[LLMMaskGenerator] Generated mask via %s (attempt %d)",
                                svc_name, attempt
                            )
                            return mask, meta
                except Exception as e:
                    last_error = e
                    logger.debug("[LLMMaskGenerator] %s attempt %d failed: %s",
                                 svc_name, attempt, e)
        logger.error("[LLMMaskGenerator] Failed after %d attempts: %s",
                     self.max_retries, last_error)
        return None, None

    def _resolve_service(self) -> Tuple[str, str, float]:
        """Определить сервис, модель и температуру."""
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
        """Вызвать LLM клиент с fallback на разные интерфейсы."""
        client_type = type(client).__name__
        logger.debug("[LLMMaskGenerator] Calling %s with model=%s temp=%s", client_type, model, temperature)

        try:
            if hasattr(client, "chat") or hasattr(client, "generate"):
                method = getattr(client, "chat", None) or getattr(client, "generate", None)
                messages = [{"role": "user", "content": prompt}]

                try:
                    response = method(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                    )
                except TypeError as te:
                    logger.debug("[LLMMaskGenerator] messages failed, trying prompt: %s", te)
                    response = method(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                    )

                text = None
                if isinstance(response, str):
                    text = response
                elif isinstance(response, dict):
                    text = response.get("text", "") or response.get("raw", "") or response.get("content", "")
                    if not text:
                        text = str(response)
                    choices = response.get("choices", [])
                    if choices and isinstance(choices, list):
                        choice = choices[0]
                        if isinstance(choice, dict):
                            msg = choice.get("message", {})
                            text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                        else:
                            text = str(choice)
                    else:
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
                    logger.info("[LLMMaskGenerator] %s success, response length=%d", client_type, len(text))
                    logger.debug("[LLM] Raw content from %s (len=%d): %r", client_type, len(text), text[:500])
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
                    return {
                        "text": str(response),
                        "model": model,
                        "tokens_prompt": 0,
                        "tokens_completion": 0,
                    }
        except Exception as e:
            logger.debug("[LLMMaskGenerator] complete() failed: %s", e)

        logger.error("[LLMMaskGenerator] All LLM call methods failed for %s", client_type)
        return None

    @staticmethod
    def _extract_json_fields(text: str) -> Optional[Dict]:
        """Извлечь pattern/params/required из JSON-like текста."""
        logger.debug("[LLM] _extract_json_fields called, text len=%d", len(text))

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

        pattern = raw_pattern.replace(r'\\', '\\')
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
        """Парсинг JSON-ответа LLM."""
        if not text:
            logger.warning("[LLMMaskGenerator] Empty response text")
            return None

        logger.debug("[LLM] Raw response for %s/%s (len=%d): %r",
                     standard, item_type, len(text), text[:800])

        data = None
        candidate = None

        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                if "content" in parsed and isinstance(parsed["content"], dict):
                    data = parsed["content"]
                    logger.debug("[LLM] Parsed via ast.literal_eval -> content")
                elif "raw" in parsed and isinstance(parsed["raw"], str):
                    text = parsed["raw"]
                    logger.debug("[LLM] Parsed via ast.literal_eval -> raw, re-parsing")
                else:
                    data = parsed
                    logger.debug("[LLM] Parsed via ast.literal_eval -> direct dict")
        except (ValueError, SyntaxError, TypeError) as e:
            logger.debug("[LLM] ast.literal_eval failed: %s", e)

        if data is None:
            try:
                import yaml
                data = yaml.safe_load(text)
                if isinstance(data, dict):
                    if "content" in data and isinstance(data["content"], dict):
                        data = data["content"]
                        logger.debug("[LLM] Parsed via yaml.safe_load -> content")
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
                            logger.debug("[LLM] Parsed raw markdown JSON via json.loads")
                        except json.JSONDecodeError:
                            try:
                                data = yaml.safe_load(raw_text)
                                if isinstance(data, dict):
                                    logger.debug("[LLM] Parsed raw markdown JSON via yaml")
                                else:
                                    data = None
                            except Exception:
                                data = None
                    else:
                        logger.debug("[LLM] Parsed via yaml.safe_load -> direct dict")
                else:
                    data = None
            except Exception as e:
                logger.debug("[LLM] yaml.safe_load failed: %s", e)

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
                    logger.debug("[LLM] Parsed markdown JSON via json.loads")
                except json.JSONDecodeError:
                    try:
                        import yaml
                        data = yaml.safe_load(candidate)
                        if isinstance(data, dict):
                            logger.debug("[LLM] Parsed markdown JSON via yaml.safe_load")
                        else:
                            data = None
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
                                logger.debug("[LLM] Parsed balanced JSON")
                            except json.JSONDecodeError:
                                try:
                                    import yaml
                                    data = yaml.safe_load(candidate)
                                    if isinstance(data, dict):
                                        logger.debug("[LLM] Parsed balanced JSON via yaml")
                                    else:
                                        data = None
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
                    logger.debug("[LLM] Parsed simple JSON")
                except json.JSONDecodeError:
                    try:
                        import yaml
                        data = yaml.safe_load(candidate)
                        if isinstance(data, dict):
                            logger.debug("[LLM] Parsed simple JSON via yaml")
                        else:
                            data = None
                    except Exception:
                        pass

        if data is None:
            logger.debug("[LLM] Trying _extract_json_fields fallback...")
            try:
                data = self._extract_json_fields(text)
                if data:
                    logger.info("[LLM] Parsed via regex field extraction for %s/%s", standard, item_type)
                else:
                    logger.debug("[LLM] _extract_json_fields returned None")
            except Exception as e:
                logger.warning("[LLM] _extract_json_fields failed: %s", e)
                data = None

        if data is None or not isinstance(data, dict):
            logger.warning(
                "[LLMMaskGenerator] Failed to parse any JSON for %s/%s. Preview: %r",
                standard, item_type, text[:300]
            )
            return None

        pattern = data.get("pattern", "")
        params = data.get("params", [])
        required = data.get("required", [])
        if not pattern or not pattern.startswith("^") or not pattern.endswith("$"):
            logger.warning("[LLMMaskGenerator] Invalid pattern: %s", pattern[:50])
            return None
        pattern = self._fix_pattern(pattern, standard, item_type)

        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as re_err:
            logger.warning(
                "[LLMMaskGenerator] Pattern fails to compile after _fix_pattern: %s — %s",
                pattern[:80], re_err
            )
            return None

        result = MaskGenerationResult(
            pattern=pattern,
            params=params,
            required=required,
            standard=standard,
            item_type=item_type,
            raw_response=text,
            service=service,
            model=model,
            temperature=temperature,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
        )
        return self._sanitize_mask_result(result)

    def _fix_pattern(self, pattern: str, standard: str, item_type: str) -> str:
        """Исправить типичные ошибки LLM в regex."""
        if "ОСТ" in standard and r"(?P<нтд_1>\d+" in pattern:
            pattern = re.sub(
                r"\(?P<нтд_1>\d+[^\)]*\)",
                f"(?P<нтд_1>{re.escape(standard)})",
                pattern
            )
            logger.info("[LLMMaskGenerator] Fixed нтд_1 for ОСТ standard")
        if "ГОСТ" in standard and r"(?P<нтд_1>\d+" in pattern:
            pattern = re.sub(
                r"\(?P<нтд_1>\d+[^\)]*\)",
                f"(?P<нтд_1>{re.escape(standard)})",
                pattern
            )
            logger.info("[LLMMaskGenerator] Fixed нтд_1 for ГОСТ standard")
        if r"\\|" in pattern:
            pattern = pattern.replace(r"\\|", "|")
            logger.info("[LLMMaskGenerator] Fixed escaped alternation")
        if "наименование_типа" in pattern and "тип_изделия" not in pattern:
            pattern = pattern.replace("наименование_типа", "тип_изделия")
            logger.info("[LLMMaskGenerator] Fixed тип_изделия name")

        nested_fix = re.sub(
            r'\(?P<([^>]+)>\((\(?P<[^>]+>[^)]+\)\)',
            lambda m: f'(?P<{m.group(1)}>(?:{re.sub(r"\(?P<[^>]+>", "(?:", m.group(2))}))',
            pattern
        )
        if nested_fix != pattern:
            logger.info("[LLMMaskGenerator] Removed nested named groups")
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
            logger.info("[LLMMaskGenerator] Iterative nested named group fix applied")

        return pattern

    def _sanitize_mask_result(self, result: MaskGenerationResult) -> MaskGenerationResult:
        r"""Универсальная очистка маски от типичных LLM-ошибок."""
        pattern = result.pattern
        params = list(result.params)
        required = list(result.required)

        for sp in LLMMaskGenerator.SKIP_PARAMS:
            if sp in params:
                params.remove(sp)
                logger.info("[LLMMaskGenerator] Sanitized: removed metadata param %s", sp)
            if sp in required:
                required.remove(sp)
            pattern = re.sub(rf'\(?P<{re.escape(sp)}>[^)]+\)\??', '', pattern)

        if "тип_изделия" in params and "наименование_типа" in params:
            params.remove("наименование_типа")
            if "наименование_типа" in required:
                required.remove("наименование_типа")
            pattern = re.sub(r"\(?P<наименование_типа>[^)]+\)(?:\s*[-\s]*)?", "", pattern)
            logger.info("[LLMMaskGenerator] Sanitized: removed duplicate наименование_типа")

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
                logger.info("[LLMMaskGenerator] Sanitized: fixed typo %s → %s", bad, good)

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
            logger.warning(
                "[LLMMaskGenerator] Sanitized pattern still invalid: %s — %s",
                pattern[:80], re_err
            )

        return result