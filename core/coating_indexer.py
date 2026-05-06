"""
Coating Indexer Module
Гибридная индексация правил покрытий:
1. Сканирует ENS-индекс, строит material→[coatings] map из фактических данных
2. Определяет "белые пятна" (стандарты с мало данных) → флаг для LLM
3. Сохраняет/обновляет coating_rules в config.yaml

LAST_FIX: 2026-05-06 14:30 — hybrid ENS+LLM coating rules builder
"""

import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Поля, которые означают "без покрытия"
NO_COATING_VALUES = {'', None, 'без покрытия', 'бп', 'нет', '-', 'none', 'н/п'}

# Минимум примеров для считать данные достаточными (без LLM)
MIN_EXAMPLES_PER_MATERIAL = 2
MIN_UNIQUE_MATERIALS = 1


class CoatingIndexer:
    """
    Индексатор правил покрытий из ENS + LLM fallback.
    """

    def __init__(self, ens_index_path: str = "cache/ens_index.yaml"):
        self.ens_index_path = Path(ens_index_path)
        self._ens_data: Optional[Dict] = None

    def load_ens(self) -> Dict:
        """Загрузка ENS индекса."""
        if self._ens_data is not None:
            return self._ens_data
        if not self.ens_index_path.exists():
            logger.warning(f"[CoatingIndexer] ENS index not found: {self.ens_index_path}")
            return {}
        try:
            with open(self.ens_index_path, 'r', encoding='utf-8') as f:
                self._ens_data = yaml.safe_load(f) or {}
            items = self._ens_data.get('items', [])
            logger.info(f"[CoatingIndexer] Loaded {len(items)} ENS items")
            return self._ens_data
        except Exception as e:
            logger.error(f"[CoatingIndexer] Failed to load ENS: {e}")
            return {}

    def build_coating_map(self, standard: str, item_type: str) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """
        Построение material→[coatings] map для конкретного (standard, item_type).

        Returns:
            (material_coating_map, metadata)
            material_coating_map: {"14Х17Н2": ["Н.Кд", "Хим.Пас"], ...}
            metadata: {"total_items": 10, "unique_materials": 1, "llm_needed": False, ...}
        """
        ens_data = self.load_ens()
        items = ens_data.get('items', [])

        # Фильтруем по стандарту и типу
        standard_lower = standard.lower().strip()
        item_type_lower = item_type.lower().strip()

        filtered = []
        for item in items:
            item_std = str(item.get('нтд', '')).lower().strip()
            item_type_field = str(
                item.get('тип_изделия', '') or item.get('наименование_типа', '')
            ).lower().strip()

            # Нормализация стандарта: убираем пробелы между ОСТ и цифрами
            item_std_norm = re.sub(r'\s+', '', item_std)
            std_norm = re.sub(r'\s+', '', standard_lower)

            if item_std_norm == std_norm and item_type_field == item_type_lower:
                filtered.append(item)

        logger.info(f"[CoatingIndexer] Found {len(filtered)} items for {standard}/{item_type}")

        if not filtered:
            return {}, {"total_items": 0, "llm_needed": True, "reason": "no_items"}

        # Собираем (material → coatings) из фактических данных
        material_coatings: Dict[str, Set[str]] = defaultdict(set)
        material_examples: Dict[str, int] = defaultdict(int)

        for item in filtered:
            material = self._extract_material(item)
            coating = self._extract_coating(item)

            if not material:
                continue

            material_examples[material] += 1

            if coating is not None:
                material_coatings[material].add(coating)
            else:
                material_coatings[material].add("Бп")  # "Без покрытия" как явное значение

        # Преобразуем set → list
        result_map = {mat: sorted(list(cats)) for mat, cats in material_coatings.items()}

        # Определяем, нужен ли LLM
        unique_materials = len(result_map)
        total_examples = sum(material_examples.values())

        # LLM нужен если: мало материалов ИЛИ у какого-то материала мало примеров
        llm_needed = False
        llm_reason = []

        if unique_materials < MIN_UNIQUE_MATERIALS:
            llm_needed = True
            llm_reason.append(f"only {unique_materials} materials (need {MIN_UNIQUE_MATERIALS})")

        for mat, count in material_examples.items():
            if count < MIN_EXAMPLES_PER_MATERIAL:
                llm_needed = True
                llm_reason.append(f"{mat}: only {count} examples (need {MIN_EXAMPLES_PER_MATERIAL})")

        metadata = {
            "total_items": total_examples,
            "unique_materials": unique_materials,
            "llm_needed": llm_needed,
            "llm_reason": "; ".join(llm_reason) if llm_needed else "",
            "material_examples": dict(material_examples),
        }

        logger.info(f"[CoatingIndexer] Built map for {standard}/{item_type}: {result_map}, meta={metadata}")
        return result_map, metadata

    @staticmethod
    def _extract_material(item: Dict) -> Optional[str]:
        """Извлечение марки материала из ENS записи."""
        for field in ['марка_материала', 'марка_стали', 'материал']:
            val = item.get(field)
            if val and str(val).strip():
                return str(val).strip()
        return None

    @staticmethod
    def _extract_coating(item: Dict) -> Optional[str]:
        """
        Извлечение покрытия из ENS записи.
        Returns: строка покрытия или None для "без покрытия"
        """
        val = item.get('покрытие')
        if val is None:
            return None
        val_str = str(val).strip()
        if not val_str:
            return None
        if val_str.lower() in NO_COATING_VALUES:
            return None
        return val_str

    def get_llm_coating_prompt(self, standard: str, item_type: str, material_map: Dict[str, List[str]]) -> str:
        """
        Формирование промпта для LLM: какие покрытия допустимы.
        """
        materials_list = ", ".join([f'"{m}"' for m in material_map.keys()]) if material_map else "(неизвестны)"

        prompt = f"""Ты — эксперт по техническим стандартам и материаловедению.

СТАНДАРТ: {standard}
ТИП ИЗДЕЛИЯ: {item_type}
МАРКИ СТАЛИ В ИНДЕКСЕ: {materials_list}

ЗАДАЧА: Определи допустимые покрытия для каждой марки стали по этому стандарту.

ПРАВИЛА:
1. Для коррозионно-стойких сталей (14Х17Н2, 12Х18Н10Т, 08Х18Н10Т и т.д.):
   - НЕ допускается простое кадмиевое покрытие ("Кд")
   - Допускаются: никелевые подслои ("Н.Кд"), химические ("Хим.Пас"), пассивация
2. Для конструкционных сталей (30ХГСА, 40Х, 35ХГСА и т.д.):
   - Допускаются: "Кд", "Цд", "Окс", "Фос", "Окс.Фос", "Неп", "Бп"
3. "Бп" означает "без покрытия"
4. Если марка неизвестна — определи по стандарту

ФОРМАТ ОТВЕТА (строго JSON, без комментариев):
```json
{{
  "material_coating_map": {{
    "МАРКА1": ["Покрытие1", "Покрытие2"],
    "МАРКА2": ["Покрытие1", "Бп"]
  }},
  "auto_substitution": [
    {{
      "material_pattern": "regex паттерн для марок",
      "wrong_coating": "какое покрытие заменить",
      "correct_coating": "на какое заменить"
    }}
  ],
  "notes": "пояснения"
}}
```

Ответь ТОЛЬКО JSON, без пояснений."""
        return prompt

    def merge_with_existing(
        self,
        new_map: Dict[str, List[str]],
        existing_rules: Dict[str, Any],
        standard: str,
        item_type: str
    ) -> Dict[str, Any]:
        """
        Слияние новых данных с существующими правилами.
        Приоритет: фактические данные из ENS > LLM > существующие правила.
        """
        merged = dict(existing_rules) if existing_rules else {}

        material_map = merged.get('material_coating_map', {})

        # Обновляем/дополняем
        for material, coatings in new_map.items():
            if material in material_map:
                # Merge: добавляем недостающие покрытия
                existing = set(material_map[material])
                for c in coatings:
                    existing.add(c)
                material_map[material] = sorted(list(existing))
                logger.debug(f"[CoatingIndexer] Merged {material}: {material_map[material]}")
            else:
                material_map[material] = list(coatings)
                logger.debug(f"[CoatingIndexer] Added {material}: {coatings}")

        merged['material_coating_map'] = material_map
        return merged

    def save_to_config(
        self,
        coating_rules: Dict[str, Any],
        config_path: str = "config/config.yaml"
    ) -> bool:
        """Сохранение coating_rules в config.yaml."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"[CoatingIndexer] Config not found: {path}")
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            config['coating_rules'] = coating_rules

            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

            logger.info(f"[CoatingIndexer] Saved coating_rules to {path}")
            return True
        except Exception as e:
            logger.error(f"[CoatingIndexer] Failed to save: {e}")
            return False


def build_coating_rules_for_standard(
    standard: str,
    item_type: str,
    ens_index_path: str = "cache/ens_index.yaml",
    config_path: str = "config/config.yaml",
    use_llm: bool = True,
    llm_generator = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Главная entry point: построение coating_rules для (standard, item_type).

    Returns:
        (coating_rules, llm_was_used)
    """
    indexer = CoatingIndexer(ens_index_path)

    # Phase 1: Индексация ENS
    material_map, meta = indexer.build_coating_map(standard, item_type)
    logger.info(f"[CoatingIndexer] Phase 1 done: {meta}")

    llm_used = False

    # Phase 2: LLM augmentation (если нужно)
    if meta.get('llm_needed') and use_llm and llm_generator:
        logger.info(f"[CoatingIndexer] Phase 2: LLM query for {standard}/{item_type}")
        prompt = indexer.get_llm_coating_prompt(standard, item_type, material_map)

        try:
            # Запрос к LLM через существующий генератор
            response = llm_generator.query_coatings(prompt, standard, item_type)
            if response and 'material_coating_map' in response:
                llm_map = response['material_coating_map']
                # Merge: ENS данные имеют приоритет над LLM
                for mat, coatings in llm_map.items():
                    if mat not in material_map:
                        material_map[mat] = coatings
                        logger.info(f"[CoatingIndexer] LLM added {mat}: {coatings}")
                    else:
                        # Дополняем, но не перезаписываем
                        existing = set(material_map[mat])
                        for c in coatings:
                            existing.add(c)
                        material_map[mat] = sorted(list(existing))

                llm_used = True
                logger.info(f"[CoatingIndexer] LLM augmented: {material_map}")
        except Exception as e:
            logger.error(f"[CoatingIndexer] LLM query failed: {e}")

    # Phase 3: Слияние с существующими правилами
    existing_rules = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        existing_rules = config.get('coating_rules', {})
    except Exception:
        pass

    final_rules = indexer.merge_with_existing(material_map, existing_rules, standard, item_type)

    # Сохраняем
    indexer.save_to_config(final_rules, config_path)

    return final_rules, llm_used