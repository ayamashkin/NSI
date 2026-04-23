"""
ENS Reference Loader Module
Адаптивная загрузка справочника ЕСН с внешним конфигом маппинга.
"""

import re
import yaml
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ENSCategory(Enum):
    """Категории номенклатуры в ЕСН."""
    HARDWARE = "hardware"
    WASHER = "hardware_washer"
    ROLLED_METAL = "rolledmetal"
    ERI = "eri"
    EKB = "ekb"
    MATERIALS = "materials"
    UNKNOWN = "unknown"


@dataclass
class ENSColumnMapping:
    """Конфигурация маппинга колонок."""
    base_mapping: Dict[str, str] = field(default_factory=dict)
    category_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)
    auto_patterns: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'ENSColumnMapping':
        """Загрузка маппинга из YAML."""
        if path is None:
            path = "config/ens_column_mapping.yaml"

        path = Path(path)
        if not path.exists():
            # Пробуем альтернативные пути
            alt_paths = [
                Path("ens_column_mapping.yaml"),
                Path("../config/ens_column_mapping.yaml"),
            ]
            for alt in alt_paths:
                if alt.exists():
                    path = alt
                    break

        if not path.exists():
            logger.warning(f"Mapping config not found at {path}, using defaults")
            return cls._default_mapping()

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Компилируем паттерны
            patterns = {}
            for pattern, mapped in data.get('auto_mapping_patterns', {}).items():
                try:
                    patterns[re.compile(pattern, re.IGNORECASE)] = mapped
                except re.error as e:
                    logger.warning(f"Invalid pattern '{pattern}': {e}")

            return cls(
                base_mapping=data.get('base_mapping', {}),
                category_mapping=data.get('category_mapping', {}),
                auto_patterns=patterns
            )
        except Exception as e:
            logger.error(f"Failed to load mapping config: {e}")
            return cls._default_mapping()

    @classmethod
    def _default_mapping(cls) -> 'ENSColumnMapping':
        """Маппинг по умолчанию."""
        return cls(
            base_mapping={
                'код': 'код',
                'наименование': 'наименование',
                'полное наименование': 'полное_наименование',
                'mdm key': 'mdm_key',
                'нтд': 'стандарт',
            },
            category_mapping={},
            auto_patterns={}
        )

    def get_mapping_for_category(self, category: str) -> Dict[str, str]:
        """Получение полного маппинга для категории."""
        result = dict(self.base_mapping)
        cat_map = self.category_mapping.get(category, {})
        result.update(cat_map)
        return result

    def auto_map_column(self, column_name: str) -> Optional[str]:
        """Авто-маппинг колонки по паттерну."""
        if not column_name:
            return None
        col_lower = column_name.lower()
        for pattern, mapped_name in self.auto_patterns.items():
            if pattern.search(col_lower):
                return mapped_name
        return None


class PromptsBasedTypeDetector:
    """Определение типа номенклатуры на основе prompts.yaml (с кэшированием)."""

    # Класс-level кэш для prompts
    _cache: Dict[str, Dict] = {}

    def __init__(self, prompts_path: str = "config/prompts.yaml"):
        self.prompts_path = Path(prompts_path) if prompts_path else Path("config/prompts.yaml")
        self.prompts: Dict[str, Any] = {}
        self.category_map: Dict[str, str] = {}
        self._load_prompts()

    def _load_prompts(self):
        """Загрузка prompts.yaml с кэшированием."""
        cache_key = str(self.prompts_path)

        # Проверяем кэш
        if cache_key in PromptsBasedTypeDetector._cache:
            cached = PromptsBasedTypeDetector._cache[cache_key]
            self.prompts = cached['prompts']
            self.category_map = cached['category_map']
            logger.debug(f"Using cached prompts from {self.prompts_path}")
            return

        # Ищем файл
        if not self.prompts_path.exists():
            alt_paths = [
                Path("prompts.yaml"),
                Path("../config/prompts.yaml"),
                Path("../../config/prompts.yaml"),
            ]
            for alt in alt_paths:
                if alt.exists():
                    self.prompts_path = alt
                    cache_key = str(self.prompts_path)
                    break

        if not self.prompts_path.exists():
            logger.warning(f"prompts.yaml not found at {self.prompts_path}")
            return

        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.prompts = data.get('prompts', {})

            for pid, cfg in self.prompts.items():
                cat = cfg.get('category', '')
                if cat:
                    self.category_map[cat] = pid

            # Сохраняем в кэш
            PromptsBasedTypeDetector._cache[cache_key] = {
                'prompts': self.prompts,
                'category_map': self.category_map
            }

            logger.info(f"Loaded {len(self.prompts)} prompts from {self.prompts_path} (cached)")
        except Exception as e:
            logger.error(f"Failed to load prompts.yaml: {e}")

    def detect_type(self, text: str) -> Tuple[Optional[str], Optional[str], float]:
        """Определение типа номенклатуры по тексту."""
        if not self.prompts:
            return None, None, 0.0

        text_lower = (text or '').lower()
        best_match = None
        best_category = None
        best_score = 0

        for prompt_id, cfg in self.prompts.items():
            keywords = cfg.get('keywords', [])
            category = cfg.get('category', '')
            score = 0

            for keyword in keywords:
                if keyword is None:
                    continue
                keyword = str(keyword).strip()

                if keyword.startswith('regex:') or keyword.startswith('re:'):
                    pattern = keyword.split(':', 1)[1].strip()
                    try:
                        if re.search(pattern, text, re.IGNORECASE):
                            score += 3
                    except re.error:
                        continue

                elif '*' in keyword or '?' in keyword:
                    pattern = keyword.replace('.', r'\.').replace('*', '.*').replace('?', '.')
                    try:
                        if re.search(pattern, text_lower):
                            score += 2
                    except re.error:
                        continue

                else:
                    if keyword.lower() in text_lower:
                        score += 1
                        if text_lower.startswith(keyword.lower()):
                            score += 1

            if score > best_score:
                best_score = score
                best_match = prompt_id
                best_category = category

        confidence = min(best_score / 5, 1.0) if best_score > 0 else 0.0
        return best_match, best_category, confidence


@dataclass
class ENSSchema:
    """Гибкая схема колонок для категории ЕСН."""
    category: ENSCategory
    column_mapping: Dict[str, str] = field(default_factory=dict)
    content_indicators: List[str] = field(default_factory=list)
    name_columns: List[str] = field(default_factory=list)


class ENSSchemaRegistry:
    """Реестр схем с адаптивным определением."""

    BASE_COLUMNS = ['Код', 'Наименование', 'Полное наименование',
                    'MDM Key', 'НТД', 'Торговая марка', 'Марка материала']

    @classmethod
    def detect_schema(cls, df: pd.DataFrame, sample_rows: int = 10,
                      prompts_path: Optional[str] = None) -> Tuple[ENSCategory, Optional[str], float]:
        """Автоопределение схемы по колонкам, содержимому и prompts.yaml."""
        columns = set((c or '').lower() for c in df.columns if c)

        try:
            sample_text = ' '.join(
                df.iloc[:sample_rows].fillna('').astype(str).values.flatten()
            ).lower()
        except Exception as e:
            logger.warning(f"Failed to get sample text: {e}")
            sample_text = ''

        detector = PromptsBasedTypeDetector(prompts_path) if prompts_path else PromptsBasedTypeDetector()
        prompt_id, category_str, prompt_confidence = detector.detect_type(sample_text[:1000])

        if prompt_id and category_str:
            category_map = {
                'hardware': ENSCategory.HARDWARE,
                'hardware_washer': ENSCategory.WASHER,
                'rolledmetal': ENSCategory.ROLLED_METAL,
                'eri': ENSCategory.ERI,
                'ekb': ENSCategory.EKB,
            }
            detected_cat = category_map.get(category_str, ENSCategory.UNKNOWN)
            if detected_cat != ENSCategory.UNKNOWN:
                logger.info(f"Detected via prompts.yaml: {detected_cat.value} (prompt: {prompt_id}, confidence: {prompt_confidence:.2f})")
                return detected_cat, prompt_id, prompt_confidence

        return ENSCategory.UNKNOWN, None, 0.0


class ENSLoader:
    """Адаптивный загрузчик справочника ЕСН с внешним конфигом маппинга."""

    # Класс-level кэш для детекторов
    _detector_cache: Dict[str, PromptsBasedTypeDetector] = {}

    def __init__(self, file_path: str, category: Optional[ENSCategory] = None,
                 adaptive: bool = True, prompts_path: Optional[str] = None,
                 mapping_config_path: Optional[str] = None):
        self.file_path = Path(file_path)
        self.category = category
        self.adaptive = adaptive
        self.prompts_path = prompts_path
        self.mapping_config_path = mapping_config_path
        self.schema: Optional[ENSSchema] = None
        self.df: Optional[pd.DataFrame] = None
        self.items: List[Dict[str, Any]] = []
        self.available_columns: List[str] = []
        self.detected_prompt_id: Optional[str] = None
        self.detection_confidence: float = 0.0
        self.column_mapping: ENSColumnMapping = ENSColumnMapping()
        self._detector: Optional[PromptsBasedTypeDetector] = None
        self._reverse_mapping: Dict[str, str] = {}  # normalized_field -> source_excel_column

    def load(self) -> List[Dict[str, Any]]:
        """Загрузка и нормализация данных ЕСН."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"ENS file not found: {self.file_path}")

        logger.info(f"Loading ENS reference: {self.file_path}")

        # Загружаем конфиг маппинга
        self.column_mapping = ENSColumnMapping.load(self.mapping_config_path)

        try:
            self.df = pd.read_excel(self.file_path)
            raw_columns = list(self.df.columns)
            self.available_columns = [str(c) if c is not None else f"unnamed_{i}" for i, c in enumerate(raw_columns)]
            self.df.columns = self.available_columns
            logger.info(f"Available columns: {len(self.available_columns)}")
        except Exception as e:
            logger.error(f"Failed to load ENS Excel: {e}")
            raise

        if self.category is None:
            self.category, self.detected_prompt_id, self.detection_confidence =                 ENSSchemaRegistry.detect_schema(self.df, prompts_path=self.prompts_path)

        # Получаем маппинг для категории
        category_str = self.category.value if self.category else 'unknown'
        mapping = self.column_mapping.get_mapping_for_category(category_str)

        # Обратный маппинг: normalized_field -> source Excel column
        self._reverse_mapping = {}

        # Добавляем авто-маппинг для неизвестных колонок
        for col in self.available_columns:
            if col and col.lower() not in [k.lower() for k in mapping.keys()]:
                auto_mapped = self.column_mapping.auto_map_column(col)
                if auto_mapped:
                    mapping[col] = auto_mapped
                    self._reverse_mapping[auto_mapped] = col

        # Также записываем reverse для явного маппинга
        for src_col, dst_field in mapping.items():
            if dst_field not in self._reverse_mapping:
                self._reverse_mapping[dst_field] = src_col

        self.schema = ENSSchema(
            category=self.category or ENSCategory.UNKNOWN,
            column_mapping=mapping,
            name_columns=['Полное наименование', 'Наименование', 'Код', 'НТД']
        )

        logger.info(f"Column mapping: {len(self.schema.column_mapping)} columns mapped")

        self.items = self._normalize_dataframe()

        logger.info(f"Loaded {len(self.items)} ENS items for category: {category_str}")
        return self.items

    @property
    def reverse_mapping(self) -> Dict[str, str]:
        """Обратный маппинг: normalized_field -> исходная колонка Excel."""
        return self._reverse_mapping.copy()

    def _normalize_dataframe(self) -> List[Dict[str, Any]]:
        """Нормализация DataFrame в список словарей."""
        items = []

        for _, row in self.df.iterrows():
            item = self._row_to_normalized_dict(row)
            if item.get('полное_наименование') or item.get('наименование'):
                items.append(item)

        return items

    def _row_to_normalized_dict(self, row: pd.Series) -> Dict[str, Any]:
        """Преобразование строки DataFrame в нормализованный словарь."""
        result = {
            '_ens_category': self.category.value if self.category else 'unknown',
            '_source_file': str(self.file_path.name),
            '_available_columns': self.available_columns,
            '_detected_prompt_id': self.detected_prompt_id,
            '_detection_confidence': self.detection_confidence,
        }

        for col in self.available_columns:
            value = row[col]

            if pd.isna(value):
                value = None
            else:
                value = str(value).strip()
                if col and any(x in (col or '').lower() for x in ['длина', 'диаметр', 'толщина', 'ширина', 'шаг', 'масса']):
                    try:
                        value = self._extract_numeric(value)
                    except:
                        pass

            # Используем маппинг из конфига
            key = self.schema.column_mapping.get(col, (col or '').lower().replace(' ', '_'))
            result[key] = value

            col_safe = (col or "").lower().replace(" ", "_")
            original_key = f"_original_{col_safe}"
            result[original_key] = value

        self._add_implicit_params(result)

        return result

    def _extract_numeric(self, value: str) -> Any:
        """Извлечение числового значения из строки."""
        import re
        if not value:
            return None
        match = re.search(r'[\d]+[.,]?[\d]*', str(value).replace(',', '.'))
        if match:
            num_str = match.group().replace(',', '.')
            try:
                return float(num_str) if '.' in num_str else int(num_str)
            except ValueError:
                return num_str
        return value

    def _get_detector(self) -> PromptsBasedTypeDetector:
        """Получение детектора (с кэшированием)."""
        cache_key = str(self.prompts_path or "default")

        if self._detector is None:
            if cache_key in ENSLoader._detector_cache:
                self._detector = ENSLoader._detector_cache[cache_key]
                logger.debug("Using cached detector")
            else:
                self._detector = PromptsBasedTypeDetector(self.prompts_path) if self.prompts_path else PromptsBasedTypeDetector()
                ENSLoader._detector_cache[cache_key] = self._detector
                logger.debug("Created new detector and cached")

        return self._detector

    def _add_implicit_params(self, item: Dict[str, Any]):
        """Добавление неявных параметров с использованием prompts.yaml."""
        name = str(item.get('полное_наименование') or item.get('наименование') or '')

        if not name:
            logger.debug("Empty name, skipping type detection")
            return

        try:
            detector = self._get_detector()
            prompt_id, category, confidence = detector.detect_type(name)
        except Exception as e:
            logger.warning(f"Type detection failed for '{name[:50]}...': {e}")
            prompt_id, category, confidence = None, None, 0.0

        if prompt_id:
            item['_detected_prompt_id'] = prompt_id
            item['_detected_category'] = category
            item['_detection_confidence'] = confidence

            if category:
                type_map = {
                    'hardware': 'крепеж',
                    'hardware_washer': 'шайба',
                    'rolledmetal': 'прокат',
                    'eri': 'эри',
                    'ekb': 'экб',
                }
                item['тип'] = type_map.get(category, category)
                item['_implicit_тип'] = True

        # НЕ заполняем пустые поля дефолтными значениями —
        # пустые ячейки в CSV должны оставаться None/пустыми,
        # чтобы LLM видел что параметр может отсутствовать.

    def get_column_info(self) -> Dict[str, Any]:
        """Получение информации о колонках файла."""
        if self.df is None:
            raise RuntimeError("DataFrame not loaded. Call load() first.")

        info = {
            'total_columns': len(self.available_columns),
            'columns': self.available_columns,
            'mapped_columns': list(self.schema.column_mapping.keys()) if self.schema else [],
            'category': self.category.value if self.category else None,
            'detected_prompt_id': self.detected_prompt_id,
            'detection_confidence': self.detection_confidence,
            'sample_values': {}
        }

        for col in self.available_columns[:20]:
            non_null = self.df[col].dropna()
            if len(non_null) > 0:
                info['sample_values'][col] = str(non_null.iloc[0])[:50]

        return info

    def get_training_examples(self, min_params: int = 2) -> List[Dict]:
        """Получение обучающих примеров для NER/LLM."""
        examples = []

        for item in self.items:
            full_name = item.get('полное_наименование') or item.get('наименование')
            if not full_name:
                continue

            params = {k: v for k, v in item.items()
                     if not k.startswith('_') and v is not None and v != ''}

            if len(params) >= min_params:
                examples.append({
                    'text': full_name,
                    'params': params,
                    'entities': self._extract_entities(full_name, params),
                    'ens_code': item.get('код'),
                    'mdm_key': item.get('mdm_key'),
                    'detected_prompt_id': item.get('_detected_prompt_id'),
                    'detected_category': item.get('_detected_category'),
                })

        return examples

    def _extract_entities(self, text: str, params: Dict) -> List[Dict]:
        """Извлечение позиций сущностей в тексте для NER."""
        entities = []
        if not text:
            return entities

        text_lower = text.lower()

        for param_name, param_value in params.items():
            if param_value is None or not isinstance(param_value, (str, int, float)):
                continue

            param_value = str(param_value)
            if not param_value:
                continue

            value_lower = param_value.lower()
            if value_lower in text_lower:
                start = text_lower.index(value_lower)
                end = start + len(value_lower)
                entities.append({
                    'start': start,
                    'end': end,
                    'label': param_name.upper(),
                    'value': param_value
                })

        return sorted(entities, key=lambda x: x['start'])

    def build_fuzzy_index(self):
        """Построение индекса для нечеткого поиска."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = []
        for item in self.items:
            name = item.get('полное_наименование') or item.get('наименование', '')
            texts.append(str(name))

        if not texts:
            return None

        self.vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),
            analyzer='char',
            lowercase=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.indexed_items = self.items

        logger.info(f"Built fuzzy index for {len(texts)} ENS items")
        return self

    def find_similar(self, query: str, k: int = 5) -> List[Dict]:
        """Поиск похожих записей в ЕСН."""
        if not hasattr(self, 'vectorizer'):
            self.build_fuzzy_index()

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = similarities.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                item = dict(self.indexed_items[idx])
                item['_similarity'] = float(similarities[idx])
                results.append(item)

        return results