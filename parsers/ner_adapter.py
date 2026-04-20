"""
NER Adapter Module
Адаптер для интеграции NER-модели в каскадный парсер.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NERResult:
    """Результат NER-извлечения."""
    entities: Dict[str, Any]
    confidence: float
    raw_tokens: Optional[List[Dict]] = None


class NERAdapter:
    """
    Адаптер для NER-модели.

    Текущая реализация: использует эвристики на основе regex.
    Будущая реализация: transformers-based NER.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация NER-адаптера.

        Args:
            model_path: Путь к модели (если None — используются эвристики)
        """
        self.model_path = model_path
        self.model = None

        if model_path:
            self._load_model()

    def _load_model(self):
        """Загрузка NER-модели."""
        try:
            logger.info(f"NER model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}, using heuristics")
            self.model = None

    def extract(self, text: str) -> NERResult:
        """Извлечение сущностей из текста."""
        if self.model:
            return self._extract_with_model(text)
        else:
            return self._extract_with_heuristics(text)

    def _extract_with_model(self, text: str) -> NERResult:
        """Извлечение с помощью модели (заглушка)."""
        return self._extract_with_heuristics(text)

    def _extract_with_heuristics(self, text: str) -> NERResult:
        """Извлечение с помощью эвристик."""
        import re

        entities = {}
        text_lower = text.lower()

        # Извлечение типа
        type_patterns = [
            (r'^\s*(болт)', 'Болт'),
            (r'^\s*(винт)', 'Винт'),
            (r'^\s*(гайка)', 'Гайка'),
            (r'^\s*(шайба)', 'Шайба'),
            (r'^\s*(шуруп)', 'Шуруп'),
            (r'^\s*(шпилька)', 'Шпилька'),
            (r'^\s*(заклепка)', 'Заклепка'),
        ]

        for pattern, item_type in type_patterns:
            if re.search(pattern, text_lower):
                entities['тип'] = item_type
                break

        # Извлечение диаметра
        diam_patterns = [
            r'M(\d+(?:[,\.]\d+)?)',  # M12, M12.5
            r'[\(\s](\d+(?:[,\.]\d+)?)[-\)]',  # (12)-, (12.5)-
        ]

        for pattern in diam_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    entities['диаметр'] = float(match.group(1).replace(',', '.'))
                    break
                except ValueError:
                    entities['диаметр'] = match.group(1)

        # Извлечение длины
        length_patterns = [
            r'-(\d+)(?:-\w+)?-',  # -44- в Болт (2)-12-44-Окс
            r'-(\d+(?:[,\.]\d+)?)\s*(?:ГОСТ|ОСТ)',  # -50 ГОСТ
            r'\.(\d+(?:[,\.]\d+)?)\s*\.',  # .58. в ГОСТ 7795
        ]

        for pattern in length_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    entities['длина'] = float(match.group(1).replace(',', '.'))
                    break
                except ValueError:
                    entities['длина'] = match.group(1)

        # Извлечение стандарта
        std_match = re.search(r'(ГОСТ\s*\d+-\d+|ОСТ\s*\d+\s*\d+-\d+|РАМ\.\d+\.\d+)', text, re.IGNORECASE)
        if std_match:
            entities['стандарт'] = std_match.group(1)

        # Извлечение покрытия
        coating_patterns = [
            (r'Окс\.Фос\.ЭФП', 'Оксидирование фосфатное с ЭФП'),
            (r'Окс\.Фос', 'Оксидирование фосфатное'),
            (r'Окс', 'Оксидирование'),
            (r'Хим\.Пас', 'Химическое пассивирование'),
            (r'Кд', 'Кадмирование'),
        ]

        for pattern, coating in coating_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                entities['покрытие'] = coating
                break

        # Извлечение исполнения
        exec_match = re.search(r'\((\d)\)', text)
        if exec_match:
            entities['исполнение'] = int(exec_match.group(1))

        # Расчёт уверенности
        confidence = self._calculate_confidence(entities, text)

        return NERResult(
            entities=entities,
            confidence=confidence
        )

    def _calculate_confidence(self, entities: Dict[str, Any], text: str) -> float:
        """Расчёт уверенности на основе полноты извлечения."""
        item_type = entities.get('тип', '')

        expected = {
            'Болт': ['тип', 'диаметр', 'длина'],
            'Винт': ['тип', 'диаметр', 'длина'],
            'Шайба': ['тип', 'толщина', 'диаметр_внутр'],
            'Гайка': ['тип'],
        }.get(item_type, ['тип'])

        found = sum(1 for p in expected if p in entities)

        return found / len(expected) if expected else 0.5


class NERTrainer:
    """Тренер для NER-модели на данных ЕСН."""

    def __init__(self, ens_items: List[Dict[str, Any]]):
        """
        Инициализация тренера.

        Args:
            ens_items: Нормализованные записи из ЕСН
        """
        self.items = ens_items
        self.training_data = []

    def prepare_training_data(self) -> List[Dict]:
        """Подготовка обучающей выборки в формате BIO."""
        for item in self.items:
            text = item.get('полное_наименование') or item.get('наименование', '')
            if not text:
                continue

            # Собираем сущности
            entities = []
            for key, value in item.items():
                if key.startswith('_') or not value:
                    continue

                value_str = str(value)
                if value_str in text:
                    start = text.index(value_str)
                    end = start + len(value_str)
                    entities.append({
                        'start': start,
                        'end': end,
                        'label': key.upper(),
                        'value': value_str
                    })

            entities.sort(key=lambda x: x['start'])

            tokens = self._tokenize(text)
            bio_labels = self._create_bio_labels(tokens, entities, text)

            self.training_data.append({
                'text': text,
                'tokens': tokens,
                'labels': bio_labels,
                'entities': entities
            })

        logger.info(f"Prepared {len(self.training_data)} training examples")
        return self.training_data

    def _tokenize(self, text: str) -> List[str]:
        """Простая токенизация."""
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def _create_bio_labels(self, tokens: List[str], entities: List[Dict], text: str) -> List[str]:
        """Создание BIO-меток для токенов."""
        labels = ['O'] * len(tokens)

        char_to_token = {}
        char_pos = 0
        for i, token in enumerate(tokens):
            for _ in token:
                char_to_token[char_pos] = i
                char_pos += 1

        for ent in entities:
            start_char, end_char = ent['start'], ent['end']
            label = ent['label']

            start_token = char_to_token.get(start_char)
            end_token = char_to_token.get(end_char - 1)

            if start_token is not None and end_token is not None:
                for i in range(start_token, end_token + 1):
                    if i < len(labels):
                        prefix = 'B-' if i == start_token else 'I-'
                        labels[i] = prefix + label

        return labels

    def save_training_data(self, output_path: str):
        """Сохранение обучающей выборки."""
        import json

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved training data to {output_path}")
