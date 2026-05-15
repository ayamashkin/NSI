"""
AutoValidator Module
Level 3: Автоматическая валидация сгенерированных масок на примерах из ЕСН.
LAST_FIX: 2026-05-15 12:52 UTC+3 — _get_ens_examples: поддержка 'наименование_типа' (ENS mapping) помимо 'тип_изделия'
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Импортируем проверку эквивалентности пустых значений
try:
    from core.parametric_client import _is_empty_equivalent
except ImportError:
    # Fallback если parametric_client недоступен
    def _is_empty_equivalent(field: str, value: Any) -> bool:
        if value is None:
            return True
        val_str = str(value).strip()
        if not val_str:
            return True
        empty_vals = {
            'покрытие': ['БП', 'бп', 'Бп', 'б/п', 'без покрытия', 'без покрыт', 'Б.П.', 'б.п.'],
        }
        return val_str.lower() in [v.lower() for v in empty_vals.get(field, [])]


@dataclass
class ValidationResult:
    """Результат валидации маски."""
    mask_id: Optional[int]
    test_count: int
    success_count: int
    score: float
    passed: bool
    details: List[Dict[str, Any]]
    error_message: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Процент успешных извлечений."""
        if self.test_count == 0:
            return 0.0
        return self.success_count / self.test_count


class AutoValidator:
    """
    Валидатор масок на основе тестовых примеров из ЕСН.

    Features:
    - Тестирование на N примерах (минимум 10)
    - Score = успешные / всего
    - Успешное извлечение = все required params найдены
    - Сохранение тестовых примеров для анализа
    """

    def __init__(
        self,
        min_examples: int = 10,
        activation_threshold: float = 0.85,
        retry_threshold: float = 0.50,
        ens_index_path: Optional[str] = None
    ):
        self.min_examples = min_examples
        self.activation_threshold = activation_threshold
        self.retry_threshold = retry_threshold
        self.ens_index_path = ens_index_path
        self._ens_items: List[Dict] = []

        if ens_index_path and Path(ens_index_path).exists():
            self._load_ens_index()

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        """Token-based Jaccard similarity для покрытий."""
        import re
        if not a or not b:
            return 0.0
        a_str = str(a).lower().strip()
        b_str = str(b).lower().strip()
        if a_str == b_str:
            return 1.0

        def _extract_tokens(text):
            raw = re.findall(r'[a-zA-Zа-яА-Я0-9]+', text)
            cleaned = []
            for t in raw:
                letters = re.sub(r'[0-9]', '', t)
                if letters:
                    cleaned.append(letters)
            return set(cleaned)

        tokens_a = _extract_tokens(a_str)
        tokens_b = _extract_tokens(b_str)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0

    def _load_ens_index(self):
        """Загрузка индекса ЕСН."""
        try:
            import pickle
            with open(self.ens_index_path, 'rb') as f:
                data = pickle.load(f)
            self._ens_items = data.get('items', [])
            logger.info(f"Loaded {len(self._ens_items)} ENS items for validation")
        except Exception as e:
            logger.warning(f"Failed to load ENS index: {e}")

    def validate_mask(
        self,
        pattern: str,
        params: List[str],
        required: List[str],
        standard: str,
        item_type: str,
        ens_examples: Optional[List[Dict]] = None
    ) -> ValidationResult:
        """
        Валидация маски на примерах.

        Args:
            pattern: Regex паттерн
            params: Список параметров
            required: Обязательные параметры
            standard: Стандарт
            item_type: Тип изделия
            ens_examples: Примеры из ЕСН (если None - берем из индекса)

        Returns:
            ValidationResult
        """
        # Получаем примеры
        examples = ens_examples or self._get_ens_examples(standard, item_type)

        if len(examples) < self.min_examples:
            logger.warning(
                f"Not enough examples for {standard}/{item_type}: "
                f"{len(examples)} < {self.min_examples}"
            )
            return ValidationResult(
                mask_id=None,
                test_count=len(examples),
                success_count=0,
                score=0.0,
                passed=False,
                details=[],
                error_message=f"Not enough examples: {len(examples)} < {self.min_examples}"
            )

        # Компилируем паттерн
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return ValidationResult(
                mask_id=None,
                test_count=0,
                success_count=0,
                score=0.0,
                passed=False,
                details=[],
                error_message=f"Invalid regex: {e}"
            )

        # Тестируем на примерах
        details = []
        success_count = 0

        for example in examples[:self.min_examples]:
            text = example.get('полное_наименование') or example.get('наименование', '')
            if not text:
                continue

            test_result = self._test_pattern(compiled_pattern, required, text, example)
            details.append(test_result)

            if test_result['success']:
                success_count += 1

        total_tests = len(details)
        score = success_count / total_tests if total_tests > 0 else 0.0
        passed = score >= self.activation_threshold

        logger.info(
            f"Validation result for {standard}/{item_type}: "
            f"score={score:.2f}, passed={passed}"
        )

        return ValidationResult(
            mask_id=None,
            test_count=total_tests,
            success_count=success_count,
            score=score,
            passed=passed,
            details=details
        )

    def _get_ens_examples(self, standard: str, item_type: str) -> List[Dict]:
        """
        Получение примеров из ЕСН по стандарту и типу.
        Тип изделия берется из 'тип_изделия' или 'наименование_типа' (поле 'Наименование типа' из ЕСН).
        """
        if not self._ens_items:
            return []

        examples = []
        standard_normalized = standard.lower().replace(' ', '')

        for item in self._ens_items:
            # Проверяем стандарт
            item_standard = str(item.get('стандарт', '')).lower().replace(' ', '')
            item_ntd = str(item.get('нтд', '')).lower().replace(' ', '')

            standard_match = (
                standard_normalized in item_standard or
                standard_normalized in item_ntd
            )

            # Проверяем тип — СНАЧАЛА по 'тип_изделия', затем 'наименование_типа' (ENS mapping)
            item_type_val = str(item.get('тип_изделия') or item.get('наименование_типа', '')).lower()
            if not item_type_val:
                # Fallback на 'тип' ( legacy )
                item_type_val = str(item.get('тип', '')).lower()

            type_match = item_type in item_type_val if item_type else False

            if standard_match or type_match:
                examples.append(item)

        return examples

    def _test_pattern(
        self,
        pattern: re.Pattern,
        required: List[str],
        text: str,
        expected: Dict
    ) -> Dict[str, Any]:
        """Тестирование паттерна на одном примере."""
        match = pattern.search(text)

        if not match:
            return {
                'text': text,
                'success': False,
                'error': 'No match',
                'extracted': {},
                'expected': {k: v for k, v in expected.items() if not k.startswith('_')}
            }

        # Извлекаем параметры
        extracted = match.groupdict()

        # Проверяем наличие required параметров
        missing = []
        for param in required:
            extracted_val = extracted.get(param)
            expected_val = expected.get(param)

            # Проверяем эквивалентность пустым значениям
            extracted_empty = extracted_val is None or extracted_val == '' or _is_empty_equivalent(param, extracted_val)
            expected_empty = expected_val is None or expected_val == '' or _is_empty_equivalent(param, expected_val)

            if extracted_empty and expected_empty:
                # Оба пустые — OK (опциональный параметр)
                continue
            elif extracted_val is None or extracted_val == '':
                missing.append(param)

        # === STRICT CHECK: extracted vs expected values ===
        mismatches = []
        for param, extracted_val in extracted.items():
            if extracted_val is None or str(extracted_val).strip() == '':
                continue
            expected_val = expected.get(param)
            if expected_val is None or str(expected_val).strip() == '':
                continue

            extracted_str = str(extracted_val).lower().strip()
            expected_str = str(expected_val).lower().strip()

            if param == 'покрытие':
                sim = self._token_similarity(extracted_str, expected_str)
                if sim < 0.8:
                    mismatches.append(f"{param}: '{extracted_val}' vs '{expected_val}' (sim={sim:.2f})")
            elif extracted_str != expected_str:
                try:
                    extracted_num = float(extracted_str.replace(',', '.'))
                    expected_num = float(expected_str.replace(',', '.'))
                    if extracted_num != expected_num:
                        mismatches.append(f"{param}: {extracted_val} vs {expected_val}")
                except ValueError:
                    mismatches.append(f"{param}: '{extracted_val}' vs '{expected_val}'")

        success = len(missing) == 0 and len(mismatches) == 0

        return {
            'text': text[:100],
            'success': success,
            'missing_params': missing,
            'mismatches': mismatches,
            'extracted': extracted,
            'expected': {k: v for k, v in expected.items() if k in required}
        }

    def validate_with_db(
        self,
        mask_id: int,
        mask_db,
        ens_examples: Optional[List[Dict]] = None
    ) -> ValidationResult:
        """
        Валидация маски из базы данных.

        Args:
            mask_id: ID маски в БД
            mask_db: Экземпляр MaskDatabase
            ens_examples: Примеры из ЕСН

        Returns:
            ValidationResult
        """
        from database.mask_database import MaskRecord

        mask = mask_db.get_mask_by_id(mask_id)
        if not mask:
            return ValidationResult(
                mask_id=mask_id,
                test_count=0,
                success_count=0,
                score=0.0,
                passed=False,
                details=[],
                error_message=f"Mask {mask_id} not found"
            )

        return self.validate_mask(
            pattern=mask.pattern,
            params=mask.params,
            required=mask.required,
            standard=mask.standard,
            item_type=mask.item_type,
            ens_examples=ens_examples
        )

    def get_validation_report(self, mask_id: int, mask_db) -> Dict[str, Any]:
        """Получение детального отчета о валидации."""
        result = self.validate_with_db(mask_id, mask_db)

        return {
            'mask_id': mask_id,
            'score': result.score,
            'passed': result.passed,
            'threshold': self.activation_threshold,
            'test_count': result.test_count,
            'success_count': result.success_count,
            'success_rate': result.success_rate,
            'details': result.details
        }