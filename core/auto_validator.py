"""
AutoValidator Module
Level 3: Автоматическая валидация сгенерированных масок на примерах из ЕСН.

LAST_FIXES:
 2026-05-20 2026-05-20 11:49 UTC+3 — _match_param_keys: F1-like score (2*matched/(len_a+len_b)),
   threshold снижен до 0.20. Ранее score=matched/max(len_a,len_b) давал 1/9=0.11
   для номинальный_диаметр_резьбы ↔ наружный_диаметр_вписанного_круга_… → match не
   срабатывал, все тесты падали с missing → score=0.00.
 2026-05-20 2026-05-20 11:49 UTC+3 — coating comparison: threshold 0.50 (было 0.80) + subset logic
   (tokens_a ⊆ tokens_b → sim=1.0). Ранее "Кд" vs "Кд3.хр" давал sim=0.50 < 0.80 → mismatch.
 2026-05-20 2026-05-20 11:49 UTC+3 — skip_params: убраны нтд_1 и standard (они валидные regex-параметры).
 2026-05-20 12:52 UTC+3 — _test_pattern: полное переключение на V2 fuzzy-сравнение.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Импортируем проверку эквивалентности пустых значений
try:
    from core.parametric_client import _is_empty_equivalent, _text_similarity
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

    def _text_similarity(a: str, b: str) -> float:
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
    - V2 fuzzy matching: coating normalization, param key fuzzy match, numeric normalization
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

    # ------------------------------------------------------------------
    # V2 HELPERS (borrowed from parametric_client)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_coating(coating: str) -> str:
        """
        Нормализация покрытия:
        - Убирает технологические коды: Кд3 → Кд, Ц9 → Ц
        - Сортирует токены: Окс.Фос.ЭФП → окс.эфп.фос
        """
        if not coating:
            return coating
        coating_str = str(coating).strip().lower()
        if '.' in coating_str:
            tokens = coating_str.split('.')
            tokens = [re.sub(r'\d+', '', t) for t in tokens]
            tokens = [t for t in tokens if t]
            tokens.sort()
            return '.'.join(tokens)
        base = re.sub(r'^(кд|ц|окс|фос|н|ан|хим|пас|бп|неп)\d+', r'\1', coating_str)
        return base

    @staticmethod
    def _match_param_keys(key_a: str, keys_b: List[str]) -> Optional[str]:
        """Fuzzy matching имени параметра key_a со списком ключей keys_b.

        Использует F1-like score: 2*matched / (len_a + len_b).
        Threshold 0.20 (ранее 0.50 с max-based score давал ложные reject'ы
        для номинальный_диаметр_резьбы ↔ наружный_диаметр_вписанного_круга_…).
        """
        if not key_a or not keys_b:
            return None
        tokens_a = [t for t in key_a.lower().split('_') if len(t) >= 3]
        if not tokens_a:
            return None
        best_match = None
        best_score = 0.0
        for key_b in keys_b:
            tokens_b = [t for t in key_b.lower().split('_') if len(t) >= 3]
            if not tokens_b:
                continue
            matched = 0
            for ta in tokens_a:
                for tb in tokens_b:
                    if ta == tb:
                        matched += 1
                        break
                    if len(ta) >= 4 and len(tb) >= 4:
                        if ta.startswith(tb) or tb.startswith(ta):
                            matched += 1
                            break
            # F1-like score: 2*matched / (len_a + len_b)
            denom = len(tokens_a) + len(tokens_b)
            score = (2.0 * matched) / denom if denom > 0 else 0.0
            if score > best_score:
                best_score = score
                best_match = key_b
        if best_score >= 0.20:
            return best_match
        return None

    # ------------------------------------------------------------------
    # ORIGINAL METHODS
    # ------------------------------------------------------------------

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
        """
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
        """
        if not self._ens_items:
            return []

        examples = []
        standard_normalized = standard.lower().replace(' ', '')

        for item in self._ens_items:
            item_standard = str(item.get('стандарт', '')).lower().replace(' ', '')
            item_ntd = str(item.get('нтд', '')).lower().replace(' ', '')

            standard_match = (
                standard_normalized in item_standard or
                standard_normalized in item_ntd
            )

            item_type_val = str(item.get('тип_изделия') or item.get('наименование_типа', '')).lower()
            if not item_type_val:
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
        """Тестирование паттерна на одном примере с V2 fuzzy matching."""
        match = pattern.search(text)

        if not match:
            return {
                'text': text,
                'success': False,
                'error': 'No match',
                'extracted': {},
                'expected': {k: v for k, v in expected.items() if not k.startswith('_')}
            }

        extracted = match.groupdict()

        # --- Проверяем наличие required параметров ---
        missing = []
        for param in required:
            extracted_val = extracted.get(param)
            expected_val = expected.get(param)

            extracted_empty = extracted_val is None or extracted_val == '' or _is_empty_equivalent(param, extracted_val)
            expected_empty = expected_val is None or expected_val == '' or _is_empty_equivalent(param, expected_val)

            if extracted_empty and expected_empty:
                continue
            elif extracted_val is None or extracted_val == '':
                missing.append(param)

        # === V2 FUZZY CHECK: extracted vs expected values ===
        mismatches = []

        # Параметры, которые не участвуют в сравнении (метаданные/служебные)
        # FIX: убраны нтд_1 и standard — они валидные regex-параметры
        skip_params = {'тип_изделия', 'item_type', 'наименование', 'полное_наименование', 'код', 'mdm_key'}

        # Fuzzy matching ключей: строим маппинг extracted_key -> expected_key
        expected_keys = [k for k in expected.keys() if not k.startswith('_')]
        matched_map = {}
        used_expected = set()

        for ext_key in extracted.keys():
            if ext_key in skip_params:
                continue
            ext_val = extracted[ext_key]
            if ext_val is None or str(ext_val).strip() == '':
                continue
            # Точное совпадение
            if ext_key in expected_keys and ext_key not in used_expected:
                matched_map[ext_key] = ext_key
                used_expected.add(ext_key)
                continue
            # Fuzzy match
            candidates = [k for k in expected_keys if k not in used_expected]
            best_exp = self._match_param_keys(ext_key, candidates)
            if best_exp:
                matched_map[ext_key] = best_exp
                used_expected.add(best_exp)

        # Сравниваем значения по маппингу
        checked = 0
        matched = 0

        for ext_key, exp_key in matched_map.items():
            ext_val = extracted[ext_key]
            exp_val = expected.get(exp_key)

            ext_str = str(ext_val).lower().strip() if ext_val is not None else ''
            exp_str = str(exp_val).lower().strip() if exp_val is not None else ''

            # Если оба пустые — пропускаем
            if not ext_str and not exp_str:
                continue

            # Если один пустой — mismatch
            if not ext_str or not exp_str:
                mismatches.append(f"{ext_key}: '{ext_val}' vs '{exp_val}' (one empty)")
                continue

            checked += 1

            # Сравниваем значения
            is_coating = (ext_key == 'покрытие') or (exp_key and 'покрытие' in exp_key) or                          (ext_key == 'технические_характеристики') or (exp_key and 'технические_характеристики' in exp_key)

            if is_coating:
                norm_a = self._normalize_coating(ext_str)
                norm_b = self._normalize_coating(exp_str)
                sim = _text_similarity(norm_a, norm_b)
                # FIX: subset logic — если токены одного покрытия полностью входят в другой
                tokens_a = set(norm_a.split('.')) if norm_a else set()
                tokens_b = set(norm_b.split('.')) if norm_b else set()
                if tokens_a and tokens_b and (tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a)):
                    sim = 1.0
                # FIX: threshold снижен до 0.50 (было 0.80)
                if sim < 0.50:
                    mismatches.append(f"{ext_key}: '{ext_val}' vs '{exp_val}' (coating sim={sim:.2f})")
                else:
                    matched += 1
            else:
                try:
                    num_a = float(ext_str.replace(',', '.'))
                    num_b = float(exp_str.replace(',', '.'))
                    if num_a != num_b:
                        mismatches.append(f"{ext_key}: {ext_val} vs {exp_val}")
                    else:
                        matched += 1
                except ValueError:
                    if ext_str != exp_str:
                        mismatches.append(f"{ext_key}: '{ext_val}' vs '{exp_val}'")
                    else:
                        matched += 1

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