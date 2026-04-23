"""
Main Processor Module
Интеграция всех уровней: StandardExtractor -> MaskDatabase -> LLM Generator ->
AutoValidator -> ParametricMatch -> TF-IDF Fallback
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessingLevel(Enum):
    """Уровни обработки."""
    LEVEL_0_EXTRACT = "standard_extraction"      # Извлечение стандарта
    LEVEL_1_MASK_LOOKUP = "mask_lookup"          # Проверка MaskDatabase
    LEVEL_2_LLM_GENERATE = "llm_generation"      # Генерация маски
    LEVEL_3_VALIDATE = "auto_validation"         # Авто-валидация
    LEVEL_5_SAVE = "save_mask"                   # Сохранение маски
    LEVEL_6_PARAMETRIC_MATCH = "parametric_match"  # Параметрическое сопоставление
    LEVEL_7_TFIDF_FALLBACK = "tfidf_fallback"    # TF-IDF fallback
    LEVEL_8_LLM_DIRECT = "llm_direct"            # Прямой LLM вызов


@dataclass
class ProcessingResult:
    """Результат обработки."""
    text: str
    level: ProcessingLevel
    success: bool
    params: Dict[str, Any]
    ens_match: Optional[Dict[str, Any]]
    confidence: float
    processing_time_ms: float
    details: Dict[str, Any]


class AutomatedParametricProcessor:
    """
    Основной процессор автоматизированного параметрического поиска.

    Архитектура (согласно ROADMAP.md):
    Level 0: Regex Extractor (standard from text)
    Level 1: MaskDatabase (check existing validated masks)
    Level 2: AutoMaskGenerator (LLM local/cloud)
    Level 3: AutoValidator (test on ENS samples, score >= 0.85)
    Level 5: Save to MaskDatabase (auto-approved)
    Level 6: ParametricMatching (extract params, compare)
    Level 7: ENSIndex (TF-IDF fallback)
    Level 8: LLM Direct (few-shot with ENS examples)
    """

    def __init__(
        self,
        mask_db,
        llm_clients: Optional[Dict[str, Any]] = None,
        ens_index_path: Optional[str] = None,
        min_mask_score: float = 0.85,
        max_llm_retries: int = 3,
        use_llm_generation: bool = True,
        settings: Optional[Any] = None
    ):
        """
        Инициализация процессора.

        Args:
            mask_db: Экземпляр MaskDatabase
            llm_clients: Словарь LLM клиентов {provider: client}
            ens_index_path: Путь к индексу ЕСН
            min_mask_score: Минимальный score для активации маски
            max_llm_retries: Максимум попыток LLM генерации
            use_llm_generation: Разрешить LLM генерацию масок
        """
        self.mask_db = mask_db
        self.llm_clients = llm_clients or {}
        self.ens_index_path = ens_index_path
        self.min_mask_score = min_mask_score
        self.max_llm_retries = max_llm_retries
        self.use_llm_generation = use_llm_generation
        self.settings = settings

        # Инициализация компонентов
        self._init_components()

    def _init_components(self):
        """Инициализация внутренних компонентов."""
        # StandardExtractor
        from parsers.standard_extractor import get_standard_extractor
        self.standard_extractor = get_standard_extractor()

        # AutoValidator
        from core.auto_validator import AutoValidator
        self.validator = AutoValidator(
            ens_index_path=self.ens_index_path,
            activation_threshold=self.min_mask_score
        )

        # LLM Generator
        if self.use_llm_generation and self.llm_clients:
            from generators.llm_mask_generator import LLMMaskGenerator
            self.llm_generator = LLMMaskGenerator(
                clients=self.llm_clients,
                settings=self.settings,
                max_retries=self.max_llm_retries
            )
        else:
            self.llm_generator = None

        # Parametric Client
        from core.parametric_client import ParametricENSClient
        self.parametric_client = ParametricENSClient(
            mask_db=self.mask_db,
            ens_index_path=self.ens_index_path
        )

        logger.info("AutomatedParametricProcessor initialized")

    def process(self, text: str) -> ProcessingResult:
        """
        Обработка одной строки номенклатуры.

        Args:
            text: Строка номенклатуры

        Returns:
            ProcessingResult
        """
        import time
        start_time = time.time()

        # Очищаем trailing punctuation (запятые, точки и т.д. в конце строки)
        clean_text = text.strip().rstrip(',.;: ')
        if clean_text != text.strip():
            logger.debug(f"Cleaned trailing punctuation: '{text}' -> '{clean_text}'")

        logger.info(f"Processing: {text[:50]}...")

        # Level 0: Извлечение стандарта
        extracted = self.standard_extractor.extract_all(clean_text)
        standard_info = extracted.get('standard_info')
        item_type = extracted.get('item_type')

        if not standard_info or not item_type:
            # Не удалось извлечь базовую информацию -> Level 8: LLM Direct
            return self._llm_direct_process(clean_text, start_time)

        standard = standard_info.normalized
        logger.info(f"[PROCESS] standard='{standard}', item_type='{item_type}', clean_text='{clean_text[:60]}'")

        # Level 1: Проверка MaskDatabase
        search_item_type = item_type.upper()  # Нормализуем в uppercase (стандарты: БОЛТ, ВИНТ)
        mask = self.mask_db.get_mask(standard, search_item_type)
        # Fallback: пробуем исходный регистр
        if mask is None:
            mask = self.mask_db.get_mask(standard, item_type)
            if mask:
                logger.info(f"[PROCESS] Found mask with original item_type: {item_type}")
        logger.info(f"[PROCESS] mask found: {mask is not None}, is_active: {getattr(mask, 'is_active', False)}")

        if mask is not None and not mask.is_active:
            # Маска найдена но не активна — активируем принудительно
            logger.info(f"[PROCESS] Mask found but inactive, activating")
            try:
                self.mask_db.activate_mask(mask.id)
                mask.is_active = True
            except Exception as e:
                logger.warning(f"[PROCESS] Failed to activate mask: {e}")

        if mask and mask.is_active:
            # Активная маска найдена -> Level 6: ParametricMatch
            logger.info(f"[PROCESS] -> Level 6: ParametricMatch with mask {mask.id}")
            return self._parametric_match(clean_text, mask, extracted, start_time)

        # Level 2: LLM Generation (если разрешено)
        if self.use_llm_generation and self.llm_generator:
            standard_info = extracted.get('standard_info')
            generated_mask = self._generate_mask(standard, item_type, clean_text, standard_info)

            if generated_mask:
                # Level 3: AutoValidation
                validation_result = self._validate_mask(
                    generated_mask, standard, item_type
                )

                if validation_result.passed:
                    # Level 5: Save mask
                    mask_record = self._save_mask(generated_mask, validation_result)

                    if mask_record:
                        # Level 6: ParametricMatch с новой маской
                        return self._parametric_match(
                            text, mask_record, extracted, start_time
                        )
                else:
                    logger.warning(
                        f"Generated mask failed validation: {validation_result.score:.2f}"
                    )

        # Level 7: TF-IDF Fallback
        return self._tfidf_fallback(text, extracted, start_time)

    def _generate_mask(
        self,
        standard: str,
        item_type: str,
        text: str = "",
        standard_info: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Генерация маски через LLM с каскадным keyword resolution."""
        if not self.llm_generator:
            return None

        # Получаем примеры из ЕСН
        examples = self.validator._get_ens_examples(standard, item_type)

        if len(examples) < 10:
            logger.warning(f"Not enough examples for {standard}/{item_type}")
            return None

        # Берем точный тип из ЕСН (тип_изделия = 'Наименование типа')
        ens_item_type = item_type
        if examples:
            first_example = examples[0]
            type_from_ens = first_example.get('тип_изделия')
            if type_from_ens and type_from_ens.strip():
                ens_item_type = type_from_ens.strip().lower()
                logger.info(
                    f"[AutoProcessor] Тип из ЕСН: '{ens_item_type}' "
                    f"(был: '{item_type}')"
                )

        mask, attempts = self.llm_generator.generate_mask(
            standard=standard,
            item_type=ens_item_type,
            examples=examples,
            name=text,
            standard_info=standard_info
        )

        if mask:
            logger.info(f"Generated mask for {standard}/{item_type}")
            return mask

        return None

    def _validate_mask(
        self,
        mask: Dict[str, Any],
        standard: str,
        item_type: str
    ) -> Any:
        """Валидация сгенерированной маски."""
        from database.mask_database import MaskRecord

        # Создаем временную запись для валидации
        temp_mask = MaskRecord(
            standard=standard,
            item_type=item_type,
            pattern=mask['pattern'],
            params=mask['params'],
            required=mask['required']
        )

        # Валидируем
        result = self.validator.validate_mask(
            pattern=temp_mask.pattern,
            params=temp_mask.params,
            required=temp_mask.required,
            standard=standard,
            item_type=item_type
        )

        return result

    def _save_mask(self, mask: Dict[str, Any], validation: Any) -> Optional[Any]:
        """Сохранение валидированной маски в БД."""
        from database.mask_database import MaskRecord

        mask_record = MaskRecord(
            standard=mask['standard'],
            item_type=mask['item_type'],
            pattern=mask['pattern'],
            params=mask['params'],
            required=mask['required'],
            auto_score=validation.score,
            is_active=validation.passed,
            source='llm',
            test_examples=validation.details[:5]  # Сохраняем первые 5 тестов
        )

        mask_id = self.mask_db.save_mask(mask_record, auto_activate=True)

        if mask_id:
            mask_record.id = mask_id
            logger.info(f"Saved mask with ID: {mask_id}")
            return mask_record

        return None

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        """
        Token-based Jaccard similarity для текстовых параметров.
        Решает проблему перестановки токенов: 'Окс.Фос.ЭФП' ~ 'Фос.Окс.ЭФП' = 100%
        Также нормализует цифры: 'Кд3' ~ 'Кд' = 100% (для покрытий цифры не значимы)
        """
        import re
        if not a or not b:
            return 0.0
        # Извлекаем токены, убираем цифры (для покрытий/материалов они не значимы)
        def _extract_tokens(text):
            raw_tokens = re.findall(r'[a-zA-Zа-яА-Я0-9]+', str(text).lower())
            # Убираем цифры из токенов: 'Кд3' -> 'кд', 'фос' -> 'фос'
            cleaned = []
            for t in raw_tokens:
                # Отделяем буквы от цифр
                letters = re.sub(r'[0-9]', '', t)
                if letters:
                    cleaned.append(letters)
            return set(cleaned)

        tokens_a = _extract_tokens(a)
        tokens_b = _extract_tokens(b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def _fuzzy_match_ens(self, extracted_params: Dict[str, str], ens_candidates: List[Dict]) -> Optional[Dict]:
        """
        Fuzzy matching извлечённых параметров с кандидатами из ЕСН.
        Для текстовых полей (покрытие, материал) использует token-similarity.
        Учитывает эквивалентность пустых значений (БП ≡ None).
        """
        TEXT_FIELDS = {'покрытие', 'материал', 'марка_материала', 'марка_стали'}

        # Импортируем проверку эквивалентности
        try:
            from core.parametric_client import _is_empty_equivalent
        except ImportError:
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

        best_match = None
        best_score = 0.0

        for candidate in ens_candidates:
            total_weight = 0.0
            matched_weight = 0.0

            for param_name, extracted_val in extracted_params.items():
                weight = 2.0 if param_name in TEXT_FIELDS else 1.0
                total_weight += weight

                # Ищем соответствующее поле в кандидате ЕСН
                candidate_val = candidate.get(param_name) or candidate.get(param_name.replace('_', ' '), '')

                # Проверяем эквивалентность пустым значениям
                extracted_empty = not extracted_val or _is_empty_equivalent(param_name, extracted_val)
                candidate_empty = not candidate_val or _is_empty_equivalent(param_name, candidate_val)

                if extracted_empty and candidate_empty:
                    # Оба пустые/эквивалентны пустым — полный match
                    matched_weight += weight
                elif extracted_empty or candidate_empty:
                    # Одно пустое, другое заполнено — partial match
                    matched_weight += weight * 0.5
                elif param_name in TEXT_FIELDS:
                    sim = self._token_similarity(extracted_val, candidate_val)
                    if sim >= 0.8:  # 80% токенов совпадают
                        matched_weight += weight * sim
                else:
                    # Числовые параметры — точное совпадение
                    if str(extracted_val).strip() == str(candidate_val).strip():
                        matched_weight += weight

            if total_weight > 0:
                score = matched_weight / total_weight
                if score > best_score:
                    best_score = score
                    best_match = {**candidate, '_fuzzy_score': best_score}

        return best_match if best_score >= 0.6 else None

    def _parametric_match(
        self,
        text: str,
        mask,
        extracted: Dict[str, Any],
        start_time: float
    ) -> ProcessingResult:
        """Параметрическое сопоставление."""
        import time

        logger.info(f"[PARAM_MATCH] text='{text[:50]}', mask.pattern='{mask.pattern[:50]}...', mask.standard='{mask.standard}', mask.item_type='{mask.item_type}'")

        match_result = self.parametric_client.match(
            text=text,
            standard=mask.standard,
            item_type=mask.item_type
        )

        logger.info(f"[PARAM_MATCH] score={match_result.score}, matched_params={match_result.matched_params}")

        # Fallback: если маска не сработала (score=0), пробуем "ослабленную" версию
        # где обязательные скобки/группы делаем опциональными
        relaxed_result = None
        if match_result.score == 0 and not match_result.matched_params:
            try:
                relaxed_pattern = self._relax_pattern(mask.pattern)
                if relaxed_pattern != mask.pattern:
                    # Создаём временную маску с ослабленным паттерном
                    relaxed_mask = type(mask)(
                        id=getattr(mask, 'id', -1),
                        standard=mask.standard,
                        item_type=mask.item_type,
                        pattern=relaxed_pattern,
                        params=mask.params,
                        required=mask.required,
                        auto_score=getattr(mask, 'auto_score', 0),
                        is_active=True,
                        source='relaxed',
                        usage_count=0,
                        test_examples='[]',
                        created_at='',
                        last_used='',
                        pattern_hash=''
                    )
                    relaxed_result = self.parametric_client.match(
                        text=text,
                        standard=mask.standard,
                        item_type=mask.item_type
                    )
                    if relaxed_result.score > 0:
                        logger.info(f"[PARAM_MATCH] Relaxed pattern matched: score={relaxed_result.score}")
                        match_result = relaxed_result
            except Exception as e:
                logger.debug(f"[PARAM_MATCH] Relaxed pattern error: {e}")

        # Fuzzy matching fallback: если точный поиск не дал результата,
        # пробуем token-based matching для текстовых параметров (покрытие, материал)
        fuzzy_ens_code = None
        fuzzy_score = 0.0
        if match_result.score < 0.7 or not match_result.ens_code:
            try:
                # Получаем кандидатов из ЕСН
                ens_candidates = self.validator._get_ens_examples(mask.standard, mask.item_type)
                if ens_candidates and match_result.matched_params:
                    fuzzy_match = self._fuzzy_match_ens(match_result.matched_params, ens_candidates)
                    if fuzzy_match:
                        fuzzy_ens_code = fuzzy_match.get('код') or fuzzy_match.get('mdm_key')
                        fuzzy_score = fuzzy_match.get('_fuzzy_score', 0.0)
                        logger.info(f"[PARAM_MATCH] Fuzzy fallback matched: score={fuzzy_score:.2f}, ens_code={fuzzy_ens_code}")
            except Exception as e:
                logger.warning(f"[PARAM_MATCH] Fuzzy fallback error: {e}")

        # Используем лучший результат (fuzzy или обычный)
        final_ens_code = match_result.ens_code or fuzzy_ens_code
        final_score = max(match_result.score, fuzzy_score)
        final_matched_params = match_result.matched_params

        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH,
            success=final_score > 0.5,
            params=final_matched_params,
            ens_match={
                'code': final_ens_code,
                'mdm_key': match_result.mdm_key if match_result.ens_code else fuzzy_ens_code,
                'score': final_score,
                'type': 'fuzzy_fallback' if fuzzy_ens_code and not match_result.ens_code else match_result.match_type
            } if final_ens_code else None,
            confidence=max(match_result.confidence, fuzzy_score),
            processing_time_ms=processing_time,
            details={
                'mask_id': mask.id,
                'mask_pattern': mask.pattern,
                'extracted_standard': extracted.get('standard_info'),
                'extracted_type': extracted.get('item_type'),
                'fuzzy_used': fuzzy_ens_code is not None and not match_result.ens_code
            }
        )

    def _tfidf_fallback(
        self,
        text: str,
        extracted: Dict[str, Any],
        start_time: float
    ) -> ProcessingResult:
        """TF-IDF fallback — всегда success=False, т.к. параметры не извлечены."""
        import time

        match_result = self.parametric_client._tfidf_fallback(text)
        processing_time = (time.time() - start_time) * 1000

        # TF-IDF fallback не извлекает параметры -> всегда неуспешен
        # ens_code сохраняем как candidate для справки, но не как match
        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_7_TFIDF_FALLBACK,
            success=False,
            params={},
            ens_match=None,  # Не возвращаем match без параметров
            confidence=0.0,
            processing_time_ms=processing_time,
            details={
                'fallback': True,
                'tfidf_score': match_result.score,
                'tfidf_ens_candidate': match_result.ens_code,
                'extracted': extracted
            }
        )

    def _llm_direct_process(self, text: str, start_time: float) -> ProcessingResult:
        """Прямая обработка через LLM (без маски)."""
        import time

        # Здесь можно добавить прямой вызов LLM
        # Пока возвращаем failed result
        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_0_EXTRACT,
            success=False,
            params={},
            ens_match=None,
            confidence=0.0,
            processing_time_ms=processing_time,
            details={'error': 'Could not extract standard or type'}
        )

    def batch_process(self, texts: List[str]) -> List[ProcessingResult]:
        """Пакетная обработка."""
        return [self.process(text) for text in texts]

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика процессора."""
        return {
            'mask_db_stats': self.mask_db.get_statistics(),
            'parametric_client_stats': self.parametric_client.get_stats(),
            'llm_generation_enabled': self.use_llm_generation,
            'min_mask_score': self.min_mask_score,
            'max_llm_retries': self.max_llm_retries
        }