"""
Main Processor Module
Интеграция всех уровней: StandardExtractor -> MaskDatabase -> LLM Generator ->
AutoValidator -> ParametricMatch -> TF-IDF Fallback

VERSION: 2025-05-06-fix7 (double-dollar-fix)
LAST_FIX: 2026-05-06 22:05 — confidence=final_score; ens_params + ens_params_mask; coating normalization; strict exact
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
    item_type: str = ''
    standard: str = ''

    @property
    def ens_params(self) -> Optional[Dict[str, Any]]:
        """Параметры из ENS записи (только при наличии ens_code)."""
        if (self.ens_match and self.ens_match.get('code')
                and 'params' in self.ens_match and self.ens_match['params']):
            return self.ens_match['params']
        return None


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
        logger.info(f"[PROCESS] Searching mask: standard='{standard}', item_type='{search_item_type}'")
        mask = self.mask_db.get_mask(standard, search_item_type)
        logger.info(f"[PROCESS] Search by (standard+UPPER): found={mask is not None}")

        # Fallback: пробуем исходный регистр
        if mask is None:
            mask = self.mask_db.get_mask(standard, item_type)
            if mask:
                logger.info(f"[PROCESS] Found mask with original item_type: {item_type}")

        # Fallback 2: ищем маску только по стандарту (любой item_type)
        if mask is None:
            try:
                all_masks = self.mask_db.get_all_masks()
                standard_masks = [m for m in all_masks if m.standard == standard]
                logger.info(f"[PROCESS] Found {len(standard_masks)} masks for standard='{standard}' (any item_type)")
                if standard_masks:
                    mask = standard_masks[0]
                    logger.info(f"[PROCESS] Using mask with item_type='{mask.item_type}'")
            except Exception as e:
                logger.warning(f"[PROCESS] Fallback search by standard error: {e}")

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
            # Убеждаемся что у маски заполнен стандарт (берем извлеченный если в маске пусто)
            effective_standard = getattr(mask, 'standard', None) or standard
            if not getattr(mask, 'standard', None):
                mask.standard = effective_standard
                logger.info(f"[PROCESS] Fixed empty mask.standard -> '{effective_standard}'")
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
        """
        TEXT_FIELDS = {'покрытие', 'материал', 'марка_материала', 'марка_стали'}
        best_match = None
        best_score = 0.0

        for candidate in ens_candidates:
            total_weight = 0.0
            matched_weight = 0.0

            for param_name, extracted_val in extracted_params.items():
                if not extracted_val:
                    continue
                weight = 2.0 if param_name in TEXT_FIELDS else 1.0
                total_weight += weight

                # Ищем соответствующее поле в кандидате ЕСН
                candidate_val = candidate.get(param_name) or candidate.get(param_name.replace('_', ' '), '')

                if param_name in TEXT_FIELDS:
                    sim = self._token_similarity(extracted_val, candidate_val)
                    if sim >= 0.8:  # 80% токенов совпадают
                        matched_weight += weight * sim
                else:
                    # Числовые параметры — точное совпадение
                    if str(extracted_val).strip() == str(candidate_val).strip():
                        matched_weight += weight

            if total_weight > 0:
                score = matched_weight / total_weight
                logger.debug(f"[FUZZY] Candidate '{candidate.get('наименование', 'N/A')[:40]}': score={score:.3f}, weight={total_weight:.1f}, matched={matched_weight:.1f}")
                if score > best_score:
                    best_score = score
                    best_match = {**candidate, '_fuzzy_score': best_score}
            else:
                logger.debug(f"[FUZZY] Candidate '{candidate.get('наименование', 'N/A')[:40]}': no comparable params (weight=0)")

        logger.debug(f"[FUZZY] Best score: {best_score:.3f}, threshold: 0.6, matched: {best_match is not None and best_score >= 0.6}")
        return best_match if best_score >= 0.6 else None


    def _remap_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Переименование неправильных имён групп из LLM-генерированных масок
        в корректные ENS-имена параметров.
        """
        if not params:
            return params

        remapped = dict(params)
        aliases = {
            # Кривые имена групп → правильные ENS-имена
            'наружный_диаметр_диаметр_вписа': 'номинальный_диаметр_резьбы',
            'наружный_диаметр': 'номинальный_диаметр_резьбы',
            'диаметр_вписанной_окружности': 'номинальный_диаметр_резьбы',
            'd_вп': 'номинальный_диаметр_резьбы',
            'наружный_диаметр_головки': 'диаметр_головки',
            'диаметр_резьбы': 'номинальный_диаметр_резьбы',
        }

        for wrong, correct in aliases.items():
            if wrong in remapped:
                value = remapped.pop(wrong)
                # Если correct уже существует (например, номинальный_диаметр_резьбы=26),
                # сдвигаем его в "длина" — типичный случай для Болт ОСТ 1 31133-80
                if correct in remapped and correct == 'номинальный_диаметр_резьбы':
                    remapped['длина'] = remapped[correct]
                remapped[correct] = value
                logger.debug(f"[REMAP] {wrong} → {correct}: {value}")

        return remapped

    def _get_generic_pattern(self, item_type: str, standard: str = None) -> Optional[str]:
        """
        Генерация "generic" паттерна для item_type когда маска БД не срабатывает.
        Извлекает основные параметры без жёсткой привязки к структуре маски.
        """
        _ru_t = chr(0x0442)
        _ru_b = chr(0x0431)
        _ru_v = 'В'

        item_upper = (item_type or '').upper()

        # Generic bolt pattern: Болт (исполнение)?-диаметр-длина-покрытие
        if item_upper in ('БОЛТ', 'БОЛ' + _ru_t, 'BOLT'):
            # Metric bolt: Болт 2M12x1,25-6gx100.58 ГОСТ 7795-70
            if standard and '7795' in standard:
                return (
                    r'^Болт\s*(?:(?P<исполнение>\d+)\s*)?'
                    r'(?:M(?P<номинальный_диаметр_резьбы>\d+)'
                    r'(?:[xX×](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?'
                    r'[-\s]*(?P<класс_поле_допуска>\d+[a-zA-Z])'
                    r'[-\s]*[xX×]?(?P<длина>\d+(?:[.,]\d+)?))'
                    r'(?:[-\s]*(?P<покрытие>[\w.]+))?'
                    r'\s*$'
                )
            # Standard bolt: Болт (2)-8-26-Кд-ОСТ 1 31133-80
            pattern = (
                r'^Болт\s*(?:\((?P<исполнение>\d+)\)[-\s]*)?'
                r'(?P<номинальный_диаметр_резьбы>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<длина>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<покрытие>[\w.]+)'
            )
            if standard:
                pattern += r'(?:[-\s]*.*)?'
            else:
                pattern += r'(?:\s+.*)?'
            return pattern

        # Generic screw pattern: Винт (исполнение)?-диаметр-длина-покрытие
        if item_upper in ('ВИНТ', 'ВИН' + _ru_t, 'SCREW'):
            pattern = (
                r'^Винт\s*(?:\((?P<исполнение>\d+)\)\s*)?'
                r'(?P<номинальный_диаметр_резьбы>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<длина>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<покрытие>[\w.]+)'
            )
            if standard:
                pattern += r'(?:[-\s]*.*)?'
            else:
                pattern += r'(?:\s+.*)?'
            return pattern

        # Generic washer pattern: Шайба диаметр-наружный-толщина-покрытие
        if item_upper in ('ШАЙБА', 'ШАЙ' + _ru_b + chr(0x0430), 'WASHER'):
            pattern = (
                r'^Шайба\s*(?:(?P<тип>[A-Z])\s*)?'
                r'(?P<диаметр>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<наружный_диаметр>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<толщина>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<покрытие>[\w.]+)'
            )
            if standard:
                pattern += r'(?:[-\s]*.*)?'
            else:
                pattern += r'(?:\s+.*)?'
            return pattern

        # Generic nut pattern: Гайка диаметр-покрытие
        if item_upper in ('ГАЙКА', 'ГАЙКА', 'NUT'):
            pattern = (
                r'^Гайка\s*(?:(?P<исполнение>\d+)\s*)?'
                r'(?P<номинальный_диаметр_резьбы>\d+(?:[.,]\d+)?)'
                r'[-\s]+(?P<покрытие>[\w.]+)'
            )
            if standard:
                pattern += r'(?:[-\s]*.*)?'
            else:
                pattern += r'(?:\s+.*)?'
            return pattern

        return None

    def _parametric_match(
        self,
        text: str,
        mask,
        extracted: Dict[str, Any],
        start_time: float
    ) -> ProcessingResult:
        """Параметрическое сопоставление."""
        import time

        # Гарантируем что у маски есть стандарт
        extracted_std = None
        if extracted.get('standard_info'):
            extracted_std = extracted['standard_info'].normalized
        effective_standard = getattr(mask, 'standard', None) or extracted_std or ''
        if not getattr(mask, 'standard', None) and extracted_std:
            mask.standard = extracted_std
            logger.info(f"[PARAM_MATCH] Fixed mask.standard from extracted: '{extracted_std}'")

        logger.debug(f"[PARAM_MATCH] text='{text[:50]}', mask.pattern='{mask.pattern[:50]}...', mask.standard='{effective_standard}', mask.item_type='{mask.item_type}'")

        match_result = self.parametric_client.match(
            text=text,
            standard=effective_standard,
            item_type=mask.item_type
        )

        logger.debug(f"[PARAM_MATCH] score={match_result.score}, matched_params={match_result.matched_params}, ens_code={match_result.ens_code}")

        # Fallback: если parametric_client вернул пустые params но score > 0,
        # извлекаем params напрямую через regex (используем re.search с IGNORECASE,
        # как и в parametric_client._apply_mask)
        fallback_params = None
        if not match_result.matched_params and match_result.score > 0:
            import re
            try:
                relaxed_for_fallback = self.parametric_client._relax_pattern(mask.pattern, standard=effective_standard)
                logger.debug(f"[PARAM_MATCH] Fallback pattern: {relaxed_for_fallback[:200]}")
                m = re.search(relaxed_for_fallback, text, re.IGNORECASE)
                if m:
                    fallback_params = {k: v for k, v in m.groupdict().items() if v is not None}
                    if fallback_params:
                        logger.info(f"[PARAM_MATCH] Fallback extraction: {fallback_params}")
                else:
                    logger.warning(f"[PARAM_MATCH] Fallback regex did NOT match. Pattern: {relaxed_for_fallback[:120]}")
            except Exception as e:
                logger.warning(f"[PARAM_MATCH] Fallback extraction error: {e}")

        # Ultimate fallback: generic pattern по item_type
        # Когда маска БД имеет неправильную структуру (лишние/отсутствующие параметры),
        # или regex вообще не сработал — пробуем generic паттерн который извлекает
        # только основные параметры (номинальный_диаметр, длина, покрытие).
        if not fallback_params:
            import re
            try:
                generic_pattern = self._get_generic_pattern(mask.item_type, effective_standard)
                if generic_pattern:
                    # Применяем те же relax-правила к generic pattern
                    relaxed_generic = self.parametric_client._relax_pattern(generic_pattern, standard=effective_standard)
                    logger.debug(f"[PARAM_MATCH] Generic pattern: {relaxed_generic[:200]}")
                    m = re.search(relaxed_generic, text, re.IGNORECASE)
                    if m:
                        fallback_params = {k: v for k, v in m.groupdict().items() if v is not None}
                        if fallback_params:
                            logger.info(f"[PARAM_MATCH] Generic extraction: {fallback_params}")
                    else:
                        logger.debug(f"[PARAM_MATCH] Generic pattern did NOT match")
            except Exception as e:
                logger.warning(f"[PARAM_MATCH] Generic fallback error: {e}")
        # где обязательные скобки/группы делаем опциональными
        relaxed_result = None
        if match_result.score == 0 and not match_result.matched_params:
            try:
                relaxed_pattern = self.parametric_client._relax_pattern(mask.pattern, standard=effective_standard)
                if relaxed_pattern != mask.pattern:
                    # Создаём временную маску с ослабленным паттерном
                    relaxed_mask = type(mask)(
                        id=getattr(mask, 'id', -1),
                        standard=effective_standard,
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
                        standard=effective_standard,
                        item_type=mask.item_type,
                        pattern=relaxed_pattern
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

        # Определяем итоговые params (fallback или обычные)
        final_matched_params = fallback_params if fallback_params else match_result.matched_params

        # Remap неправильных имён групп на корректные ENS-имена
        if final_matched_params:
            final_matched_params = self._remap_params(final_matched_params)
            logger.debug(f"[PARAM_MATCH] Remapped params: {final_matched_params}")

        if match_result.score < 0.7 or not match_result.ens_code:
            try:
                # Получаем кандидатов из ЕСН
                ens_candidates = self.validator._get_ens_examples(effective_standard, mask.item_type)
                logger.info(f"[PARAM_MATCH] ENS candidates for fuzzy: count={len(ens_candidates) if ens_candidates else 0}, standard={effective_standard}, item_type={mask.item_type}")
                # Сначала пробуем fuzzy match с remapped params
                if ens_candidates and final_matched_params:
                    logger.info(f"[PARAM_MATCH] Trying fuzzy match with params: {final_matched_params}")
                    fuzzy_match = self._fuzzy_match_ens(final_matched_params, ens_candidates)
                    if fuzzy_match:
                        fuzzy_ens_code = fuzzy_match.get('код') or fuzzy_match.get('mdm_key')
                        fuzzy_score = fuzzy_match.get('_fuzzy_score', 0.0)
                        logger.info(f"[PARAM_MATCH] Fuzzy fallback matched: score={fuzzy_score:.2f}, ens_code={fuzzy_ens_code}")
                    else:
                        logger.warning(f"[PARAM_MATCH] Fuzzy fallback: no match above threshold 0.6")
                elif not ens_candidates:
                    logger.warning(f"[PARAM_MATCH] Fuzzy fallback: no ENS candidates found")
                elif not final_matched_params:
                    logger.warning(f"[PARAM_MATCH] Fuzzy fallback: empty final_matched_params")
                # Если fuzzy не сработал и есть remapped params — пробуем parametric ENS search
                if not fuzzy_ens_code and final_matched_params:
                    # Фильтруем None значения
                    clean_params = {k: v for k, v in final_matched_params.items() if v is not None}
                    manual_ens = self.parametric_client._find_in_ens(
                        clean_params,
                        list(clean_params.keys()),  # required = все распознанные поля
                        standard=effective_standard
                    )
                    if manual_ens and manual_ens.get('code'):
                        fuzzy_ens_code = manual_ens['code']
                        fuzzy_score = manual_ens.get('_match_score', 0.8)
                        logger.info(f"[PARAM_MATCH] Manual ENS search matched: ens_code={fuzzy_ens_code}, score={fuzzy_score:.2f}")
            except Exception as e:
                logger.warning(f"[PARAM_MATCH] Fuzzy fallback error: {e}")

        # Используем лучший результат (fuzzy или обычный)
        final_ens_code = match_result.ens_code or fuzzy_ens_code
        final_score = max(match_result.score, fuzzy_score)

        # Если есть ens_code но нет имени — получим name из ENS индекса
        final_ens_name = match_result.ens_name
        final_mdm_key = match_result.mdm_key if match_result.ens_code else None
        if final_ens_code and not final_ens_name:
            try:
                ens_item = self._get_ens_by_code(final_ens_code)
                if ens_item:
                    final_ens_name = ens_item.get('наименование') or ens_item.get('name')
                    if not final_mdm_key:
                        final_mdm_key = ens_item.get('mdm_key')
                    logger.debug(f"[PARAM_MATCH] ENS name resolved: '{final_ens_name}'")
            except Exception as e:
                logger.warning(f"[PARAM_MATCH] Failed to resolve ENS name: {e}")

        # Получаем параметры из ENS записи (не из текста!)
        ens_params_from_index = None
        if final_ens_code:
            try:
                ens_item = self._get_ens_by_code(final_ens_code)
                if ens_item:
                    import math
                    # Жёсткая фильтрация: только технические параметры изделия
                    skip_fields = {
                        'нтд', 'код', 'наименование', 'name', 'mdm_key',
                        'полное_наименование', 'наименование_типа', 'наименование_типа.1',
                        'единицы_измерения', 'тип', '_implicit_тип',
                        '_match_score', '_match_type', 'item_type', 'standard'
                    }
                    ens_params_from_index = {}
                    for k, v in ens_item.items():
                        # Пропускаем служебные поля: всё что начинается с _
                        if k.startswith('_'):
                            continue
                        # Пропускаем служебные поля по имени
                        if k in skip_fields:
                            continue
                        # Пропускаем пустые значения
                        if v is None or v == '':
                            continue
                        # Пропускаем NaN (float nan)
                        if isinstance(v, float) and math.isnan(v):
                            continue
                        # Пропускаем списки (например _available_columns — но оно уже отфильтровано по _)
                        if isinstance(v, list):
                            continue
                        ens_params_from_index[k] = v
                    logger.debug(f"[PARAM_MATCH] ENS params from index: {ens_params_from_index}")
            except Exception as e:
                logger.warning(f"[PARAM_MATCH] Failed to get ENS params: {e}")

        # Нормализация типов: float 12.0 → int 12, str "2.0" → int 2
        if final_matched_params:
            final_matched_params = self._normalize_params(final_matched_params)
            logger.debug(f"[PARAM_MATCH] Normalized params: {final_matched_params}")
        if ens_params_from_index:
            ens_params_from_index = self._normalize_params(ens_params_from_index)
            logger.debug(f"[PARAM_MATCH] Normalized ens_params: {ens_params_from_index}")

        processing_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH,
            success=final_score >= 0.99,
            params=final_matched_params,
            ens_match={
                'code': final_ens_code,
                'name': final_ens_name,
                'mdm_key': final_mdm_key or final_ens_code,
                'score': final_score,
                'type': 'fuzzy_fallback' if fuzzy_ens_code and not match_result.ens_code else match_result.match_type,
                'params': ens_params_from_index  # ← из ENS индекса, нормализованные типы
            } if final_ens_code else None,
            confidence=final_score,
            ens_params=ens_params_from_index,
            ens_params_mask=match_result.ens_params_mask if hasattr(match_result, "ens_params_mask") else {},
            processing_time_ms=processing_time,
            details={
                'mask_id': mask.id,
                'mask_pattern': mask.pattern,
                'extracted_standard': extracted.get('standard_info'),
                'extracted_type': extracted.get('item_type'),
                'fuzzy_used': fuzzy_ens_code is not None and not match_result.ens_code
            },
            item_type=mask.item_type,
            standard=mask.standard
        )

    def _normalize_value_types(self, value: Any) -> Any:
        """
        Нормализация типов значений параметров.
        - float 12.0 → int 12
        - str "2.0" → int 2
        - str "1.5" → float 1.5
        - str "abc" → str "abc" (не меняем)
        """
        if isinstance(value, float):
            if value == int(value):
                return int(value)
            return value
        if isinstance(value, str):
            try:
                f = float(value)
                if f == int(f):
                    return int(f)
                return f
            except ValueError:
                return value
        return value

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализация типов для всех значений в словаре параметров."""
        if not params:
            return params
        return {k: self._normalize_value_types(v) for k, v in params.items()}

    def _get_ens_by_code(self, ens_code: str) -> Optional[Dict[str, Any]]:
        """Поиск записи ЕСН по коду в индексе."""
        if not self.parametric_client or not self.parametric_client._ens_index:
            return None
        items = self.parametric_client._ens_index.get('items', [])
        for item in items:
            if str(item.get('код', '')) == str(ens_code) or str(item.get('mdm_key', '')) == str(ens_code):
                return item
        return None

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
            },
            item_type=extracted.get('item_type', ''),
            standard=extracted.get('standard_info', {}).get('standard', '') if isinstance(extracted.get('standard_info'), dict) else ''
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
            details={'error': 'Could not extract standard or type'},
            item_type='',
            standard=''
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