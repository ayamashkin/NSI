"""
Main Processor Module
Интеграция всех уровней: StandardExtractor -> MaskDatabase -> LLM Generator ->
AutoValidator -> ParametricMatch -> TF-IDF Fallback

VERSION: 2025-05-06-fix9

LAST_FIXES:
  2026-05-07 14:45 UTC+3 — match_type в results.json (name_exact, parametric_full, v2_exact, coating_substituted, fuzzy)
  2026-05-07 12:10 UTC+3 — кэширование ENS candidates и _find_in_ens; оптимизация производительности
  2026-05-07 12:10 UTC+3 — coating_substitution: использует raw_params (до remap, т.к. remap очищал dict)
  2026-05-07 12:10 UTC+3 — coating_substitution в details (original/corrected/material/reason)
  2026-05-07 11:53 UTC+3 — coating auto_substitution ДО exact match (раньше только в fuzzy)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Lazy import для избежания circular dependency
_matching_config = None

def _get_matching_config():
    """Ленивая загрузка MatchingConfig из settings."""
    global _matching_config
    if _matching_config is None:
        try:
            from config.settings import get_settings
            _matching_config = get_settings().matching
        except Exception:
            # Fallback: дефолтные значения если конфиг не загрузился
            class _FallbackMatchingConfig:
                success_threshold = 0.7
                fuzzy_threshold = 0.6
                v2_exact_threshold = 0.99
                coating_similarity_threshold = 0.8
                strict_union_keys = False
            _matching_config = _FallbackMatchingConfig()
    return _matching_config

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


class ProcessingResult:
    """Результат обработки."""
    def __init__(
        self,
        text: str = '',
        level: ProcessingLevel = ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH,
        success: bool = False,
        params: Optional[Dict[str, Any]] = None,
        ens_params: Optional[Dict[str, Any]] = None,
        ens_params_mask: Optional[Dict[str, Any]] = None,
        ens_match: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        processing_time_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        item_type: str = '',
        standard: str = ''
    ):
        self.text = text
        self.level = level
        self.success = success
        self.params = params or {}
        self.ens_params = ens_params or {}
        self.ens_params_mask = ens_params_mask or {}
        self.ens_match = ens_match
        self.confidence = confidence
        self.processing_time_ms = processing_time_ms
        self.details = details or {}
        self.item_type = item_type
        self.standard = standard

    @property
    def ens_params_from_match(self) -> Optional[Dict[str, Any]]:
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

    def _get_coating_variants(self, coating: str) -> List[str]:
        """Generate coating search variants (Кд -> Кд6, Кд9.фос.окс etc)."""
        if not coating:
            return [coating]
        coating_str = str(coating).strip().lower()
        variants = [coating]  # original
        if coating_str in ('кд', 'кд.'):
            for v in ['Кд6', 'Кд9', 'Кд6.фос', 'Кд9.фос',
                      'Кд6.фос.окс', 'Кд9.фос.окс', 'Кд.фос.окс']:
                variants.append(v)
        return variants

    # Кэш для coating_rules (читаем один раз за сессию)
    _coating_rules_cache: Optional[Dict] = None

    # Кэш для ENS candidates по (standard, item_type) — не грузить повторно
    _ens_candidates_cache: Dict[Tuple[str, str], List[Dict]] = {}

    def _get_cached_ens_candidates(self, standard: str, item_type: str) -> List[Dict]:
        """Кэшированная загрузка ENS candidates по (standard, item_type)."""
        key = (standard.upper(), item_type.upper())
        if key not in self._ens_candidates_cache:
            candidates = self.validator._get_ens_examples(standard, item_type) or []
            self._ens_candidates_cache[key] = candidates
            logger.debug(f"[ENS_CACHE] Loaded {len(candidates)} candidates for {key} (cache miss)")
        else:
            candidates = self._ens_candidates_cache[key]
            logger.debug(f"[ENS_CACHE] Using {len(candidates)} cached candidates for {key} (cache hit)")
        return candidates

    def _load_coating_rules(self) -> Optional[Dict]:
        """Получение coating_rules: сначала из settings, fallback на прямое чтение YAML."""
        if self._coating_rules_cache is not None:
            return self._coating_rules_cache

        # Способ 1: через settings (если settings.py обновлён)
        try:
            from config.settings import get_settings
            settings = get_settings()
            coating_rules = getattr(settings, 'coating_rules', None)
            if coating_rules:
                self._coating_rules_cache = coating_rules
                logger.info("[COATING_SUBST] Loaded from settings")
                return coating_rules
        except Exception as e:
            logger.debug(f"[COATING_SUBST] Settings not available: {e}")

        # Способ 2: прямое чтение YAML с поиском по всем возможным путям
        import yaml
        from pathlib import Path
        import os

        search_paths = []

        # Относительно CWD и вверх по дереву
        cwd = Path.cwd()
        for level in range(6):
            search_paths.append(cwd / "config" / "config.yaml")
            search_paths.append(cwd / "config.yaml")
            cwd = cwd.parent

        # Относительно расположения automated_processor.py
        try:
            import inspect
            script_dir = Path(os.path.dirname(os.path.abspath(inspect.getfile(self.__class__))))
            search_paths.extend([
                script_dir / ".." / "config" / "config.yaml",
                script_dir / ".." / ".." / "config" / "config.yaml",
            ])
        except Exception:
            pass

        # Абсолютные пути
        search_paths.extend([
            Path("/app/config/config.yaml"),
            Path("/workspace/config/config.yaml"),
        ])

        # Переменная окружения
        env_path = os.environ.get('NSI_CONFIG_PATH')
        if env_path:
            search_paths.insert(0, Path(env_path))

        for path in search_paths:
            try:
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                    coating_rules = config.get('coating_rules')
                    if coating_rules:
                        self._coating_rules_cache = coating_rules
                        logger.info(f"[COATING_SUBST] Loaded from {path}")
                        return coating_rules
            except Exception:
                continue

        logger.warning("[COATING_SUBST] coating_rules not found anywhere")
        self._coating_rules_cache = {}
        return None

    def _apply_coating_substitution(self, extracted_params: Dict[str, str], ens_candidates: List[Dict]) -> Tuple[Dict[str, str], Optional[Dict]]:
        """
        Применить auto_substitution из coating_rules к extracted_params.

        Логика:
        1. Делаем пробный fuzzy pass для определения марки материала
        2. Если марка требует substitution (например, 14Х17Н2 → Н.Кд вместо Кд)
           → заменяем покрытие в params
        3. Возвращаем (возможно изменённые) params + информацию о substitution

        Returns:
            Tuple[extracted_params (возможно изменённые), substitution_info или None]
            substitution_info содержит: original, corrected, material, reason, rule_note
        """
        logger.debug(f"[COATING_SUBST] Called with coating={extracted_params.get('покрытие')}, candidates={len(ens_candidates)}")

        # Загружаем coating rules напрямую из YAML (не через settings, т.к.
        # оригинальный settings.py может не иметь поля coating_rules)
        coating_rules = self._load_coating_rules()
        if not coating_rules:
            logger.warning("[COATING_SUBST] No coating_rules in config, skipping substitution")
            return extracted_params, None

        # Проверяем включена ли auto_substitution
        if not coating_rules.get('auto_substitution_enabled', False):
            logger.debug("[COATING_SUBST] auto_substitution_enabled=false, skipping")
            return extracted_params, None

        # Пробный fuzzy pass для определения марки материала
        trial_match, trial_debug = self._fuzzy_match_ens_debug(extracted_params, ens_candidates)
        if not trial_match and not trial_debug:
            logger.debug("[COATING_SUBST] No candidates for trial match, skipping")
            return extracted_params, None

        # Определяем марку материала из кандидатов
        material = None
        if trial_match:
            material = trial_match.get('марка_материала') or trial_match.get('марка_стали')
        # Если лучший match не дал марку — берем из key_ens_params первых кандидатов
        if not material and trial_debug:
            for cd in trial_debug[:5]:
                key_params = cd.get('key_ens_params', {})
                material = key_params.get('марка_материала') or key_params.get('марка_стали')
                if material:
                    break

        if not material:
            logger.debug("[COATING_SUBST] Could not determine material from candidates")
            return extracted_params, None

        logger.debug(f"[COATING_SUBST] Detected material: '{material}'")

        # Применяем auto_substitution правила
        import re
        coating = extracted_params.get('покрытие', '')
        if not coating:
            return extracted_params, None

        for rule in coating_rules.get('auto_substitution', []):
            material_pattern = rule.get('material_pattern', '')
            wrong_coating = rule.get('wrong_coating', '')
            correct_coating = rule.get('correct_coating', '')

            # Проверяем material_pattern
            if not re.search(material_pattern, str(material), re.IGNORECASE):
                continue

            # Проверяем wrong_coating (fuzzy match)
            wrong_sim = self._token_similarity(coating, wrong_coating)
            if wrong_sim < 0.5:
                continue

            # Substitution применима! — формируем info для details
            new_params = dict(extracted_params)
            new_params['покрытие'] = correct_coating
            substitution_info = {
                'original': coating,
                'corrected': correct_coating,
                'material': str(material),
                'reason': rule.get('note', ''),
                'rule': {
                    'material_pattern': material_pattern,
                    'wrong_coating': wrong_coating,
                    'correct_coating': correct_coating,
                }
            }
            logger.info(
                f"[COATING_SUBST] Applied: '{coating}' → '{correct_coating}' "
                f"(material='{material}', rule={rule.get('note', '')})"
            )
            return new_params, substitution_info

        logger.debug(f"[COATING_SUBST] No matching rule for coating='{coating}', material='{material}'")
        return extracted_params, None

    def _fuzzy_match_ens(self, extracted_params: Dict[str, str], ens_candidates: List[Dict]) -> Optional[Dict]:
        """
        Fuzzy matching извлечённых параметров с кандидатами из ЕСН.
        Для покрытия пробует expanded variants (Кд -> Кд6/Кд9.фос.окс).
        Возвращает best_match или None. Для debug используйте _fuzzy_match_ens_debug.
        """
        best_match, _ = self._fuzzy_match_ens_debug(extracted_params, ens_candidates)
        return best_match

    def _fuzzy_match_ens_debug(self, extracted_params: Dict[str, str], ens_candidates: List[Dict]) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Fuzzy matching с подробным debug-выводом всех кандидатов.
        Возвращает: (best_match, debug_candidates_list)
        Каждый элемент debug_candidates_list содержит: name, score, matched_params, key_params
        """
        TEXT_FIELDS = {'покрытие', 'материал', 'марка_материала', 'марка_стали'}
        best_match = None
        best_score = 0.0
        debug_candidates = []

        # Expand coating variants for matching
        coating_variants = None
        if 'покрытие' in extracted_params:
            coating_variants = self._get_coating_variants(extracted_params['покрытие'])

        # Debug: выводим извлечённые параметры
        extracted_str = ", ".join(f"{k}={v}" for k, v in extracted_params.items() if v is not None)
        logger.debug(f"[FUZZY] Extracted params: {extracted_str}")
        logger.debug(f"[FUZZY] Total candidates to check: {len(ens_candidates) if ens_candidates else 0}")

        for candidate in ens_candidates:
            total_weight = 0.0
            matched_weight = 0.0
            candidate_debug = {
                'name': candidate.get('наименование', candidate.get('полное_наименование', 'N/A')),
                'ens_code': candidate.get('код', candidate.get('mdm_key', 'N/A')),
                'params_matched': {},
                'params_mismatched': {},
                'params_missing': [],
            }

            for param_name, extracted_val in extracted_params.items():
                if not extracted_val:
                    continue
                weight = 2.0 if param_name in TEXT_FIELDS else 1.0
                total_weight += weight

                # Ищем соответствующее поле в кандидате ЕСН
                candidate_val = candidate.get(param_name) or candidate.get(param_name.replace('_', ' '), '')

                if param_name in TEXT_FIELDS:
                    # For coating: try all variants
                    if param_name == 'покрытие' and coating_variants and len(coating_variants) > 1:
                        best_sim = max(self._token_similarity(v, candidate_val) for v in coating_variants)
                        sim = best_sim
                    else:
                        sim = self._token_similarity(extracted_val, candidate_val)
                    matched = sim >= 0.5
                    if matched:
                        matched_weight += weight * sim
                        candidate_debug['params_matched'][param_name] = f"'{extracted_val}' ~ '{candidate_val}' (sim={sim:.2f})"
                    else:
                        candidate_debug['params_mismatched'][param_name] = f"'{extracted_val}' vs '{candidate_val}' (sim={sim:.2f})"
                else:
                    # Числовые параметры — точное совпадение (нормализуем int vs float)
                    try:
                        matched = float(str(extracted_val).replace(',', '.')) == float(str(candidate_val).replace(',', '.'))
                    except (ValueError, TypeError):
                        matched = str(extracted_val).strip() == str(candidate_val).strip()
                    if matched:
                        matched_weight += weight
                        candidate_debug['params_matched'][param_name] = f"{extracted_val} == {candidate_val}"
                    else:
                        candidate_debug['params_mismatched'][param_name] = f"{extracted_val} != {candidate_val}"

            # Дополнительно: найдём ключевые ENS-параметры кандидата для debug
            key_ens_params = {}
            for key_field in ['длина', 'номинальный_диаметр_резьбы', 'исполнение', 'покрытие', 'марка_материала', 'шаг_резьбы', 'тип_резьбы']:
                if key_field in candidate and candidate[key_field] is not None and str(candidate[key_field]).strip():
                    key_ens_params[key_field] = candidate[key_field]
            candidate_debug['key_ens_params'] = key_ens_params

            if total_weight > 0:
                score = matched_weight / total_weight
                candidate_debug['score'] = round(score, 3)
                candidate_debug['weight'] = round(total_weight, 1)
                candidate_debug['matched_weight'] = round(matched_weight, 1)
                logger.debug(f"[FUZZY] Candidate '{candidate_debug['name'][:50]}': score={score:.3f}, weight={total_weight:.1f}, matched={matched_weight:.1f}")
                # Подробный debug per-parameter (управляется через config.yaml matching.debug_per_parameter)
                if _get_matching_config().debug_per_parameter:
                    ens_params_str = ", ".join(f"{k}={v}" for k, v in key_ens_params.items())
                    logger.debug(f"[FUZZY]   ENS params: {ens_params_str}")
                    if candidate_debug['params_matched']:
                        matched_str = ", ".join(f"{k}: {v}" for k, v in candidate_debug['params_matched'].items())
                        logger.debug(f"[FUZZY]   MATCHED: {matched_str}")
                    if candidate_debug['params_mismatched']:
                        mismatched_str = ", ".join(f"{k}: {v}" for k, v in candidate_debug['params_mismatched'].items())
                        logger.debug(f"[FUZZY]   MISMATCHED: {mismatched_str}")
                if score > best_score:
                    best_score = score
                    best_match = {**candidate, '_fuzzy_score': best_score}
            else:
                candidate_debug['score'] = 0.0
                candidate_debug['reason'] = 'no comparable params (weight=0)'
                logger.debug(f"[FUZZY] Candidate '{candidate_debug['name'][:50]}': no comparable params (weight=0)")

            debug_candidates.append(candidate_debug)

        # Сортируем по score убыванию
        debug_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Итоговый debug: top candidates (только при debug_per_parameter=true)
        if debug_candidates and _get_matching_config().debug_per_parameter:
            top_n = min(5, len(debug_candidates))
            logger.debug(f"[FUZZY] Top {top_n} candidates:")
            for i, cd in enumerate(debug_candidates[:top_n], 1):
                logger.debug(f"[FUZZY]   #{i}: '{cd.get('name','')[:50]}' score={cd.get('score',0)}, code={cd.get('ens_code','N/A')}")
                if cd.get('params_mismatched'):
                    for pk, pv in cd['params_mismatched'].items():
                        logger.debug(f"[FUZZY]       mismatch: {pk}: {pv}")
        logger.debug(f"[FUZZY] Best score: {best_score:.3f}, threshold: 0.6, matched: {best_match is not None and best_score >= 0.6}")
        return (best_match if best_score >= 0.6 else None), debug_candidates


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
                # ГОСТ 7795-70: Болт 2M12x1,25-6gx100.58 ГОСТ 7795-70
                # Точка в "100.58" — разделитель: длина=100, группа=5.8
                return (
                    r'^Болт\s*(?:(?P<исполнение>\d+)\s*)?'
                    r'(?:M(?P<номинальный_диаметр_резьбы>\d+)'
                    r'(?:[xX×](?P<шаг_резьбы>\d+(?:[.,]\d+)?))?'
                    r'[-\s]*(?P<класс_поле_допуска>\d+[a-zA-Z])'
                    r'[xX×](?P<длина>\d+)\.(?P<группа_класс_прочности>\d+(?:\.\d+)?))'
                    r'(?:[-\s]*(?P<покрытие>[\w.]+))?'
                    r'\s*ГОСТ\s*7795-70\s*$'
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
                r'^Винт\s*(?:\((?P<исполнение>\d+)\)[-\s]*)?'
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

        # Информация о coating substitution (если применена)
        substitution_info: Optional[Dict] = None

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

        # === COATING AUTO-SUBSTITUTION ===
        # Всегда проверяем, даже если score высокий — для корректного покрытия в details.
        # Используем raw_params (fallback или matched, до remap).
        raw_params = fallback_params if fallback_params else match_result.matched_params
        logger.debug(f"[PARAM_MATCH] Coating substitution check: raw_params={raw_params}")
        if raw_params and raw_params.get('покрытие'):
            logger.debug(f"[PARAM_MATCH] Coating substitution: has покрытие='{raw_params.get('покрытие')}', calling _apply_coating_substitution")
            try:
                # Загружаем кандидатов для определения марки материала
                ens_candidates_sub = self._get_cached_ens_candidates(effective_standard, mask.item_type)
                logger.debug(f"[PARAM_MATCH] Coating substitution: got {len(ens_candidates_sub)} candidates")
                if ens_candidates_sub:
                    _, substitution_info = self._apply_coating_substitution(raw_params, ens_candidates_sub)
                    if substitution_info:
                        logger.info(f"[PARAM_MATCH] Coating substitution result: {substitution_info}")
                    else:
                        logger.debug(f"[PARAM_MATCH] Coating substitution: no substitution applied")
                else:
                    logger.debug(f"[PARAM_MATCH] Coating substitution: no candidates, skipping")
            except Exception as e:
                logger.warning(f"[PARAM_MATCH] Coating substitution error: {e}", exc_info=True)
        else:
            logger.debug(f"[PARAM_MATCH] Coating substitution: no покрытие in raw_params, skipping")

        # Debug: собираем информацию о всех проверенных кандидатах
        debug_candidates = []

        if match_result.score < 0.7 or not match_result.ens_code:
            try:
                # Получаем кандидатов из ЕСН
                ens_candidates = self._get_cached_ens_candidates(effective_standard, mask.item_type)
                logger.info(f"[PARAM_MATCH] ENS candidates: count={len(ens_candidates)}, standard={effective_standard}, item_type={mask.item_type}")

                # === COATING AUTO-SUBSTITUTION для ПОИСКА ===
                # Если substitution применена — используем исправленные params для поиска.
                search_params = final_matched_params
                if substitution_info and ens_candidates:
                    search_params = dict(final_matched_params)
                    search_params['покрытие'] = substitution_info['corrected']
                    logger.info(f"[PARAM_MATCH] Using substituted params for search: {search_params}")

                # СНАЧАЛА exact match через _find_in_ens (coating expansion + name_exact)
                if search_params:
                    clean_params = {k: v for k, v in search_params.items() if v is not None}
                    manual_ens, find_debug = self.parametric_client._find_in_ens_debug(
                        clean_params,
                        list(clean_params.keys()),
                        standard=effective_standard,
                        text=text,
                        item_type=mask.item_type if mask else None
                    )
                    if find_debug:
                        debug_candidates.extend(find_debug[:10])  # top-10 parametric candidates
                    ens_code_field = manual_ens.get('код') or manual_ens.get('code') if manual_ens else None
                    if manual_ens and ens_code_field:
                        fuzzy_ens_code = ens_code_field
                        fuzzy_score = manual_ens.get('_match_score', 0.8)
                        if not match_result.ens_name:
                            match_result.ens_name = manual_ens.get('наименование') or manual_ens.get('name')
                        logger.info(f"[PARAM_MATCH] Exact ENS search matched: ens_code={fuzzy_ens_code}, score={fuzzy_score:.2f}")

                # ЗАТЕМ fuzzy match (только если exact не сработал)
                if not fuzzy_ens_code and ens_candidates and search_params:
                    logger.info(f"[PARAM_MATCH] Trying fuzzy match with params: {search_params}")

                    # Первый fuzzy pass (с already-substituted params)
                    fuzzy_match, fuzzy_debug = self._fuzzy_match_ens_debug(search_params, ens_candidates)
                    if fuzzy_debug:
                        debug_candidates.extend(fuzzy_debug[:10])

                    # Если substitution была применена и не помогла — пробуем исходные params
                    if not fuzzy_match and search_params != final_matched_params:
                        logger.info(f"[PARAM_MATCH] Substituted params didn't match, trying original: {final_matched_params}")
                        fuzzy_match, fuzzy_debug = self._fuzzy_match_ens_debug(final_matched_params, ens_candidates)
                        if fuzzy_debug:
                            debug_candidates.extend(fuzzy_debug[:10])

                    if fuzzy_match:
                        fuzzy_ens_code = fuzzy_match.get('код') or fuzzy_match.get('mdm_key')
                        fuzzy_score = fuzzy_match.get('_fuzzy_score', 0.0)
                        logger.info(f"[PARAM_MATCH] Fuzzy fallback matched: score={fuzzy_score:.2f}, ens_code={fuzzy_ens_code}")
                    else:
                        logger.warning(f"[PARAM_MATCH] Fuzzy fallback: no match above threshold 0.6")
                elif not ens_candidates:
                    logger.warning(f"[PARAM_MATCH] Fuzzy fallback: no ENS candidates found")
                elif not search_params:
                    logger.warning(f"[PARAM_MATCH] Fuzzy fallback: empty search_params")
            except Exception as e:
                logger.warning(f"[PARAM_MATCH] Fuzzy fallback error: {e}")

        # Используем лучший результат (fuzzy или обычный)
        final_ens_code = match_result.ens_code or fuzzy_ens_code
        final_ens_name = match_result.ens_name
        final_mdm_key = match_result.mdm_key if match_result.ens_code else None

        # Если есть ens_code но нет имени — получим name из ENS индекса
        if final_ens_code and not final_ens_name:
            try:
                ens_item = self._get_ens_by_code(final_ens_code)
                if ens_item:
                    final_ens_name = ens_item.get('наименование') or ens_item.get('name')
                    if not final_mdm_key:
                        final_mdm_key = ens_item.get('mdm_key')
                    logger.info(f"[PARAM_MATCH] ENS name resolved: '{final_ens_name}'")
            except Exception as e:
                logger.info(f"[PARAM_MATCH] Failed to resolve ENS name: {e}")

        # Fallback: парсим text через generic pattern (всегда, для V2 scoring)
        generic_params = None
        if mask:
            try:
                import re
                generic = self._get_generic_pattern(mask.item_type, effective_standard)
                if generic:
                    m = re.search(generic, text, re.IGNORECASE)
                    if m:
                        generic_params = {k: v for k, v in m.groupdict().items() if v is not None}
                        generic_params = self._normalize_params(generic_params)
                        logger.info(f"[PARAM_MATCH] Params from generic on text: {generic_params}")
                        # Если final_matched_params пуст — используем generic
                        if not final_matched_params:
                            final_matched_params = generic_params
            except Exception as e:
                logger.debug(f"[PARAM_MATCH] Generic parse on text error: {e}")

        # НОВАЯ ЛОГИКА: generic pattern → parse ens_name → _calculate_match_score_v2
        v2_score = 0.0
        v2_match_type = None
        v2_computed = False  # был ли V2 score вычислен
        params_for_v2 = final_matched_params or generic_params
        # V2 работает если есть params и есть либо ens_name, либо ens_code (для получения имени)
        if params_for_v2 and (final_ens_name or final_ens_code):
            # Если нет имени но есть код — попробуем получить имя ещё раз
            if not final_ens_name and final_ens_code:
                try:
                    ens_item = self._get_ens_by_code(final_ens_code)
                    if ens_item:
                        final_ens_name = ens_item.get('наименование') or ens_item.get('name')
                        logger.info(f"[PARAM_MATCH] ENS name resolved (V2): '{final_ens_name}'")
                except Exception as e:
                    logger.info(f"[PARAM_MATCH] ENS name resolution failed (V2): {e}")
            try:
                import re
                generic = self._get_generic_pattern(mask.item_type, effective_standard)
                if generic and final_ens_name:
                    ens_name_str = str(final_ens_name)
                    logger.info(f"[PARAM_MATCH] V2 trying generic on ENS name: '{ens_name_str[:60]}', pattern: {generic[:80]}...")
                    m = re.search(generic, ens_name_str, re.IGNORECASE)
                    if m:
                        generic_ens_mask = {k: v for k, v in m.groupdict().items() if v is not None}
                        logger.info(f"[PARAM_MATCH] Generic parsed ENS name: {generic_ens_mask}")
                        v2_score, v2_match_type, v2_details = self.parametric_client._calculate_match_score_v2(
                            text=text,
                            ens_name=final_ens_name,
                            params=params_for_v2,
                            ens_params={},  # не используем ens_params из индекса
                            ens_params_mask=generic_ens_mask,
                            required=list(params_for_v2.keys())
                        )
                        v2_computed = True
                        logger.info(f"[PARAM_MATCH] V2 score: {v2_score}, type: {v2_match_type}")
                    else:
                        logger.info(f"[PARAM_MATCH] V2: generic pattern did NOT match ENS name")
                elif not final_ens_name:
                    logger.info(f"[PARAM_MATCH] V2 skipped: no ENS name for code {final_ens_code}")
                elif not generic:
                    logger.info(f"[PARAM_MATCH] V2 skipped: no generic pattern for {mask.item_type}")
            except Exception as e:
                logger.info(f"[PARAM_MATCH] V2 scoring error: {e}")

        # Финальный score: V2 + fuzzy — берём лучший результат
        # V2 scoring строгий (1.0 или 0.0), fuzzy — мягкий (0.0..1.0)
        # Если V2 дал точный match (1.0) — используем его
        # Если V2 дал 0.0, но fuzzy нашёл кандидата — используем fuzzy
        if v2_computed:
            if v2_score >= _get_matching_config().v2_exact_threshold:
                final_score = v2_score
                if v2_match_type:
                    match_result.match_type = v2_match_type
            else:
                # V2 не подтвердил match — доверяем fuzzy/parametric результату
                final_score = max(match_result.score, fuzzy_score)
        else:
            final_score = max(match_result.score, fuzzy_score)

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

        # Fallback: заполняем ens_params_mask если он пустой, но есть ens_name
        ens_params_mask = match_result.ens_params_mask if hasattr(match_result, "ens_params_mask") else {}
        if not ens_params_mask and final_ens_name:
            # Fallback A: маска БД с релаксацией
            if mask and mask.pattern:
                try:
                    import re
                    relaxed_for_ens = self.parametric_client._relax_pattern(mask.pattern, standard=effective_standard)
                    m = re.search(relaxed_for_ens, str(final_ens_name), re.IGNORECASE)
                    if m:
                        ens_params_mask = {k: v for k, v in m.groupdict().items() if v is not None}
                        logger.debug(f"[PARAM_MATCH] ENS params mask (from mask): {ens_params_mask}")
                except Exception as e:
                    logger.debug(f"[PARAM_MATCH] Failed to parse ens_name with mask: {e}")
            # Fallback B: generic pattern (если маска БД не сработала)
            if not ens_params_mask and mask:
                try:
                    generic = self._get_generic_pattern(mask.item_type, effective_standard)
                    if generic:
                        m = re.search(generic, str(final_ens_name), re.IGNORECASE)
                        if m:
                            ens_params_mask = {k: v for k, v in m.groupdict().items() if v is not None}
                            logger.debug(f"[PARAM_MATCH] ENS params mask (from generic): {ens_params_mask}")
                except Exception as e:
                    logger.debug(f"[PARAM_MATCH] Generic pattern parse error: {e}")

        # Для JSON вывода: remap params и ens_params_mask на правильные ENS-имена
        display_params = self._remap_params(final_matched_params) if final_matched_params else {}
        display_ens_params_mask = self._remap_params(ens_params_mask) if ens_params_mask else {}

        processing_time = (time.time() - start_time) * 1000

        # Определяем match_type для вывода в JSON
        if substitution_info and final_ens_code:
            match_type_out = 'coating_substituted'
        elif fuzzy_ens_code and not match_result.ens_code:
            match_type_out = 'fuzzy_fallback'
        elif match_result.match_type:
            match_type_out = match_result.match_type
        else:
            match_type_out = None

        # Логирование coating_substitution и match_type перед возвратом
        logger.info(f"[PARAM_MATCH] RETURN substitution_info={'SET' if substitution_info else 'NONE'}: {substitution_info}")
        logger.info(f"[PARAM_MATCH] RETURN match_type={match_type_out}")

        return ProcessingResult(
            text=text,
            level=ProcessingLevel.LEVEL_6_PARAMETRIC_MATCH,
            success=final_score >= _get_matching_config().success_threshold,
            params=display_params,
            ens_match={
                'code': final_ens_code,
                'name': final_ens_name,
                'mdm_key': final_mdm_key or final_ens_code,
                'score': final_score,
                'type': match_type_out,
                'params': ens_params_from_index  # ← из ENS индекса, нормализованные типы
            } if final_ens_code else None,
            confidence=final_score,
            ens_params=ens_params_from_index,
            ens_params_mask=display_ens_params_mask,
            processing_time_ms=processing_time,
            details={
                'mask_id': mask.id,
                'mask_pattern': mask.pattern,
                'match_type': match_type_out,  # ← тип сопоставления (name_exact, parametric_full, v2_exact, coating_substituted, fuzzy)
                'extracted_standard': extracted.get('standard_info'),
                'extracted_type': extracted.get('item_type'),
                'fuzzy_used': fuzzy_ens_code is not None and not match_result.ens_code,
                'debug_candidates': debug_candidates[:15] if debug_candidates else [],
                'fuzzy_score': round(fuzzy_score, 3) if fuzzy_score else 0,
                'match_result_score': round(match_result.score, 3) if match_result else 0,
                'v2_score': round(v2_score, 3) if 'v2_score' in locals() else None,
                'v2_computed': v2_computed if 'v2_computed' in locals() else False,
                'coating_substitution': substitution_info,
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