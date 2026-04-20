"""
Cascade Parser Module
Трёхуровневый каскад для разбора номенклатуры крепежа.
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ParseLevel(Enum):
    """Уровни парсинга."""
    REGEX = "regex"
    NER = "ner"
    LLM = "llm"
    FAILED = "failed"


@dataclass
class ParseResult:
    """Результат разбора номенклатуры."""
    original: str
    params: Dict[str, Any]
    confidence: float
    level: ParseLevel
    raw_match: Optional[Any] = None
    ens_matches: Optional[List[Dict]] = None
    processing_time_ms: Optional[float] = None


class RegexFastenerParser:
    """Уровень 1: Быстрый regex-парсер для стандартных паттернов."""

    # Паттерны для разных типов крепежа
    PATTERNS = {
        # Болты ОСТ 1 31133-80: Болт (2)-12-44-Окс.Фос.ЭФП-ОСТ 1 31133-80
        'bolt_ost_31133': re.compile(
            r'Болт\s*\((?P<исполнение>\d)\)-(?P<диаметр>\d+)-(?P<длина>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*31133-80)',
            re.IGNORECASE
        ),

        # Винты ОСТ 1 31502-80
        'screw_ost_31502': re.compile(
            r'Винт\s*\((?P<исполнение>\d)\)-(?P<диаметр>\d+)-(?P<длина>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*31502-80)',
            re.IGNORECASE
        ),

        # Винты ОСТ 1 31503-80
        'screw_ost_31503': re.compile(
            r'Винт\s*\((?P<исполнение>\d)\)-(?P<диаметр>\d+)-(?P<длина>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*31503-80)',
            re.IGNORECASE
        ),

        # Шайбы ОСТ 1 34505-80
        'washer_ost_34505': re.compile(
            r'Шайба\s+(?P<толщина>[\d,]+)-(?P<диаметр_внутр>\d+)-(?P<диаметр_наружн>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*34505-80)',
            re.IGNORECASE
        ),

        # Шайбы ОСТ 1 34507-80
        'washer_ost_34507': re.compile(
            r'Шайба\s+(?P<толщина>[\d,]+)-(?P<диаметр_внутр>\d+)-(?P<диаметр_наружн>\d+)-(?P<покрытие>[\w\.]+)-(?P<стандарт>ОСТ\s*1\s*34507-80)',
            re.IGNORECASE
        ),

        # Болты ГОСТ 7795-70: Болт 2M12x1,25-6gx100.58 ГОСТ 7795-70
        'bolt_gost_7795': re.compile(
            r'Болт\s+(?P<исполнение>\d)?M(?P<диаметр>[\d,]+)(?:x(?P<шаг>[\d,]+))?-(?P<допуск>[\dgx]+)\.(?P<длина>[\d,]+)\.(?P<материал>\d+)?\s*(?P<стандарт>ГОСТ\s*7795-70)',
            re.IGNORECASE
        ),

        # Болты ГОСТ 7798-70: Болт М12х50 ГОСТ 7798-70
        'bolt_gost_7798': re.compile(
            r'Болт\s+(?P<исполнение>\d)?M?(?P<диаметр>[\d,]+)(?:x(?P<шаг>[\d,]+))?-?(?P<длина>[\d,]+)\s*(?P<стандарт>ГОСТ\s*7798-70)',
            re.IGNORECASE
        ),

        # Гайки простые (по коду)
        'nut_by_code': re.compile(
            r'Гайка\s+(?P<код>[\w\.]+)',
            re.IGNORECASE
        ),
    }

    # Нормализация покрытий
    COATING_NORMALIZATION = {
        'Кд': 'Кадмирование',
        'Окс': 'Оксидирование',
        'Окс.Фос': 'Оксидирование фосфатное',
        'Окс.Фос.ЭФП': 'Оксидирование фосфатное с ЭФП',
        'Хим.Пас': 'Химическое пассивирование',
        'БП': 'Без покрытия',
        '': 'БП',
    }

    TYPE_NORMALIZATION = {
        'болт': 'Болт',
        'винт': 'Винт',
        'гайка': 'Гайка',
        'шайба': 'Шайба',
        'шуруп': 'Шуруп',
        'шпилька': 'Шпилька',
        'заклепка': 'Заклепка',
    }

    def parse(self, text: str) -> Optional[ParseResult]:
        """Парсинг строки номенклатуры."""
        text = text.strip()

        for pattern_name, pattern in self.PATTERNS.items():
            match = pattern.match(text)
            if match:
                params = self._normalize_params(match.groupdict(), pattern_name)
                confidence = self._calculate_confidence(params, pattern_name)

                return ParseResult(
                    original=text,
                    params=params,
                    confidence=confidence,
                    level=ParseLevel.REGEX,
                    raw_match=match
                )

        return None

    def _normalize_params(self, params: Dict[str, Optional[str]], pattern_name: str) -> Dict[str, Any]:
        """Нормализация параметров."""
        result = {}

        # Определение типа
        if 'bolt' in pattern_name:
            result['тип'] = 'Болт'
        elif 'screw' in pattern_name:
            result['тип'] = 'Винт'
        elif 'washer' in pattern_name:
            result['тип'] = 'Шайба'
        elif 'nut' in pattern_name:
            result['тип'] = 'Гайка'

        # Числовые параметры
        for key in ['исполнение', 'диаметр', 'длина', 'шаг', 'толщина', 
                    'диаметр_внутр', 'диаметр_наружн', 'материал']:
            if key in params and params[key]:
                val = params[key].replace(',', '.')
                try:
                    result[key] = float(val) if '.' in val else int(val)
                except ValueError:
                    result[key] = params[key]

        # Покрытие
        coating = params.get('покрытие', '') or ''
        result['покрытие'] = self.COATING_NORMALIZATION.get(coating, coating) if coating else 'БП'
        if not coating:
            result['_implicit_покрытие'] = True

        # Стандарт
        if params.get('стандарт'):
            result['стандарт'] = params['стандарт'].strip()

        # Допуск
        if params.get('допуск'):
            result['допуск'] = params['допуск']

        # Код
        if params.get('код'):
            result['код'] = params['код']

        return result

    def _calculate_confidence(self, params: Dict[str, Any], pattern_name: str) -> float:
        """Расчёт уверенности."""
        base_scores = {
            'bolt_ost_31133': 0.95,
            'screw_ost_31502': 0.95,
            'screw_ost_31503': 0.95,
            'washer_ost_34505': 0.95,
            'washer_ost_34507': 0.95,
            'bolt_gost_7795': 0.90,
            'bolt_gost_7798': 0.90,
            'nut_by_code': 0.70,
        }

        base = base_scores.get(pattern_name, 0.50)

        required = {
            'Болт': ['тип', 'диаметр', 'длина'],
            'Винт': ['тип', 'диаметр', 'длина'],
            'Шайба': ['тип', 'толщина', 'диаметр_внутр'],
            'Гайка': ['тип'],
        }

        item_type = params.get('тип', '')
        missing = sum(1 for p in required.get(item_type, []) if p not in params)

        return max(0.3, base - missing * 0.1)


class LLMNormalizer:
    """Уровень 3: Нормализация через LLM (Qwen3)."""

    def __init__(self, client, ens_index, confidence_threshold: float = 0.7):
        self.client = client
        self.ens_index = ens_index
        self.threshold = confidence_threshold
        self._cache = {}

    async def normalize(self, text: str, context: Optional[Dict] = None) -> ParseResult:
        """Нормализация через LLM с few-shot."""
        import asyncio
        import time

        start_time = time.time()

        # Поиск похожих примеров
        similar = []
        if self.ens_index:
            similar = self.ens_index.find_similar(text, k=3)

        # Формируем промпт
        prompt = self._build_few_shot_prompt(text, context, similar)

        # Проверяем кэш
        cache_key = hash(prompt)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.processing_time_ms = (time.time() - start_time) * 1000
            return cached

        # Запрос к LLM
        try:
            response = await self._call_llm(prompt)

            params = self._extract_json(response)
            params['_llm_raw'] = response.get('raw', '')

            if 'покрытие' not in params or not params['покрытие']:
                params['покрытие'] = 'БП'
                params['_implicit_покрытие'] = True

            processing_time = (time.time() - start_time) * 1000

            result = ParseResult(
                original=text,
                params=params,
                confidence=0.85,
                level=ParseLevel.LLM,
                ens_matches=similar,
                processing_time_ms=processing_time
            )

            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"LLM normalization failed for '{text}': {e}")
            processing_time = (time.time() - start_time) * 1000

            return ParseResult(
                original=text,
                params={'error': str(e), '_raw_text': text},
                confidence=0.0,
                level=ParseLevel.FAILED,
                processing_time_ms=processing_time
            )

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Вызов LLM через клиент."""
        import concurrent.futures
        import asyncio

        loop = asyncio.get_event_loop()

        def _sync_call():
            return self.client.complete(
                prompt=prompt,
                model="qwen3:30b",
                temperature=0.1,
                system_prompt="Вы - эксперт по номенклатуре крепежа ГОСТ/ОСТ."
            )

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, _sync_call)

    def _build_few_shot_prompt(self, text: str, context: Optional[Dict], examples: List[Dict]) -> str:
        """Построение few-shot промпта."""

        few_shot = []
        for ex in examples[:3]:
            ex_text = ex.get('полное_наименование') or ex.get('наименование', '')
            ex_params = {k: v for k, v in ex.items() 
                        if not k.startswith('_') and v is not None}

            few_shot.append(f"Вход: {ex_text}\\nВыход: {json.dumps(ex_params, ensure_ascii=False)}")

        context_str = ""
        if context:
            context_str = f"Предварительный разбор (NER/Regex):\\n{json.dumps(context, ensure_ascii=False)}\\n\\n"

        prompt = f"""Ты — эксперт по номенклатуре крепежа авиационной промышленности. 
Разбери строку на структурированные параметры в формате JSON.

Примеры из справочника ЕСН:
{chr(10).join(few_shot) if few_shot else 'Нет примеров'}

{context_str}Строка для разбора: "{text}"

Правила интерпретации:
1. Если покрытие не указано явно → "покрытие": "БП" (без покрытия)
2. "Кд" = кадмирование, "Окс" = оксидирование, "Хим.Пас" = химическое пассивирование
3. Стандарт может быть ОСТ, ГОСТ, РАМ и т.д.
4. Для болтов: исполнение в скобках, затем диаметр, длина, покрытие
5. Для шайб: толщина-внутр.диам-наружн.диам-покрытие
6. Все числовые значения — как числа (int или float)
7. Неизвестные параметры — null или пустая строка

Выведи ТОЛЬКО JSON без комментариев и markdown:
{{"тип": "...", "исполнение": "...", "диаметр": ..., "длина": ..., "покрытие": "...", "стандарт": "...", "материал": "..."}}"""

        return prompt

    def _extract_json(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение JSON из ответа LLM."""
        content = response.get('content')
        raw = response.get('raw', '')

        if isinstance(content, dict):
            return content

        try:
            if '```json' in raw:
                json_str = raw.split('```json')[1].split('```')[0]
                return json.loads(json_str.strip())

            if '```' in raw:
                json_str = raw.split('```')[1].split('```')[0]
                return json.loads(json_str.strip())

            return json.loads(raw.strip())

        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to extract JSON from LLM response: {e}")
            return {'_parse_error': str(e), '_raw': raw[:500]}


class CascadeParser:
    """Основной каскадный парсер номенклатуры."""

    def __init__(
        self,
        llm_client=None,
        ens_index=None,
        regex_confidence_threshold: float = 0.85,
        use_llm: bool = True
    ):
        self.regex_parser = RegexFastenerParser()
        self.llm_normalizer = None

        if use_llm and llm_client:
            self.llm_normalizer = LLMNormalizer(llm_client, ens_index)

        self.regex_threshold = regex_confidence_threshold
        self.stats = {
            'regex': 0,
            'llm': 0,
            'failed': 0,
            'total': 0
        }

    async def parse(self, text: str) -> ParseResult:
        """Парсинг через каскад."""
        import time
        start_time = time.time()

        self.stats['total'] += 1

        # Уровень 1: Regex
        regex_result = self.regex_parser.parse(text)
        if regex_result and regex_result.confidence >= self.regex_threshold:
            self.stats['regex'] += 1
            regex_result.processing_time_ms = (time.time() - start_time) * 1000
            return regex_result

        # Подготовка контекста для LLM
        context = None
        if regex_result:
            context = regex_result.params

        # Уровень 3: LLM
        if self.llm_normalizer:
            llm_result = await self.llm_normalizer.normalize(text, context)
            if llm_result.level != ParseLevel.FAILED:
                self.stats['llm'] += 1
                return llm_result

        # Fallback
        if regex_result:
            self.stats['regex'] += 1
            regex_result.processing_time_ms = (time.time() - start_time) * 1000
            return regex_result

        self.stats['failed'] += 1
        return ParseResult(
            original=text,
            params={'_error': 'Failed to parse', '_raw': text},
            confidence=0.0,
            level=ParseLevel.FAILED,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    def _parse_sync(self, text: str) -> ParseResult:
        """Синхронный парсинг (только regex)."""
        import time
        start = time.time()

        result = self.regex_parser.parse(text)
        if result:
            self.stats['regex'] += 1
            result.processing_time_ms = (time.time() - start) * 1000
            return result

        self.stats['failed'] += 1
        return ParseResult(
            original=text,
            params={'_error': 'Failed to parse', '_raw': text},
            confidence=0.0,
            level=ParseLevel.FAILED,
            processing_time_ms=(time.time() - start) * 1000
        )

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики."""
        total = self.stats['total']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'regex_pct': round(self.stats['regex'] / total * 100, 2),
            'llm_pct': round(self.stats['llm'] / total * 100, 2),
            'failed_pct': round(self.stats['failed'] / total * 100, 2),
        }
