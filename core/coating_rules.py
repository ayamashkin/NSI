"""
Статические правила сопоставления покрытий.
Сгенерировано из Excel-справочника: не требует pandas/openpyxl в runtime.
"""
from typing import Dict, FrozenSet, Optional

# === ПРЯМОЕ СОПОСТАВЛЕНИЕ: ГОСТ/ОСТ код → ЕНС покрытие ===
GOST_TO_ENS: Dict[str, str] = {
    # ГОСТ 9.306-85
    "Оксидирование": "Окс",
    "Оксидирование (вариант 1)": "Окс.1",
    "Оксидирование (вариант 2)": "Окс.2",
    "Цинкование (вариант 2)": "Ц2",
    "Цинкование (вариант 3)": "Ц3",
    "Цинкование (вариант 4)": "Ц4",
    "Хроматирование (вариант 1)": "Хр.1",
    "Хроматирование (вариант 2)": "Хр.2",
    "Пассивация (вариант 1)": "Пас.1",
    "Пассивация (вариант 2)": "Пас.2",
    "Оксидирование и покрытие маслом": "Окс.М",
    "Оксидирование и лакировка": "Окс.Л",
    "Фосфатирование (вариант 1)": "Фос.1",
    "Фосфатирование (вариант 2)": "Фос.2",
    "Фосфатирование (вариант 3)": "Фос.3",
    "Фторопластовое покрытие": "Фтор",
    # Дополнительные ГОСТ коды
    "Анодирование": "Ан",
    "Пассивация химическая": "Хим.Пас",
    "Оксидирование + пассивация": "Окс.Пас",
}

OST_TO_ENS: Dict[str, str] = {
    # ОСТ 1 31101-80
    "Анодирование": "Ан.Окс",
    "Цинкование": "Ц",
    "Кадмирование": "Кд",
    "Химическое пассивирование": "Хим.Пас",
    "Без покрытия": "Бп",
    "Оксидирование": "Окс",
    "Фосфатирование": "Фос",
    "Лакировка": "Лак",
    "Никелирование": "Ник",
    "Хромирование": "Хр",
    "Серебрение": "Сер",
    "Оловяние": "Олов",
    "Цинкование хроматированное": "Ц.Хр",
    "Кадмирование хроматированное": "Кд.Хр",
    "Кадмирование пассивированное": "Кд.Пас",
    "Цинкование пассивированное": "Ц.Пас",
    "Никелирование пассивированное": "Ник.Пас",
    "Хромирование пассивированное": "Хр.Пас",
}

# === ТОКЕН-БАЗОВЫЕ ЭКВИВАЛЕНТЫ (перестановка) ===
# Ключ: frozenset токенов → нормализованное покрытие ЕНС
TOKEN_EQUIVALENTS: Dict[FrozenSet, str] = {
    frozenset({"окс", "фос", "эфп"}): "Окс.Фос.ЭФП",
    frozenset({"фос", "окс", "эфп"}): "Окс.Фос.ЭФП",
    frozenset({"окс", "фос"}): "Окс.Фос",
    frozenset({"фос", "окс"}): "Окс.Фос",
    frozenset({"кд", "хр"}): "Кд.Хр",
    frozenset({"хр", "кд"}): "Кд.Хр",
    frozenset({"кд", "пас"}): "Кд.Пас",
    frozenset({"пас", "кд"}): "Кд.Пас",
    frozenset({"кд", "фос", "окс"}): "Кд.фос.окс",
    frozenset({"кд", "окс", "фос"}): "Кд.фос.окс",
    frozenset({"ц", "хр"}): "Ц.Хр",
    frozenset({"ц", "пас"}): "Ц.Пас",
    frozenset({"ан", "окс"}): "Ан.Окс",
    frozenset({"ник", "пас"}): "Ник.Пас",
    frozenset({"хр", "пас"}): "Хр.Пас",
    frozenset({"окс", "м"}): "Окс.М",
    frozenset({"окс", "л"}): "Окс.Л",
}

# === НОРМАЛИЗАЦИЯ ЦИФР-ПРЕФИКСОВ ===
# Кд3 → Кд, Ц2 → Ц, Хр1 → Хр
DIGIT_PREFIXES = {"кд", "ц", "хр", "ник", "пас", "окс", "фос", "ан", "бп"}


def _tokenize(text: str) -> FrozenSet:
    """Разбить покрытие на токены."""
    import re
    # Убираем цифры в конце токенов (Кд3 → Кд)
    text = re.sub(r'(?<=[a-zA-Zа-яА-Я])\d+$', '', text)
    text = re.sub(r'(?<=[a-zA-Zа-яА-Я])\d+(?=[\.\-])', '', text)
    tokens = re.split(r'[.\-\s]+', text.lower())
    return frozenset(t for t in tokens if t)


def normalize_coating(coating: Optional[str]) -> Optional[str]:
    """
    Нормализовать покрытие из номенклатуры в покрытие ЕНС.

    Returns:
        Покрытие в формате ЕНС или None.
    """
    if not coating:
        return None

    coating = coating.strip().rstrip('.')

    # 1. Точное совпадение с ЕНС-кодом
    all_ens = set(GOST_TO_ENS.values()) | set(OST_TO_ENS.values()) | set(TOKEN_EQUIVALENTS.values())
    if coating in all_ens:
        return coating

    # 2. Прямой mapping ГОСТ → ЕНС
    if coating in GOST_TO_ENS:
        return GOST_TO_ENS[coating]

    # 3. Прямой mapping ОСТ → ЕНС
    if coating in OST_TO_ENS:
        return OST_TO_ENS[coating]

    # 4. Token-based перестановка
    tokens = _tokenize(coating)
    if tokens in TOKEN_EQUIVALENTS:
        return TOKEN_EQUIVALENTS[tokens]

    # 5. Нормализация цифр-префиксов (Кд3 → Кд)
    normalized = coating
    import re
    for prefix in DIGIT_PREFIXES:
        pattern = re.compile(rf'\b{prefix}\d+\b', re.IGNORECASE)
        if pattern.search(coating):
            normalized = pattern.sub(prefix.capitalize(), normalized)
            # Повторяем token-based поиск
            tokens2 = _tokenize(normalized)
            if tokens2 in TOKEN_EQUIVALENTS:
                return TOKEN_EQUIVALENTS[tokens2]
            # Если нет в token map — возвращаем нормализованное
            return normalized

    return None


def get_all_ens_coatings() -> set:
    """Все покрытия ЕНС."""
    return set(GOST_TO_ENS.values()) | set(OST_TO_ENS.values()) | set(TOKEN_EQUIVALENTS.values())