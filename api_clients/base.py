"""
Base API Client Module
Абстрактный класс для клиентов LLM API.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLLMClient(ABC):
    """
    Абстрактный базовый класс для клиентов LLM API.

    Все конкретные клиенты (OpenWebUI, MWS GPT) должны наследовать этот класс
    и реализовать методы complete() и health_check().
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 120):
        """
        Инициализация клиента.

        Args:
            base_url: Базовый URL API
            api_key: API ключ (опционально)
            timeout: Таймаут запросов в секундах
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    def complete(self, prompt: str, model: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Отправка запроса на генерацию.

        Args:
            prompt: Текст промпта
            model: Название модели
            temperature: Температура генерации

        Returns:
            Словарь с результатом: {
                'success': bool,
                'content': Any,  # Распарсенный JSON или текст
                'raw': str,      # Сырой ответ
                'model': str,    # Использованная модель
                'error': str     # Описание ошибки (если success=False)
            }
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Проверка доступности API.

        Returns:
            True если API доступен
        """
        pass

    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Извлечение JSON из markdown code blocks или текста.

        Args:
            text: Текст ответа от модели

        Returns:
            Распарсенный JSON или None
        """
        import json

        # Пробуем найти JSON в markdown code blocks
        if "```json" in text:
            try:
                json_str = text.split("```json")[1].split("```")[0]
                return json.loads(json_str.strip())
            except (IndexError, json.JSONDecodeError):
                pass

        if "```" in text:
            try:
                json_str = text.split("```")[1].split("```")[0]
                return json.loads(json_str.strip())
            except (IndexError, json.JSONDecodeError):
                pass

        # Пробуем распарсить весь текст как JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        return None
