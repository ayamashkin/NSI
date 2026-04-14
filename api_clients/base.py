"""
Base API Client Module
Абстрактный класс для клиентов LLM API.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseLLMClient(ABC):
    """
    Абстрактный базовый класс для клиентов LLM API.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.username = username
        self.password = password

    @abstractmethod
    def complete(self, prompt: str, model: str, temperature: float = 0.1,
                 system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Отправка запроса на генерацию."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Проверка доступности API."""
        pass

    @abstractmethod
    def get_models(self) -> List[str]:
        """
        Получение списка доступных моделей.

        Returns:
            Список названий моделей
        """
        pass

    def _extract_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Извлечение JSON из markdown code blocks или текста."""
        import json

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

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        return None