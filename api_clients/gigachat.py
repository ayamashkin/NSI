"""
GigaChat API Client Module
Клиент для работы с GigaChat API через официальную библиотеку Сбера.
"""

import logging
import base64
from typing import Dict, Any, Optional, List

from gigachat import GigaChat
from gigachat.exceptions import AuthenticationError, ResponseError

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class GigaChatClient(BaseLLMClient):
    """
    Клиент для GigaChat API с использованием официальной библиотеки.

    Установка: pip install gigachat
    Документация: https://github.com/ai-forever/gigachat
    """

    def __init__(
        self,
        base_url: str = "https://gigachat.devices.sberbank.ru/api/v1",
        api_key: Optional[str] = None,  # Authorization Key (Base64)
        scope: str = "GIGACHAT_API_PERS",
        timeout: int = 120,
        verify_ssl: bool = False
    ):
        """
        Args:
            api_key: Authorization Key (Base64) из личного кабинета
            scope: GIGACHAT_API_PERS, GIGACHAT_API_B2B или GIGACHAT_API_CORP
        """
        super().__init__(base_url, api_key, timeout)
        self.scope = scope
        self.verify_ssl = verify_ssl

        # Исправляем формат credentials если нужно
        credentials = self._fix_credentials_format(api_key)

        # Инициализация официального клиента
        try:
            self.client = GigaChat(
                credentials=credentials,
                scope=scope,
                verify_ssl_certs=verify_ssl,
                timeout=timeout
            )
            logger.info("GigaChat client initialized via official library")
        except Exception as e:
            logger.error(f"Failed to initialize GigaChat client: {e}")
            raise

    def _fix_credentials_format(self, api_key: Optional[str]) -> str:
        """
        Исправляет формат credentials для библиотеки gigachat.
        Ожидается чистый Base64 без префиксов.
        """
        if not api_key:
            raise ValueError("Authorization Key is required")

        creds = api_key.strip()

        # Убираем префиксы если есть
        if creds.startswith('Basic '):
            creds = creds[6:].strip()
        elif creds.startswith('Bearer '):
            creds = creds[7:].strip()

        # Проверяем что это похоже на Base64
        import re
        if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', creds):
            logger.warning(f"Credentials don't look like Base64: {creds[:20]}...")

        # Добавляем padding если нужно (длина должна быть кратна 4)
        padding_needed = 4 - (len(creds) % 4)
        if padding_needed != 4:
            creds += '=' * padding_needed
            logger.debug(f"Added {padding_needed} padding characters")

        return creds

    def complete(self, prompt: str, model: str, temperature: float = 0.1) -> Dict[str, Any]:
        """Отправка запроса на генерацию."""
        try:
            response = self.client.chat({
                "model": model,
                "messages": [
                    {"role": "system", "content": "Вы - эксперт по техническим стандартам ГОСТ."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "stream": False
            })

            content = response.choices[0].message.content
            parsed = self._extract_json_from_response(content)

            return {
                "success": parsed is not None,
                "content": parsed,
                "raw": content,
                "model": model,
                "error": None if parsed else "JSON parse error",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except AuthenticationError as e:
            logger.error(f"GigaChat authentication failed: {e}")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": f"Authentication error: {e}",
                "model": model
            }
        except ResponseError as e:
            logger.error(f"GigaChat API error: {e}")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": f"API error: {e}",
                "model": model
            }
        except Exception as e:
            logger.error(f"GigaChat request failed: {e}")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": str(e),
                "model": model
            }

    def health_check(self) -> bool:
        """Проверка доступности API."""
        try:
            self.client.token
            return True
        except Exception as e:
            logger.debug(f"GigaChat health check failed: {e}")
            return False

    def get_models(self) -> List[str]:
        """Получение списка доступных моделей."""
        try:
            models_response = self.client.get_models()
            models = []
            for model in models_response.data:
                # Пробуем разные варианты имени атрибута
                model_id = getattr(model, 'id_',
                                   getattr(model, 'id',
                                           getattr(model, 'name', str(model))))
                models.append(model_id)

            return sorted(models)
        except Exception as e:
            logger.error(f"Failed to get models from GigaChat: {e}")
            return ["GigaChat", "GigaChat-Pro", "GigaChat-Max"]