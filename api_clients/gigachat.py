"""
GigaChat API Client Module
Клиент для работы с GigaChat API через официальную библиотеку Сбера.

LAST_FIXES:
  2026-05-20 2026-05-20 12:49 UTC+3 UTC+3 — complete(): возвращает tokens_prompt/tokens_completion
    на верхнем уровне dict (ранее были только вложены в usage) для совместимости
    с LLMMaskGenerator._call_llm.
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

    def complete(self, prompt: str, model: str, temperature: float = 0.1,
                 system_prompt: Optional[str] = None, max_retries: int = 3) -> Dict[str, Any]:
        """Отправка запроса на генерацию с retry при rate limiting."""
        import time

        # Используем переданный system_prompt или значение по умолчанию
        system_content = system_prompt if system_prompt else "Вы - эксперт по техническим стандартам ГОСТ."

        for attempt in range(max_retries):
            try:
                response = self.client.chat({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "stream": False
                })

                content = response.choices[0].message.content
                parsed = self._extract_json_from_response(content)

                # FIX: выносим tokens на верхний уровень для единообразия
                usage = response.usage if response.usage else None
                tokens_prompt = usage.prompt_tokens if usage else None
                tokens_completion = usage.completion_tokens if usage else None

                return {
                    "success": parsed is not None,
                    "content": parsed,
                    "raw": content,
                    "model": model,
                    "error": None if parsed else "JSON parse error",
                    "tokens_prompt": tokens_prompt,
                    "tokens_completion": tokens_completion,
                }

            except ResponseError as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 секунд
                    logger.warning(
                        f"Rate limited (429), retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"GigaChat API error: {e}")
                    return {
                        "success": False,
                        "content": None,
                        "raw": None,
                        "error": f"API error: {e}",
                        "model": model,
                        "tokens_prompt": None,
                        "tokens_completion": None,
                    }
            except Exception as e:
                logger.error(f"GigaChat request failed: {e}")
                return {
                    "success": False,
                    "content": None,
                    "raw": None,
                    "error": str(e),
                    "model": model,
                    "tokens_prompt": None,
                    "tokens_completion": None,
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