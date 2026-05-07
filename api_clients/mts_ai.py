"""
MTS AI API Client
OpenAI-compatible API для модели cotype_pro_2.5.

API Endpoint: https://demo6-fundres.dev.mts.ai/
Auth: API_KEY в заголовке Authorization: Bearer {API_KEY}
Docs: https://demo6-fundres.dev.mts.ai/

LAST_FIXES:
  2026-05-07 13:30 UTC+3 — создание клиента по аналогии с mws_gpt
"""

import logging
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path

from api_clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class MTSAIClient(BaseLLMClient):
    """
    Клиент для MTS AI API (cotype).
    OpenAI-compatible: /v1/chat/completions
    """

    def __init__(
        self,
        base_url: str = "https://demo6-fundres.dev.mts.ai/",
        api_key: Optional[str] = None,
        api_key_file: Optional[str] = None,
        timeout: int = 120,
        default_model: str = "cotype_pro_2.5"
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.default_model = default_model

        # Загружаем API key
        if api_key:
            self.api_key = api_key
        elif api_key_file:
            key_path = Path(api_key_file)
            if key_path.exists():
                self.api_key = key_path.read_text(encoding='utf-8').strip()
                logger.info(f"[MTSAI] Loaded API key from {api_key_file}")
            else:
                raise ValueError(f"API key file not found: {api_key_file}")
        else:
            raise ValueError("MTS AI API key required (api_key or api_key_file)")

        logger.info(f"[MTSAI] Client initialized: base_url={self.base_url}, model={default_model}")

    def _get_headers(self) -> Dict[str, str]:
        """Заголовки для запросов к MTS AI API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def complete(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Реализация abstract method из BaseLLMClient.
        Принимает строку prompt, отправляет как single-turn chat completion.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, **kwargs)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Отправка запроса к /v1/chat/completions.

        Args:
            messages: Список сообщений [{"role": "...", "content": "..."}]
            model: Имя модели (default: cotype_pro_2.5)
            temperature: Температура (0.0-2.0)
            max_tokens: Максимум токенов в ответе
            **kwargs: Дополнительные параметры

        Returns:
            Текст ответа модели или None при ошибке
        """
        model = model or self.default_model
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Добавляем дополнительные параметры
        payload.update(kwargs)

        logger.debug(f"[MTSAI] Request: url={url}, model={model}, messages_count={len(messages)}")

        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            usage = data.get("usage", {})
            logger.info(
                f"[MTSAI] Response: tokens_prompt={usage.get('prompt_tokens')}, "
                f"tokens_completion={usage.get('completion_tokens')}, "
                f"content_length={len(content)}"
            )
            return content

        except requests.exceptions.Timeout:
            logger.error(f"[MTSAI] Timeout after {self.timeout}s")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"[MTSAI] HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"[MTSAI] Request error: {e}")
            return None
        except Exception as e:
            logger.error(f"[MTSAI] Unexpected error: {e}")
            return None

    def get_models(self) -> List[str]:
        """Получение списка доступных моделей через /v1/models."""
        try:
            url = f"{self.base_url}/v1/models"
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [m.get('id', m.get('name', str(m))) for m in data.get('data', [])]
                logger.info(f"[MTSAI] Models: {len(models)} found")
                return models
            else:
                logger.warning(f"[MTSAI] Models: status={response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"[MTSAI] Failed to get models: {e}")
            return []

    def health_check(self) -> bool:
        """Проверка доступности API через get_models."""
        return len(self.get_models()) > 0