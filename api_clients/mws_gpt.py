"""
MWS GPT API Client Module
Клиент для работы с MWS Cloud GPT API.
"""

import requests
import json
import logging
from typing import Dict, Any, Optional
from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class MWSGPTClient(BaseLLMClient):
    """
    Клиент для MWS Cloud GPT API.

    Использует OpenAI-compatible API от MWS.
    Документация: https://mws.ru/docs/cloud-platform/gpt/
    """

    def complete(self, prompt: str, model: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Отправка запроса на генерацию через MWS GPT.

        Args:
            prompt: Текст промпта
            model: Название модели (например, "gpt-oss-120b")
            temperature: Температура генерации

        Returns:
            Результат генерации
        """
        url = f"{self.base_url}/chat/completions"

        # Добавить логирование
        #logger.info(f"MWS API URL: {url}")
        #logger.info(f"MWS base_url from config: {self.base_url}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }

        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            content = data['choices'][0]['message']['content']

            # Пытаемся извлечь JSON
            parsed = self._extract_json_from_response(content)

            return {
                "success": parsed is not None,
                "content": parsed,
                "raw": content,
                "model": model,
                "error": None if parsed else "JSON parse error"
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"MWS GPT request failed: {e}")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": str(e),
                "model": model
            }

    def health_check(self) -> bool:
        """Проверка доступности MWS GPT API."""
        try:
            # Пробуем получить список моделей
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"MWS GPT health check failed: {e}")
            return False
