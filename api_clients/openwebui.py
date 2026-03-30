"""
OpenWebUI API Client Module
Клиент для работы с локальным OpenWebUI API.
"""

import requests
import json
import logging
from typing import Dict, Any, Optional, List
from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenWebUIClient(BaseLLMClient):
    """Клиент для OpenWebUI API."""

    def complete(self, prompt: str, model: str, temperature: float = 0.1) -> Dict[str, Any]:
        """Отправка запроса на генерацию через OpenWebUI."""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Вы - эксперт по техническим стандартам ГОСТ."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "stream": False
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
            parsed = self._extract_json_from_response(content)

            return {
                "success": parsed is not None,
                "content": parsed,
                "raw": content,
                "model": model,
                "error": None if parsed else "JSON parse error"
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"OpenWebUI request failed: {e}")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": str(e),
                "model": model
            }

    def health_check(self) -> bool:
        """Проверка доступности OpenWebUI API."""
        try:
            response = requests.get(
                f"{self.base_url}/api/models",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"OpenWebUI health check failed: {e}")
            return False

    def get_models(self) -> List[str]:
        """Получение списка доступных моделей из OpenWebUI."""
        try:
            response = requests.get(
                f"{self.base_url}/api/models",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for model in data.get('data', []):
                if isinstance(model, dict) and 'id' in model:
                    models.append(model['id'])
                elif isinstance(model, str):
                    models.append(model)

            return sorted(models)
        except Exception as e:
            logger.error(f"Failed to get models from OpenWebUI: {e}")
            return []