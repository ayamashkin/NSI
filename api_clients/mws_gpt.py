"""
MWS GPT API Client Module
Клиент для работы с MWS Cloud GPT API.

LAST_FIXES:
  2026-05-20 2026-05-20 12:48 UTC+3 UTC+3 — complete(): возвращает tokens_prompt/tokens_completion
    из data["usage"] для совместимости с LLMMaskGenerator._call_llm.
"""
import requests
import json
import logging
from typing import Dict, Any, Optional, List
from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class MWSGPTClient(BaseLLMClient):
    """Клиент для MWS Cloud GPT API."""

    def complete(self, prompt: str, model: str, temperature: float = 0.1,
                 system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Отправка запроса на генерацию через MWS GPT."""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Формируем сообщения
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
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
            parsed = self._extract_json_from_response(content)

            # FIX: извлекаем usage/tokens
            usage = data.get("usage", {})
            tokens_prompt = usage.get("prompt_tokens")
            tokens_completion = usage.get("completion_tokens")

            return {
                "success": parsed is not None,
                "content": parsed,
                "raw": content,
                "model": model,
                "error": None if parsed else "JSON parse error",
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"MWS GPT request failed: {e}")
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
        """Проверка доступности MWS GPT API."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"MWS GPT health check failed: {e}")
            return False

    def get_models(self) -> List[str]:
        """Получение списка доступных моделей из MWS GPT."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for model in data.get('data', []):
                if isinstance(model, dict) and 'id' in model:
                    models.append(model['id'])

            return sorted(models)
        except Exception as e:
            logger.error(f"Failed to get models from MWS GPT: {e}")
            return []