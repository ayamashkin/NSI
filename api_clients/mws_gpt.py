# api_clients/mws_gpt.py
import requests
import json
import logging
from typing import Dict, Any
from api_clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class MWSGPTClient(BaseLLMClient):
    def complete(self, prompt: str, model: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        MWS Cloud GPT API
        Документация: https://mws.ru/docs/cloud-platform/gpt/general/quickstart-gpt.html
        """
        url = f"{self.base_url}/chat/completions"

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
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            content = data['choices'][0]['message']['content']

            # Парсинг JSON аналогично OpenWebUI
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0]
                else:
                    json_str = content

                parsed = json.loads(json_str.strip())
                return {
                    "success": True,
                    "content": parsed,
                    "raw": content,
                    "model": model
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "content": None,
                    "raw": content,
                    "error": "JSON parse error",
                    "model": model
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
        try:
            # Проверка через получение списка моделей или простой запрос
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False