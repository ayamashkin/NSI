# api_clients/openwebui.py
import requests
import json
import logging
from typing import Dict, Any, Optional
from api_clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenWebUIClient(BaseLLMClient):
    def complete(self, prompt: str, model: str, temperature: float = 0.1) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Вы - эксперт по техническим стандартам."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            content = data['choices'][0]['message']['content']

            # Пытаемся извлечь JSON из ответа
            try:
                # Ищем JSON в markdown code blocks
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
            logger.error(f"OpenWebUI request failed: {e}")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": str(e),
                "model": model
            }

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/models", timeout=10)
            return response.status_code == 200
        except:
            return False