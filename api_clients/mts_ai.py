"""
MTS AI API Client
OpenAI-compatible API для модели cotype_pro_2.5.

API Endpoint: https://demo6-fundres.dev.mts.ai/
Auth: API_KEY в заголовке Authorization: Bearer {API_KEY}
Docs: https://demo6-fundres.dev.mts.ai/

LAST_FIXES:
  2026-05-20 2026-05-20 12:47 UTC+3 UTC+3 — chat_completion: возвращает tokens_prompt/tokens_completion
    из data["usage"] (prompt_tokens/completion_tokens) для совместимости с
    LLMMaskGenerator._call_llm и Excel-статистики.
  2026-05-18 11:16 UTC+3 — _extract_json_from_text: поддержка markdown ```python + balanced braces
  2026-05-18 10:35 UTC+3 — complete()/chat_completion() возвращают Dict[str, Any]
                           (success, content, raw, error, model) для совместимости с LLMMaskGenerator
  2026-05-07 13:30 UTC+3 — создание клиента по аналогии с mws_gpt
"""
import json
import logging
import re
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path

from api_clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Извлекает JSON-объект из текста с markdown-блоками или inline JSON.
    Пробует несколько стратегий: markdown code blocks (json/python/none), balanced braces.
    """
    if not text:
        return None

    # Стратегия 1: Markdown code block ```json ... ``` / ```python ... ``` / ``` ... ```
    for lang in [r'(?:json)?', r'(?:python)?', r'']:
        pattern = rf'```{lang}\s*(.*?)\s*```'
        md_json = re.search(pattern, text, re.DOTALL)
        if md_json:
            try:
                return json.loads(md_json.group(1))
            except json.JSONDecodeError:
                pass

    # Стратегия 2: Найти первый {...} с балансом скобок (игнорируем содержимое строк)
    for start in re.finditer(r'(?m)^\s*\{', text):
        pos = start.start()
        brace_count = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[pos:], start=pos):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = text[pos:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break  # пробуем следующий match
        # если brace_count не 0 — не валидный JSON, идём дальше

    # Стратегия 3: Простой fallback — первый {...}
    simple = re.search(r'\{.*?\}', text, re.DOTALL)
    if simple:
        try:
            return json.loads(simple.group())
        except json.JSONDecodeError:
            pass

    return None


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

    def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Реализация abstract method из BaseLLMClient.
        Возвращает Dict с полями: success, content, raw, error, model, tokens_prompt, tokens_completion
        (совместимо с LLMMaskGenerator._call_llm).
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
    ) -> Dict[str, Any]:
        """
        Отправка запроса к /v1/chat/completions.

        Returns:
            Dict[str, Any]: {
                "success": bool,
                "content": Any | None,   # распарсенный JSON (если есть)
                "raw": str | None,       # сырой текст ответа
                "error": str | None,
                "model": str,
                "tokens_prompt": int | None,
                "tokens_completion": int | None
            }
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
            tokens_prompt = usage.get("prompt_tokens")
            tokens_completion = usage.get("completion_tokens")
            logger.info(
                f"[MTSAI] Response: tokens_prompt={tokens_prompt}, "
                f"tokens_completion={tokens_completion}, "
                f"content_length={len(content)}"
            )

            # Извлекаем JSON из ответа (для совместимости с mask generation)
            parsed = _extract_json_from_text(content)
            if parsed is None:
                logger.warning(
                    f"[MTSAI] Could not extract JSON from response (len={len(content)}). "
                    f"Raw preview: {content[:200]!r}"
                )

            return {
                "success": True,
                "content": parsed,
                "raw": content,
                "error": None,
                "model": model,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
            }

        except requests.exceptions.Timeout:
            logger.error(f"[MTSAI] Timeout after {self.timeout}s")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": f"Timeout after {self.timeout}s",
                "model": model,
                "tokens_prompt": None,
                "tokens_completion": None,
            }
        except requests.exceptions.HTTPError as e:
            logger.error(f"[MTSAI] HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                "model": model,
                "tokens_prompt": None,
                "tokens_completion": None,
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"[MTSAI] Request error: {e}")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": str(e),
                "model": model,
                "tokens_prompt": None,
                "tokens_completion": None,
            }
        except Exception as e:
            logger.error(f"[MTSAI] Unexpected error: {e}")
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": str(e),
                "model": model,
                "tokens_prompt": None,
                "tokens_completion": None,
            }

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