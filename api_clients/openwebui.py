"""
OpenWebUI API Client Module
Клиент для работы с локальным OpenWebUI API.
Поддерживает аутентификацию через API key или JWT token (login/password).
"""

import requests
import json
import logging
from typing import Dict, Any, Optional, List
from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenWebUIClient(BaseLLMClient):
    """
    Клиент для OpenWebUI API.

    Поддерживает два режима аутентификации:
    1. API Key (традиционный) - через api_key в конструкторе
    2. JWT Token (login/password) - через username/password в конструкторе

    При использовании JWT сначала выполняется signin для получения токена,
    который затем используется во всех запросах.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Инициализация клиента OpenWebUI.

        Args:
            base_url: URL OpenWebUI API (например, "https://webui.example.com/api")
            api_key: API ключ (если используется аутентификация по ключу)
            username: Email/логин (если используется аутентификация по паролю)
            password: Пароль (если используется аутентификация по паролю)
            timeout: Таймаут запросов в секундах

        Note:
            Нужно указать либо api_key, либо username+password, но не оба сразу.
            При использовании username+password клиент автоматически получит JWT токен.
        """
        # Вызываем конструктор базового класса с правильными параметрами
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            username=username,
            password=password
        )

        self._jwt_token: Optional[str] = None
        self._auth_method: str = "unknown"

        # Определяем метод аутентификации
        if api_key:
            self._auth_method = "api_key"
            logger.info("OpenWebUI client initialized with API key auth")
        elif username and password:
            self._auth_method = "jwt"
            logger.info(f"OpenWebUI client initialized with JWT auth (user: {username})")
        else:
            logger.warning("OpenWebUI client initialized without authentication!")

    def _ensure_authenticated(self) -> bool:
        """
        Проверяет и обеспечивает наличие валидного токена аутентификации.

        Returns:
            True если аутентификация успешна, False в противном случае
        """
        if self._auth_method == "api_key":
            return bool(self.api_key)

        if self._auth_method == "jwt":
            # Если JWT токен ещё не получен или истёк - получаем новый
            if not self._jwt_token:
                return self._authenticate_jwt()
            return True

        return False

    def _authenticate_jwt(self) -> bool:
        """
        Аутентификация через login/password для получения JWT токена.

        Returns:
            True если аутентификация успешна, False в противном случае
        """
        if not self.username or not self.password:
            logger.error("Cannot authenticate: username and password required")
            return False

        auth_url = f"{self.base_url}/v1/auths/signin"

        payload = {
            "email": self.username,
            "password": self.password
        }

        try:
            logger.debug(f"Authenticating user {self.username} at {auth_url}")
            response = requests.post(
                auth_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Извлекаем токен из ответа
            token = data.get("token")
            if not token:
                logger.error("Authentication response missing token field")
                return False

            self._jwt_token = token
            logger.info(f"Successfully authenticated user {self.username}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"JWT authentication failed: {e}")
            return False
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse authentication response: {e}")
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Формирует заголовки авторизации для запросов.

        Returns:
            Словарь с заголовками, включая Authorization
        """
        headers = {"Content-Type": "application/json"}

        if self._auth_method == "api_key" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self._auth_method == "jwt" and self._jwt_token:
            headers["Authorization"] = f"Bearer {self._jwt_token}"

        return headers

    def complete(self, prompt: str, model: str, temperature: float = 0.1,
                 system_prompt: Optional[str] = None, stream: bool = False) -> Dict[str, Any]:
        """
        Отправка запроса на генерацию через OpenWebUI.
        """
        # Проверяем аутентификацию
        if not self._ensure_authenticated():
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": "Authentication failed - check credentials",
                "model": model
            }

        url = f"{self.base_url}/chat/completions"
        headers = self._get_auth_headers()

        # Используем переданный system_prompt или значение по умолчанию
        system_content = system_prompt if system_prompt else "Вы - эксперт по техническим стандартам ГОСТ."

        # Формируем сообщения
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }

        try:
            logger.debug(f"Sending request to {url}")
            logger.debug(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            # Логируем ответ для отладки
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response text (first 500 chars): {response.text[:500]}")

            # Если получили 401, возможно JWT истёк
            if response.status_code == 401 and self._auth_method == "jwt":
                logger.warning("JWT token expired or invalid, re-authenticating...")
                self._jwt_token = None
                if self._authenticate_jwt():
                    headers = self._get_auth_headers()
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )

            # Проверяем статус ответа
            if response.status_code == 400:
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', error_data)
                    logger.error(f"OpenWebUI 400 Bad Request: {error_detail}")
                except:
                    logger.error(f"OpenWebUI 400 Bad Request: {response.text[:500]}")

                return {
                    "success": False,
                    "content": None,
                    "raw": response.text,
                    "error": f"400 Bad Request - Model '{model}' not found or invalid parameters",
                    "model": model
                }

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
            # Проверяем аутентификацию перед health check
            if not self._ensure_authenticated():
                return False

            headers = self._get_auth_headers()
            response = requests.get(
                f"{self.base_url}/api/models",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"OpenWebUI health check failed: {e}")
            return False

    def get_models(self) -> List[str]:
        """Получение списка доступных моделей из OpenWebUI."""
        try:
            # Проверяем аутентификацию перед запросом
            if not self._ensure_authenticated():
                logger.error("Cannot get models: authentication failed")
                return []

            headers = self._get_auth_headers()

            # Используем правильный endpoint - base_url уже содержит /api
            # Пробуем /api/models (если base_url заканчивается на /api)
            # или /models (если base_url без /api)
            if self.base_url.rstrip('/').endswith('/api'):
                url = f"{self.base_url}/models"
            else:
                url = f"{self.base_url}/api/models"

            logger.debug(f"Fetching models from: {url}")

            response = requests.get(
                url,
                headers=headers,
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

    def get_auth_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о текущем методе аутентификации.

        Returns:
            Словарь с информацией о методе аутентификации
        """
        return {
            "method": self._auth_method,
            "username": self.username if self._auth_method == "jwt" else None,
            "has_api_key": bool(self.api_key) if self._auth_method == "api_key" else None,
            "is_authenticated": bool(self._jwt_token) if self._auth_method == "jwt" else bool(self.api_key)
        }