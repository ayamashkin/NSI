"""
Coating LLM Client Module
Клиент для LLM-запросов правил покрытий.
Использует существующие клиенты (openwebui, mws, gigachat).

LAST_FIX: 2026-05-06 14:30 — LLM query for coating rules per standard+material
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class CoatingLLMClient:
    """
    Клиент для запросов к LLM о правилах покрытий.
    Использует существующих провайдеров из llm_mask_generator.
    """

    def __init__(self, mask_generator: Any):
        """
        Args:
            mask_generator: Экземпляр LLMMaskGenerator с инициализированными клиентами
        """
        self.mask_generator = mask_generator

    def query_coatings(
        self,
        prompt: str,
        standard: str,
        item_type: str,
        temperature: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """
        Запрос к LLM для получения правил покрытий.

        Returns:
            {"material_coating_map": {...}, "auto_substitution": [...]} или None
        """
        logger.info(f"[CoatingLLM] Querying LLM for {standard}/{item_type}")

        try:
            # Используем первый доступный клиент
            response = self._call_llm(prompt, temperature)
            if not response:
                return None

            # Парсим JSON из ответа
            result = self._parse_coating_response(response)
            if result:
                logger.info(f"[CoatingLLM] Parsed coating rules: {result.get('material_coating_map', {})}")
            return result

        except Exception as e:
            logger.error(f"[CoatingLLM] Query failed: {e}")
            return None

    def _call_llm(self, prompt: str, temperature: float) -> Optional[str]:
        """Вызов LLM через mask_generator клиенты."""
        from llm_mask_generator import LLMProvider

        # Пробуем провайдеры по порядку
        for provider_name in ['openwebui', 'mws', 'gigachat']:
            try:
                provider = LLMProvider(provider_name)
                if not self.mask_generator._has_client(provider):
                    continue

                client = self.mask_generator._get_client(provider)

                # Определяем модель из конфига
                prompt_cfg = self.mask_generator._get_prompt_config('hardware')
                model = getattr(prompt_cfg, 'model', None) if prompt_cfg else None
                if not model:
                    model = 'qwen2.5:7b' if provider_name == 'openwebui' else 'qwen2.5-72b-instruct'

                logger.debug(f"[CoatingLLM] Calling {provider_name}/{model}")

                response = client.complete(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    system_prompt="Ты — эксперт по техническим стандартам. Отвечай ТОЛЬКО JSON без пояснений."
                )

                if isinstance(response, dict):
                    raw = response.get('raw') or response.get('content')
                    if raw:
                        return str(raw)
                elif isinstance(response, str):
                    return response

            except Exception as e:
                logger.warning(f"[CoatingLLM] Provider {provider_name} failed: {e}")
                continue

        logger.error("[CoatingLLM] All providers failed")
        return None

    @staticmethod
    def _parse_coating_response(raw: str) -> Optional[Dict[str, Any]]:
        """Парсинг JSON с правилами покрытий из ответа LLM."""
        if not raw:
            return None

        # Ищем JSON блок
        # 1. Пробуем найти ```json ... ```
        json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 2. Пробуем найти { ... }
            json_match = re.search(r'(\{.*\})', raw, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = raw

        try:
            result = json.loads(json_str)

            # Валидация структуры
            if 'material_coating_map' not in result:
                logger.warning("[CoatingLLM] Response missing 'material_coating_map'")
                return None

            # Нормализуем: убеждаемся что все coatings — списки строк
            material_map = result['material_coating_map']
            normalized = {}
            for mat, coatings in material_map.items():
                if isinstance(coatings, str):
                    coatings = [coatings]
                if isinstance(coatings, list):
                    normalized[mat] = [str(c) for c in coatings if c]

            result['material_coating_map'] = normalized

            # Нормализуем auto_substitution
            if 'auto_substitution' in result:
                subs = result['auto_substitution']
                if isinstance(subs, dict):
                    result['auto_substitution'] = [subs]
                elif not isinstance(subs, list):
                    result['auto_substitution'] = []
            else:
                result['auto_substitution'] = []

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[CoatingLLM] JSON parse failed: {e}. Raw: {raw[:200]}")
            return None