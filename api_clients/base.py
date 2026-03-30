# api_clients/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLLMClient(ABC):
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key

    @abstractmethod
    def complete(self, prompt: str, model: str, temperature: float = 0.1) -> Dict[str, Any]:
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass