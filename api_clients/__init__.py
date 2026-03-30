"""
API Clients Package
Клиенты для работы с внешними API.
"""

from .base import BaseLLMClient
from .openwebui import OpenWebUIClient
from .mws_gpt import MWSGPTClient

__all__ = [
    'BaseLLMClient',
    'OpenWebUIClient',
    'MWSGPTClient'
]
