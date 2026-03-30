"""
Configuration Module
Единый источник настроек с загрузкой API ключей из внешних файлов.
"""

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Конфигурация API."""
    base_url: str
    api_key_file: Optional[str] = None
    api_key: Optional[str] = None  # Загружается из файла
    timeout: int = 120
    default_model: Optional[str] = None

    def load_key(self) -> Optional[str]:
        """Загружает ключ из указанного файла."""
        if self.api_key_file:
            key_path = Path(self.api_key_file)
            if key_path.exists():
                try:
                    self.api_key = key_path.read_text(encoding='utf-8').strip()
                    logger.debug(f"Loaded API key from {self.api_key_file}")
                    return self.api_key
                except Exception as e:
                    logger.error(f"Failed to load key from {self.api_key_file}: {e}")
            else:
                logger.warning(f"Key file not found: {self.api_key_file}")
        return self.api_key


@dataclass
class DatabaseConfig:
    """Конфигурация базы данных."""
    path: str = "results.db"
    backup_enabled: bool = True
    backup_interval: int = 1000


@dataclass
class ProcessingConfig:
    """Конфигурация обработки."""
    default_workers: int = 4
    default_timeout: int = 120
    batch_size: int = 100
    retry_attempts: int = 3
    retry_delay: int = 5


@dataclass
class PromptConfig:
    """Конфигурация промпта."""
    id: str
    name: str
    file: str  # Путь к файлу промпта (как в YAML)
    category: str
    keywords: List[str]
    service: str  # "openwebui" или "mws"
    model: str
    temperature: float = 0.1

    @property
    def file_path(self) -> str:
        """Свойство для совместимости (возвращает self.file)."""
        return self.file

    def get_api_config(self, settings: 'Settings') -> 'APIConfig':
        """Получает конфиг API для этого промпта."""
        return settings.api[self.service]


@dataclass
class Settings:
    """
    Единый класс настроек.
    Загружает конфигурацию из YAML и подгружает API ключи из файлов.
    """

    api: Dict[str, APIConfig] = field(default_factory=dict)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    prompts: Dict[str, PromptConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: str = "config/config.yaml", 
             prompts_path: str = "config/prompts.yaml") -> 'Settings':
        """
        Загрузка настроек из YAML файлов.

        Args:
            config_path: Путь к основному конфигу
            prompts_path: Путь к реестру промптов

        Returns:
            Объект Settings с загруженными значениями
        """
        # Загрузка основного конфига
        config_data = cls._load_yaml(config_path)

        # Загрузка реестра промптов
        prompts_data = cls._load_yaml(prompts_path)

        # Парсинг API конфигураций
        api_configs = {}
        for name, cfg in config_data.get('api', {}).items():
            api_cfg = APIConfig(**cfg)
            api_cfg.load_key()  # Загружаем ключ из файла
            api_configs[name] = api_cfg

        # Парсинг промптов
        prompt_configs = {}
        for pid, pdata in prompts_data.get('prompts', {}).items():
            pdata['id'] = pid
            prompt_configs[pid] = PromptConfig(**pdata)

        return cls(
            api=api_configs,
            database=DatabaseConfig(**config_data.get('database', {})),
            processing=ProcessingConfig(**config_data.get('processing', {})),
            prompts=prompt_configs
        )

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        """Загрузка YAML файла."""
        file_path = Path(path)
        if not file_path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def get_api_key(self, service_name: str) -> Optional[str]:
        """Получает ключ API по имени сервиса."""
        if service_name in self.api:
            return self.api[service_name].api_key
        return None

    def get_prompt(self, prompt_id: str) -> Optional[PromptConfig]:
        """Получает конфигурацию промпта по ID."""
        return self.prompts.get(prompt_id)

    def list_prompts(self) -> List[PromptConfig]:
        """Возвращает список всех промптов."""
        return list(self.prompts.values())

    def reload_keys(self):
        """Перезагрузка API ключей из файлов."""
        for config in self.api.values():
            config.load_key()
        logger.info("API keys reloaded")


# Глобальный singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Получение настроек (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings() -> Settings:
    """Принудительная перезагрузка настроек."""
    global _settings
    _settings = Settings.load()
    return _settings
