"""
Configuration Module
Загружает настройки из YAML и подставляет API ключи из файлов.
"""

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
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
                    logger.debug(f"Loaded key from {self.api_key_file}")
                    return self.api_key
                except Exception as e:
                    logger.error(f"Failed to load key: {e}")
        return None


@dataclass
class Settings:
    api: Dict[str, APIConfig] = field(default_factory=dict)
    database_path: str = "results.db"
    default_workers: int = 4
    batch_size: int = 100

    @classmethod
    def load(cls, path: str = "config/config.yaml") -> "Settings":
        """Загружает конфигурацию из YAML."""
        config_path = Path(path)

        if not config_path.exists():
            logger.warning(f"Config not found: {path}, using defaults")
            return cls()

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Парсим API конфиги
        api_configs = {}
        for name, cfg in data.get('api', {}).items():
            api_configs[name] = APIConfig(**cfg)
            api_configs[name].load_key()  # Загружаем ключ из файла

        return cls(
            api=api_configs,
            database_path=data.get('database', {}).get('path', 'results.db'),
            default_workers=data.get('processing', {}).get('default_workers', 4),
            batch_size=data.get('processing', {}).get('batch_size', 100)
        )

    def get_api_key(self, name: str) -> Optional[str]:
        """Получает ключ API по имени."""
        if name in self.api:
            return self.api[name].api_key
        return None


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings