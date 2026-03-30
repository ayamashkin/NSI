"""
Configuration Module
Единый источник настроек с загрузкой API ключей из внешних файлов.
"""

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

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
        """Загрузка ключа из файла если указан."""
        if self.api_key_file and Path(self.api_key_file).exists():
            try:
                key = Path(self.api_key_file).read_text(encoding='utf-8').strip()
                self.api_key = key
                logger.debug(f"Loaded API key from {self.api_key_file}")
                return key
            except Exception as e:
                logger.warning(f"Failed to load key from {self.api_key_file}: {e}")
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
class PromptsConfig:
    """Конфигурация промптов."""
    directory: str = "prompts/templates"
    registry: str = "config/prompts.yaml"


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""
    level: str = "INFO"
    file: str = "logs/processor.log"
    max_size: str = "10MB"
    backup_count: int = 5


@dataclass
class Settings:
    """
    Единый класс настроек.
    Загружает конфигурацию из YAML и подгружает API ключи из файлов.
    """

    api: Dict[str, APIConfig] = field(default_factory=dict)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Инициализация после создания объекта."""
        # Загружаем ключи для всех API
        for name, config in self.api.items():
            if isinstance(config, APIConfig):
                config.load_key()

    @classmethod
    def load(cls, path: str = "config/config.yaml") -> "Settings":
        """
        Загрузка настроек из YAML файла.

        Args:
            path: Путь к YAML конфигу

        Returns:
            Объект Settings с загруженными значениями
        """
        config_path = Path(path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Парсинг API конфигураций
            api_configs = {}
            if 'api' in data:
                for name, cfg in data['api'].items():
                    api_configs[name] = APIConfig(**cfg)

            # Создание объекта
            settings = cls(
                api=api_configs,
                database=DatabaseConfig(**data.get('database', {})),
                processing=ProcessingConfig(**data.get('processing', {})),
                prompts=PromptsConfig(**data.get('prompts', {})),
                logging=LoggingConfig(**data.get('logging', {}))
            )

            logger.info(f"Loaded configuration from {path}")
            return settings

        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return cls()

    def get_api_config(self, name: str) -> Optional[APIConfig]:
        """Получение конфигурации API по имени."""
        return self.api.get(name)

    def reload_keys(self):
        """Перезагрузка API ключей из файлов."""
        for config in self.api.values():
            if isinstance(config, APIConfig):
                config.load_key()
        logger.info("API keys reloaded")

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь (для отладки, без ключей)."""
        result = asdict(self)
        # Маскируем ключи
        for api_name, api_cfg in result.get('api', {}).items():
            if api_cfg.get('api_key'):
                api_cfg['api_key'] = '***HIDDEN***'
        return result


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
