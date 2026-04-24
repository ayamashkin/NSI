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
    base_url: str
    api_key_file: Optional[str] = None
    api_key: Optional[str] = None
    username: Optional[str] = None  # ← ДОБАВЛЕНО! Для JWT аутентификации OpenWebUI
    password: Optional[str] = None  # Для JWT аутентификации OpenWebUI
    password_file: Optional[str] = None  # Файл с паролем
    auth_url: Optional[str] = None
    scope: Optional[str] = None
    timeout: int = 120
    default_model: Optional[str] = None

    def load_credentials(self) -> None:
        """Загружает ключи, пароли и учетные данные из указанных файлов."""
        logger.debug(f"Loading credentials for {self.base_url}")
        logger.debug(f"  username: {self.username}")
        logger.debug(f"  api_key_file: {self.api_key_file}")
        logger.debug(f"  password_file: {self.password_file}")

        # Загружаем API key из файла
        if self.api_key_file:
            key_path = Path(self.api_key_file)
            if key_path.exists():
                try:
                    self.api_key = key_path.read_text(encoding='utf-8').strip()
                    logger.info(f"Loaded API key from {self.api_key_file}")
                except Exception as e:
                    logger.error(f"Failed to load key from {self.api_key_file}: {e}")
            else:
                logger.warning(f"Key file not found: {self.api_key_file}")

        # Загружаем пароль из файла (для JWT аутентификации OpenWebUI)
        if self.password_file:
            pwd_path = Path(self.password_file)
            if pwd_path.exists():
                try:
                    self.password = pwd_path.read_text(encoding='utf-8').strip()
                    logger.info(f"Loaded password from {self.password_file}")
                except Exception as e:
                    logger.error(f"Failed to load password from {self.password_file}: {e}")
            else:
                logger.warning(f"Password file not found: {self.password_file}")

        logger.debug(f"After loading - api_key present: {bool(self.api_key)}, password present: {bool(self.password)}")

    # Обратная совместимость - старый метод load_key
    def load_key(self) -> Optional[str]:
        """Загружает ключ/логин из указанного файла и пароль если есть."""
        self.load_credentials()
        return self.api_key


@dataclass
class DatabaseConfig:
    """Конфигурация базы данных."""
    path: str = "cache/results.db"
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
    cache_completed_only: bool = True
    retry_errors: bool = True
    retry_ignored: bool = True


@dataclass
class PromptConfig:
    """Конфигурация промпта."""
    id: str
    name: str
    file: str  # Путь к файлу промпта (как в YAML)
    category: str
    keywords: List[str]
    service: str  # "openwebui", "mws", "gigachat"
    model: str
    temperature: float = 0.1
    system_prompt: Optional[str] = None  # Системный промпт для модели

    @property
    def file_path(self) -> str:
        """Свойство для совместимости (возвращает self.file)."""
        return self.file

    def get_api_config(self, settings: 'Settings') -> 'APIConfig':
        """Получает конфиг API для этого промпта."""
        return settings.api[self.service]




@dataclass
class MaskGenerationConfig:
    """Конфигурация генерации масок (fallback при отсутствии prompts.yaml)."""
    default_service: str = "mws"
    default_model: str = "qwen2.5-72b-instruct"
    default_temperature: float = 0.1
    keyword_match_from_name: bool = True
    prompt_template: str = "prompts/templates/mask_generation.txt"
    save_debug_prompts: bool = True
    debug_prompts_dir: str = "prompts/debug"
    deduplicate_by_standard_type: bool = False       # Не создавать дубликаты масок

@dataclass
class OutputConfig:
    """Конфигурация вывода результатов."""
    include_mask_pattern: bool = False               # Добавлять regex в результат
    include_ens_details: bool = True                 # Добавлять ENS имя, fuzzy score
    ens_params_skip_fields: List[str] = field(default_factory=lambda: [
        '_id', '_index', '_source', 'id', 'created_at', 'updated_at', 'hash', 'pattern_hash'
    ])


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
    mask_generation: MaskGenerationConfig = field(default_factory=MaskGenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

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
        logger.debug(f"Loaded config data: {config_data}")

        # Загрузка реестра промптов
        prompts_data = cls._load_yaml(prompts_path)

        # Парсинг API конфигураций
        api_configs = {}
        for name, cfg in config_data.get('api', {}).items():
            logger.info(f"Parsing API config for: {name}")
            logger.debug(f"Raw config: {cfg}")

            # Создаем APIConfig из YAML
            api_cfg = APIConfig(**cfg)
            logger.debug(f"APIConfig created - username: {api_cfg.username}, has api_key: {bool(api_cfg.api_key)}")

            # Загружаем ключи из файлов
            api_cfg.load_credentials()

            api_configs[name] = api_cfg
            auth_type = 'api_key' if api_cfg.api_key else 'jwt' if (api_cfg.username and api_cfg.password) else 'none'
            logger.info(f"API config loaded for {name} - auth: {auth_type}")

        # Парсинг промптов
        prompt_configs = {}
        for pid, pdata in prompts_data.get('prompts', {}).items():
            pdata['id'] = pid
            prompt_configs[pid] = PromptConfig(**pdata)

        return cls(
            api=api_configs,
            database=DatabaseConfig(**config_data.get('database', {})),
            processing=ProcessingConfig(**config_data.get('processing', {})),
            prompts=prompt_configs,
            mask_generation=MaskGenerationConfig(**config_data.get('mask_generation', {})),
            output=OutputConfig(**config_data.get('output', {}))
        )

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        """Загрузка YAML файла."""
        file_path = Path(path)
        if not file_path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            logger.debug(f"YAML loaded from {path}: {data}")
            return data

    def get_api_key(self, service_name: str) -> Optional[str]:
        """Получает ключ API по имени сервиса."""
        if service_name in self.api:
            return self.api[service_name].api_key
        return None

    def get_api_password(self, service_name: str) -> Optional[str]:
        """Получает пароль API по имени сервиса."""
        if service_name in self.api:
            return self.api[service_name].password
        return None

    def get_api_username(self, service_name: str) -> Optional[str]:
        """Получает username API по имени сервиса."""
        if service_name in self.api:
            return self.api[service_name].username
        return None

    def get_api_scope(self, service_name: str) -> Optional[str]:
        """Получает OAuth scope по имени сервиса (для GigaChat)."""
        if service_name in self.api:
            return self.api[service_name].scope
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
            config.load_credentials()
        logger.info("API keys and credentials reloaded")


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