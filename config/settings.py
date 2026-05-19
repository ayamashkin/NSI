#!/usr/bin/env python3
"""
Configuration Module
Единый источник настроек с загрузкой API ключей из внешних файлов.

VERSION: 2026-05-19 22:35 UTC+3
  + MatchingConfig: numeric_field_weight, text_field_weight, default_field_weight,
    length_tolerance, numeric_tolerance, confidence_penalty_per_mismatch, max_confidence_penalty
"""

import yaml
import logging
import logging.handlers
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def _parse_size(size_str: str) -> int:
    """Парсит строку размера в байты: '10MB' -> 10485760."""
    size_str = size_str.strip().upper()
    multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            return int(float(size_str[:-len(suffix)].strip()) * mult)
    return int(size_str)


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""
    level: str = "INFO"
    file: str = "logs/processor.log"
    max_size: str = "10MB"
    backup_count: int = 5
    format: str = DEFAULT_LOG_FORMAT


@dataclass
class APIConfig:
    base_url: str
    api_key_file: Optional[str] = None
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    password_file: Optional[str] = None
    auth_url: Optional[str] = None
    scope: Optional[str] = None
    timeout: int = 120
    default_model: Optional[str] = None

    def __post_init__(self):
        """Нормализация полей после инициализации."""
        if self.base_url:
            self.base_url = self.base_url.strip()
        if self.api_key_file:
            self.api_key_file = self.api_key_file.strip()
        if self.password_file:
            self.password_file = self.password_file.strip()

    def load_credentials(self) -> None:
        """Загружает ключи, пароли и учетные данные из указанных файлов."""
        if self.base_url:
            self.base_url = self.base_url.strip()
        if self.api_key_file:
            self.api_key_file = self.api_key_file.strip()
        if self.password_file:
            self.password_file = self.password_file.strip()

        logger.debug(f"Loading credentials for {self.base_url}")
        logger.debug(f" username: {self.username}")
        logger.debug(f" api_key_file: {self.api_key_file}")
        logger.debug(f" password_file: {self.password_file}")

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
class MatchingConfig:
    """Конфигурация порогов сопоставления."""
    success_threshold: float = 0.7
    fuzzy_threshold: float = 0.6
    v2_exact_threshold: float = 0.99
    coating_similarity_threshold: float = 0.8
    strict_union_keys: bool = False
    debug_per_parameter: bool = True
    fuzzy_params_comparison: bool = False
    # NEW (2026-05-19) — веса параметров при fuzzy matching
    numeric_field_weight: float = 5.0
    text_field_weight: float = 2.0
    default_field_weight: float = 1.0
    # NEW — tolerance для post-validation
    length_tolerance: float = 1.0
    numeric_tolerance: float = 0.01
    # NEW — штрафы к confidence
    confidence_penalty_per_mismatch: float = 0.15
    max_confidence_penalty: float = 0.5


@dataclass
class PromptConfig:
    """Конфигурация промпта."""
    id: str
    name: str
    file: str
    category: str
    keywords: List[str]
    service: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.1
    system_prompt: Optional[str] = None

    @property
    def file_path(self) -> str:
        return self.file

    def resolve_service(self, settings: 'Settings') -> str:
        if self.service:
            return self.service
        mg = settings.mask_generation
        if mg.default_service:
            return mg.default_service
        raise ValueError(f"Service not specified for prompt '{self.id}' and no mask_generation.default_service")

    def resolve_model(self, settings: 'Settings') -> str:
        if self.model:
            return self.model
        service = self.resolve_service(settings)
        api_config = settings.api.get(service)
        if api_config and api_config.default_model:
            return api_config.default_model
        mg = settings.mask_generation
        if mg.default_model:
            return mg.default_model
        raise ValueError(f"Model not specified for prompt '{self.id}' and no defaults configured")

    def get_api_config(self, settings: 'Settings') -> 'APIConfig':
        service = self.resolve_service(settings)
        return settings.api[service]


@dataclass
class MaskGenerationConfig:
    """Конфигурация генерации масок."""
    default_service: str = "mws"
    default_model: str = "qwen2.5-72b-instruct"
    default_temperature: float = 0.1
    keyword_match_from_name: bool = True
    prompt_template: str = "prompts/templates/mask_generation.txt"
    save_debug_prompts: bool = True
    debug_prompts_dir: str = "prompts/debug"
    deduplicate_by_standard_type: bool = False


@dataclass
class OutputConfig:
    """Конфигурация вывода результатов."""
    include_mask_pattern: bool = False
    include_ens_details: bool = True
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
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    coating_rules: Dict[str, Any] = field(default_factory=dict)
    prompts: Dict[str, PromptConfig] = field(default_factory=dict)
    mask_generation: MaskGenerationConfig = field(default_factory=MaskGenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: str = "config/config.yaml",
             prompts_path: str = "config/prompts.yaml") -> 'Settings':
        config_data = cls._load_yaml(config_path)
        logger.debug(f"Loaded config data: {config_data}")

        prompts_data = cls._load_yaml(prompts_path)

        api_configs = {}
        for name, cfg in config_data.get('api', {}).items():
            logger.info(f"Parsing API config for: {name}")
            logger.debug(f"Raw config: {cfg}")
            api_cfg = APIConfig(**cfg)
            logger.debug(f"APIConfig created - username: {api_cfg.username}, has api_key: {bool(api_cfg.api_key)}")
            api_cfg.load_credentials()
            api_configs[name] = api_cfg
            auth_type = 'api_key' if api_cfg.api_key else 'jwt' if (api_cfg.username and api_cfg.password) else 'none'
            logger.info(f"API config loaded for {name} - auth: {auth_type}")

        prompt_configs = {}
        for pid, pdata in prompts_data.get('prompts', {}).items():
            pdata['id'] = pid
            prompt_configs[pid] = PromptConfig(**pdata)

        return cls(
            api=api_configs,
            database=DatabaseConfig(**config_data.get('database', {})),
            processing=ProcessingConfig(**config_data.get('processing', {})),
            matching=MatchingConfig(**config_data.get('matching', {})),
            coating_rules=config_data.get('coating_rules', {}),
            prompts=prompt_configs,
            mask_generation=MaskGenerationConfig(**config_data.get('mask_generation', {})),
            output=OutputConfig(**config_data.get('output', {})),
            logging=LoggingConfig(**config_data.get('logging', {}))
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
        if service_name in self.api:
            return self.api[service_name].api_key
        return None

    def get_api_password(self, service_name: str) -> Optional[str]:
        if service_name in self.api:
            return self.api[service_name].password
        return None

    def get_api_username(self, service_name: str) -> Optional[str]:
        if service_name in self.api:
            return self.api[service_name].username
        return None

    def get_api_scope(self, service_name: str) -> Optional[str]:
        if service_name in self.api:
            return self.api[service_name].scope
        return None

    def get_prompt(self, prompt_id: str) -> Optional[PromptConfig]:
        return self.prompts.get(prompt_id)

    def list_prompts(self) -> List[PromptConfig]:
        return list(self.prompts.values())

    def reload_keys(self):
        for config in self.api.values():
            config.load_credentials()
        logger.info("API keys and credentials reloaded")


def setup_logging(config_path: str = "config/config.yaml") -> None:
    """
    Настройка логирования из конфигурационного файла.
    """
    config_data = Settings._load_yaml(config_path)
    logging_cfg = LoggingConfig(**config_data.get('logging', {}))

    level = getattr(logging, logging_cfg.level.upper(), logging.INFO)

    log_path = Path(logging_cfg.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(logging_cfg.format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    max_bytes = _parse_size(logging_cfg.max_size)
    file_handler = logging.handlers.RotatingFileHandler(
        logging_cfg.file,
        maxBytes=max_bytes,
        backupCount=logging_cfg.backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.getLogger("config.settings").info(
        f"Logging configured: level={logging_cfg.level}, file={logging_cfg.file}"
    )


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