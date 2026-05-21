"""
Configuration Management Module
Handles loading and validation of application configuration from YAML files.

VERSION: 2026-05-20 2026-05-20 14:46 UTC+3 UTC+3
 + empty_values: Dict[str, List[str]] для validation (auto_validator)
 + api: username/password для OpenWebUI JWT auth
 + mask_generation: default_service для LLMMaskGenerator

LAST_FIXES:
  2026-05-20 2026-05-20 14:46 UTC+3 UTC+3 — Settings: добавлено поле empty_values (Dict[str, List[str]])
    для конфигурации пустых значений в валидации (убран хардкод в auto_validator).
  2026-05-19 22:35 UTC+3 — Settings: добавлено поле default_service для mask_generation.
  2026-05-18 22:40 UTC+3 — Settings: добавлено поле username для APIConfig (OpenWebUI JWT).
  2026-05-18 20:45 UTC+3 — Settings: добавлено поле mask_generation с default_service.
  2026-05-18 19:30 UTC+3 — Settings: добавлено поле default_service.
  2026-05-18 18:15 UTC+3 — Settings: добавлено поле debug_per_parameter.
  2026-05-18 16:00 UTC+3 — Settings: добавлено поле result_db_path.
"""
import os
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    path: str = "cache/results.db"
    connection_string: str = ""


@dataclass
class APIConfig:
    base_url: str = ""
    api_key: str = ""
    api_key_file: str = ""
    username: Optional[str] = None
    password: str = ""
    password_file: str = ""
    timeout: int = 120
    default_model: str = ""

    def __post_init__(self):
        if self.api_key_file:
            key_path = Path(self.api_key_file)
            if key_path.exists():
                self.api_key = key_path.read_text(encoding='utf-8').strip()
                logger.info(f"API key loaded from {self.api_key_file}")
        if self.password_file:
            pwd_path = Path(self.password_file)
            if pwd_path.exists():
                self.password = pwd_path.read_text(encoding='utf-8').strip()
                logger.info(f"Password loaded from {self.password_file}")


@dataclass
class PromptConfig:
    name: str = ""
    category: str = ""
    service: str = ""
    model: str = ""
    keywords: List[str] = field(default_factory=list)
    template: str = ""

    def resolve_service(self, settings: 'Settings') -> str:
        return self.service or settings.default_service

    def resolve_model(self, settings: 'Settings') -> str:
        return self.model or "qwen2.5-72b-instruct"


@dataclass
class MaskGenerationConfig:
    default_service: str = ""
    default_model: str = ""
    prompt_template: str = ""
    save_debug_prompts: bool = False
    debug_prompts_dir: str = "prompts/debug"
    min_examples: int = 10
    activation_threshold: float = 0.85
    retry_threshold: float = 0.50


@dataclass
class OutputConfig:
    format: str = "json"
    path: str = "output"
    include_raw: bool = False
    include_full_request: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/processor.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Settings:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: Dict[str, APIConfig] = field(default_factory=dict)
    prompts: Dict[str, PromptConfig] = field(default_factory=dict)
    mask_generation: MaskGenerationConfig = field(default_factory=MaskGenerationConfig)
    default_service: str = ""
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    empty_values: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> 'Settings':
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        with open(path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}

        # Load API configs
        api_configs = {}
        for name, cfg in config_data.get('api', {}).items():
            api_configs[name] = APIConfig(**cfg)

        # Load prompt configs
        prompt_configs = {}
        for name, cfg in config_data.get('prompts', {}).items():
            prompt_configs[name] = PromptConfig(**cfg)

        # Load mask generation config
        mask_gen_cfg = MaskGenerationConfig(**config_data.get('mask_generation', {}))

        return cls(
            database=DatabaseConfig(**config_data.get('database', {})),
            api=api_configs,
            prompts=prompt_configs,
            mask_generation=mask_gen_cfg,
            default_service=config_data.get('default_service', ''),
            output=OutputConfig(**config_data.get('output', {})),
            logging=LoggingConfig(**config_data.get('logging', {})),
            empty_values=config_data.get('empty_values', {})
        )

    def get_api_config(self, service: str) -> Optional[APIConfig]:
        return self.api.get(service)

    def get_prompt_config(self, prompt_id: str) -> Optional[PromptConfig]:
        return self.prompts.get(prompt_id)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(config_path: str = "config/config.yaml") -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load(config_path)
    return _settings


def reload_settings(config_path: str = "config/config.yaml") -> Settings:
    global _settings
    _settings = Settings.load(config_path)
    logger.info(f"Settings reloaded from {config_path}")
    return _settings


def setup_logging(config_path: str = "config/config.yaml"):
    settings = get_settings(config_path)
    log_config = settings.logging

    log_dir = Path(log_config.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_config.level.upper(), logging.INFO),
        format=log_config.format,
        handlers=[
            logging.FileHandler(log_config.file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging configured: level={log_config.level}, file={log_config.file}")