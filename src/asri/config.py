"""Configuration management for ASRI."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "postgresql+asyncpg://asri_user:asri_dev_password@localhost:5432/asri"
    database_sync_url: str = "postgresql://asri_user:asri_dev_password@localhost:5432/asri"

    # API Keys
    defillama_api_key: str | None = None
    messari_api_key: str | None = None
    token_terminal_api_key: str | None = None
    fred_api_key: str | None = None

    # Application
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Rate Limiting
    rate_limit_free: int = 100
    rate_limit_pro: int = 5000

    # Scheduler
    scheduler_enabled: bool = True
    daily_update_hour: int = 1


settings = Settings()
