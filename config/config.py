from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    model_dir: str = "models"
    api_key: str | None = None
    allowed_origins: str = "*"
    redis_url: str | None = None

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)


def get_settings() -> Settings:
    return Settings()
