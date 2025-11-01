# src/settings.py
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # core
    APP_NAME: str = Field(default="Magic Bot")
    ENV: str = Field(default=os.getenv("APP_ENV", "dev"))
    DEBUG: bool = Field(default=True)

    # example secret
    OPENAI_API_KEY: str | None = None

    # read root-level .env.dev (as you want)
    model_config = SettingsConfigDict(
        env_file=".env.dev",
        extra="ignore",
    )

    # keep backward-compat for app.py
    @property
    def app_name(self) -> str:
        return self.APP_NAME


settings = Settings()
