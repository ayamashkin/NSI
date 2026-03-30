import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenWebUI
    OPENWEBUI_URL: str = "https://webui.game73.ru/api"
    OPENWEBUI_API_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjczM2Y2MTRiLTEzZTgtNDdkYi1iYmY2LWYxNTIyNzczNmJjZSIsImV4cCI6MTc3NzAxMjg5OSwianRpIjoiYmEwMmIxZmQtZjRhOS00NDA2LTgwNGEtMmYzODM4ZDNlM2U4In0.WQiz9f_u1o6TGvBWF2a0py5KVRcQX6Zg0bc_H1vMLTY"

    # MWS Cloud GPT
    MWS_URL: str = "https://gpt.mwsapis.ru/projects/YOUR_PROJECT_NAME/openai/v1"
    MWS_API_KEY: str = "sk-5yljdUG5NSYoHFV3PU-s0Q"

    # Database
    DATABASE_PATH: str = "results.db"

    class Config:
        env_file = ".env"