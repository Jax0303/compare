from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

SETTINGS = Settings()
if not SETTINGS.gemini_api_key:
    # Don't raise here to allow --help etc., but methods will check before calling LLM.
    pass
