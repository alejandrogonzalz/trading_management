import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Spot Credentials
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "").strip().strip("'").strip('"')
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "").strip().strip("'").strip('"')
    
    # Lead Trading / Futures Credentials
    LEAD_API_KEY: str = (os.getenv("LEAD_API_KEY") or os.getenv("BINANCE_COPY_TRADING_KEY") or "").strip().strip("'").strip('"')
    LEAD_API_SECRET: str = (os.getenv("LEAD_API_SECRET") or os.getenv("BINANCE_COPY_TRADING_SECRET") or "").strip().strip("'").strip('"')
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LANGGRAPH_URL: str = os.getenv("LANGGRAPH_URL", "http://localhost:2024")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:14b")
    SCANNER_INTERVAL_MINUTES: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Debug: Masked log to verify key loading
if not settings.BINANCE_API_KEY:
    print("WARNING: BINANCE_API_KEY is empty!")
else:
    print(f"Spot API Key loaded: {settings.BINANCE_API_KEY[:4]}...{settings.BINANCE_API_KEY[-4:]}")

if not settings.LEAD_API_KEY:
    print("INFO: LEAD_API_KEY not configured. Lead Trading features will be restricted.")
else:
    print(f"Lead API Key loaded: {settings.LEAD_API_KEY[:4]}...{settings.LEAD_API_KEY[-4:]}")
