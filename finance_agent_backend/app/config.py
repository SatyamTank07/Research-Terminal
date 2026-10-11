import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class Settings:
    PROJECT_NAME: str = "Finance Agent API"

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_BASE_URL: str = os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    LANGFUSE_HOST: str = LANGFUSE_BASE_URL

    # PostgreSQL Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "finance_agent_db")

    # Default Authorized Single User
    DEFAULT_USER_USERNAME: str = os.getenv("DEFAULT_USER_USERNAME", "finance_user")
    DEFAULT_USER_EMAIL: str = os.getenv("DEFAULT_USER_EMAIL", "user@financeagent.local")
    DEFAULT_USER_FULL_NAME: str = os.getenv("DEFAULT_USER_FULL_NAME", "Finance User")

    # Valuation market assumptions (fallbacks when live data is unavailable)
    DEFAULT_RISK_FREE_RATE: float = float(os.getenv("DEFAULT_RISK_FREE_RATE", "0.042"))
    DEFAULT_EQUITY_RISK_PREMIUM: float = float(os.getenv("DEFAULT_EQUITY_RISK_PREMIUM", "0.050"))
    DEFAULT_CREDIT_SPREAD: float = float(os.getenv("DEFAULT_CREDIT_SPREAD", "0.0125"))

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


settings = Settings()
