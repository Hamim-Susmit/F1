import sqlalchemy as sa

from config.config import get_settings


def get_engine() -> sa.Engine:
    """Create a SQLAlchemy engine using environment-driven settings."""
    settings = get_settings()
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required")
    return sa.create_engine(settings.database_url, future=True)
