import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

logger = logging.getLogger("finance_agent.database")

# Create SQLAlchemy engine with pool_pre_ping for resilient connections
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that provides a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> dict:
    """Executes a simple query to verify PostgreSQL connectivity."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            return {
                "status": "connected",
                "database": settings.database_url.split("@")[-1] if "@" in settings.database_url else "configured",
                "ping": result == 1,
            }
    except Exception as e:
        logger.error(f"PostgreSQL connection check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def init_db():
    """Creates database tables and seeds the default single user if absent."""
    import app.models as models

    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        existing_user = db.query(models.User).first()
        if existing_user is None:
            username = settings.DEFAULT_USER_USERNAME
            email = settings.DEFAULT_USER_EMAIL
            full_name = settings.DEFAULT_USER_FULL_NAME

            logger.info(f"Seeding default single user '{username}'...")
            default_user = models.User(
                username=username,
                email=email,
                full_name=full_name,
                is_active=True,
            )
            db.add(default_user)
            db.commit()
            db.refresh(default_user)
            logger.info(f"Default single user seeded successfully with ID {default_user.id}.")
        else:
            logger.info(f"Active single user found: '{existing_user.username}' (ID {existing_user.id}).")
    except Exception as e:
        logger.error(f"Failed to initialize or seed database: {e}")
        db.rollback()
        raise e
    finally:
        db.close()


def get_single_user(db):
    """Retrieves the single configured user and ensures they are active."""
    import app.models as models

    user = db.query(models.User).first()
    if not user:
        return None
    return user
